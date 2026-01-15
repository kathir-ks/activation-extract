#!/usr/bin/env python3
"""
Model Verification Tool - Test weight loading and forward pass correctness

This script verifies:
1. Weight conversion from HF to JAX is correct
2. Forward pass produces same outputs as HF model
3. Activations are extracted correctly
4. Multi-device sharding works properly
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Dict, List, Tuple
import argparse
from pathlib import Path
import json

from qwen2_jax import QwenConfig, QwenModel, convert_hf_to_jax_weights
from qwen2_jax_with_hooks import create_model_with_hooks


class ModelVerifier:
    """Verify JAX model against HuggingFace reference"""

    def __init__(self, model_path: str, verbose: bool = True):
        self.model_path = model_path
        self.verbose = verbose

        # Load HF model and tokenizer
        if self.verbose:
            print(f"Loading HuggingFace model: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            trust_remote_code=True
        )
        self.hf_model.eval()

        # Create JAX config
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.config = QwenConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            max_position_embeddings=hf_config.max_position_embeddings,
            rope_theta=hf_config.rope_theta,
            rms_norm_eps=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings
        )

        if self.verbose:
            print(f"  ✓ Config: {self.config.num_hidden_layers} layers, "
                  f"{self.config.hidden_size} hidden, "
                  f"{self.config.num_attention_heads} heads")

    def verify_weight_conversion(self, tolerance: float = 1e-5) -> Dict[str, bool]:
        """Verify weight conversion is correct"""
        if self.verbose:
            print("\n" + "="*70)
            print("VERIFYING WEIGHT CONVERSION")
            print("="*70)

        # Convert weights
        converted_params = convert_hf_to_jax_weights(self.hf_model, self.config)
        hf_state_dict = self.hf_model.state_dict()

        results = {}

        # Check embedding weights
        if self.verbose:
            print("\n1. Checking embedding weights...")

        hf_embed = hf_state_dict['model.embed_tokens.weight'].numpy()
        jax_embed = converted_params['embed_tokens']['embedding']

        max_diff = np.abs(hf_embed - jax_embed).max()
        results['embeddings'] = max_diff < tolerance

        if self.verbose:
            print(f"   Max diff: {max_diff:.2e} (tolerance: {tolerance:.2e}) - "
                  f"{'✓ PASS' if results['embeddings'] else '✗ FAIL'}")

        # Check layer weights
        if self.verbose:
            print("\n2. Checking layer weights...")

        layer_checks = []
        for layer_idx in range(min(3, self.config.num_hidden_layers)):  # Check first 3 layers
            hf_prefix = f'model.layers.{layer_idx}'
            jax_prefix = f'layers_{layer_idx}'

            # Check Q projection
            hf_q = hf_state_dict[f'{hf_prefix}.self_attn.q_proj.weight'].T.numpy()
            jax_q = converted_params[jax_prefix]['self_attn']['q_proj']['kernel']

            max_diff_q = np.abs(hf_q - jax_q).max()
            layer_ok = max_diff_q < tolerance
            layer_checks.append(layer_ok)

            if self.verbose:
                print(f"   Layer {layer_idx} Q-proj: {max_diff_q:.2e} - "
                      f"{'✓' if layer_ok else '✗'}")

        results['layers'] = all(layer_checks)

        # Check final norm
        if self.verbose:
            print("\n3. Checking final norm...")

        hf_norm = hf_state_dict['model.norm.weight'].numpy()
        jax_norm = converted_params['norm']['weight']

        max_diff = np.abs(hf_norm - jax_norm).max()
        results['final_norm'] = max_diff < tolerance

        if self.verbose:
            print(f"   Max diff: {max_diff:.2e} - "
                  f"{'✓ PASS' if results['final_norm'] else '✗ FAIL'}")

        return results

    def verify_forward_pass(self, test_inputs: List[str] = None,
                           tolerance: float = 1e-3) -> Dict[str, bool]:
        """Verify forward pass matches HF model"""
        if self.verbose:
            print("\n" + "="*70)
            print("VERIFYING FORWARD PASS")
            print("="*70)

        if test_inputs is None:
            test_inputs = [
                "The quick brown fox",
                "Hello world!",
                "Artificial intelligence is"
            ]

        # Create JAX model
        jax_model = QwenModel(self.config)
        converted_params = convert_hf_to_jax_weights(self.hf_model, self.config)
        params = {'params': converted_params}

        results = {}

        for i, text in enumerate(test_inputs):
            if self.verbose:
                print(f"\n{i+1}. Testing: '{text}'")

            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt")
            input_ids_torch = inputs['input_ids']
            input_ids_jax = jnp.array(input_ids_torch.numpy())

            # HF forward pass
            with torch.no_grad():
                hf_outputs = self.hf_model(input_ids_torch)
                hf_logits = hf_outputs.logits.numpy()

            # JAX forward pass
            jax_logits, _ = jax_model.apply(params, input_ids_jax)
            jax_logits_np = np.array(jax_logits)

            # Compare
            max_diff = np.abs(hf_logits - jax_logits_np).max()
            mean_diff = np.abs(hf_logits - jax_logits_np).mean()

            # Check top-5 predictions match
            hf_top5 = np.argsort(hf_logits[0, -1, :])[-5:][::-1]
            jax_top5 = np.argsort(jax_logits_np[0, -1, :])[-5:][::-1]
            top5_match = np.array_equal(hf_top5, jax_top5)

            test_passed = max_diff < tolerance and top5_match
            results[f'test_{i}'] = test_passed

            if self.verbose:
                print(f"   Logits shape: {jax_logits_np.shape}")
                print(f"   Max diff: {max_diff:.2e} (tolerance: {tolerance:.2e})")
                print(f"   Mean diff: {mean_diff:.2e}")
                print(f"   Top-5 predictions match: {top5_match}")
                print(f"   Result: {'✓ PASS' if test_passed else '✗ FAIL'}")

                # Show top prediction
                hf_pred = self.tokenizer.decode([hf_top5[0]])
                jax_pred = self.tokenizer.decode([jax_top5[0]])
                print(f"   HF top pred: '{hf_pred}', JAX top pred: '{jax_pred}'")

        return results

    def verify_activation_extraction(self, layers_to_extract: List[int] = None) -> Dict[str, bool]:
        """Verify activation extraction works correctly"""
        if self.verbose:
            print("\n" + "="*70)
            print("VERIFYING ACTIVATION EXTRACTION")
            print("="*70)

        if layers_to_extract is None:
            layers_to_extract = [0, self.config.num_hidden_layers // 2,
                                self.config.num_hidden_layers - 1]

        # Create model with hooks
        jax_model = create_model_with_hooks(self.config, layers_to_extract)
        converted_params = convert_hf_to_jax_weights(self.hf_model, self.config)
        params = {'params': converted_params}

        # Test input
        text = "Testing activation extraction"
        inputs = self.tokenizer(text, return_tensors="np")
        input_ids = jnp.array(inputs['input_ids'])

        if self.verbose:
            print(f"\nTest input: '{text}'")
            print(f"Extracting layers: {layers_to_extract}")

        # Extract activations
        logits, kv_caches, activations = jax_model.apply(
            params, input_ids, return_activations=True
        )

        results = {}

        # Check activations extracted
        for layer_idx in layers_to_extract:
            layer_key = f'layer_{layer_idx}'
            has_activation = layer_key in activations
            results[f'layer_{layer_idx}_extracted'] = has_activation

            if self.verbose:
                if has_activation:
                    act_shape = activations[layer_key].shape
                    print(f"  ✓ Layer {layer_idx}: shape {act_shape}")
                else:
                    print(f"  ✗ Layer {layer_idx}: NOT extracted")

        # Check activation shapes
        seq_len = input_ids.shape[1]
        expected_shape = (1, seq_len, self.config.hidden_size)

        for layer_idx in layers_to_extract:
            layer_key = f'layer_{layer_idx}'
            if layer_key in activations:
                actual_shape = activations[layer_key].shape
                shape_correct = actual_shape == expected_shape
                results[f'layer_{layer_idx}_shape'] = shape_correct

                if self.verbose and not shape_correct:
                    print(f"  ✗ Layer {layer_idx} shape mismatch: "
                          f"expected {expected_shape}, got {actual_shape}")

        # Check activations are non-zero
        for layer_idx in layers_to_extract:
            layer_key = f'layer_{layer_idx}'
            if layer_key in activations:
                act_mean = np.abs(np.array(activations[layer_key])).mean()
                act_std = np.array(activations[layer_key]).std()
                non_zero = act_mean > 0 and act_std > 0
                results[f'layer_{layer_idx}_nonzero'] = non_zero

                if self.verbose:
                    print(f"  Layer {layer_idx} stats: mean={act_mean:.4f}, std={act_std:.4f}")

        return results

    def verify_sharding(self, mesh_type: str = '2d') -> Dict[str, bool]:
        """Verify model sharding works on multiple devices"""
        if self.verbose:
            print("\n" + "="*70)
            print("VERIFYING SHARDING")
            print("="*70)

        devices = jax.devices()
        num_devices = len(devices)

        if self.verbose:
            print(f"Available devices: {num_devices}")
            for i, device in enumerate(devices):
                print(f"  {i}: {device.device_kind}")

        results = {'num_devices': num_devices}

        if num_devices < 2:
            if self.verbose:
                print("⚠ Need at least 2 devices for sharding test")
            results['sharding_test'] = False
            return results

        # Import sharding utilities
        from jax.sharding import Mesh, PartitionSpec, NamedSharding
        from jax.experimental import mesh_utils

        # Create mesh
        if mesh_type == '1d':
            device_mesh = mesh_utils.create_device_mesh((num_devices,))
            mesh = Mesh(device_mesh, axis_names=('model',))
        elif mesh_type == '2d':
            if num_devices >= 4:
                device_mesh = mesh_utils.create_device_mesh((2, num_devices // 2))
                mesh = Mesh(device_mesh, axis_names=('data', 'model'))
            else:
                device_mesh = mesh_utils.create_device_mesh((1, num_devices))
                mesh = Mesh(device_mesh, axis_names=('data', 'model'))

        if self.verbose:
            print(f"Created {mesh_type.upper()} mesh with axes: {mesh.axis_names}")
            print(f"Mesh shape: {device_mesh.shape}")

        # Create model and shard parameters
        from extract_activations_fineweb_multihost import create_sharding_strategy, shard_params

        jax_model = QwenModel(self.config)
        converted_params = convert_hf_to_jax_weights(self.hf_model, self.config)
        params = {'params': converted_params}

        sharding_rules = create_sharding_strategy(mesh)

        if self.verbose:
            print(f"Sharding parameters with {len(sharding_rules)} rules...")

        with mesh:
            sharded_params = shard_params(params, mesh, sharding_rules)

        # Test forward pass with sharded params
        text = "Testing sharded inference"
        inputs = self.tokenizer(text, return_tensors="np")
        input_ids = jnp.array(inputs['input_ids'])

        try:
            with mesh:
                logits, _ = jax_model.apply(sharded_params, input_ids)

            if self.verbose:
                print(f"✓ Sharded forward pass successful")
                print(f"  Output shape: {logits.shape}")

            results['sharding_test'] = True
        except Exception as e:
            if self.verbose:
                print(f"✗ Sharded forward pass failed: {e}")
            results['sharding_test'] = False

        return results

    def run_all_tests(self, save_report: bool = True) -> Dict:
        """Run all verification tests"""
        print("="*70)
        print("MODEL VERIFICATION SUITE")
        print("="*70)
        print(f"Model: {self.model_path}")
        print(f"JAX devices: {jax.devices()}")
        print("="*70)

        all_results = {}

        # 1. Weight conversion
        weight_results = self.verify_weight_conversion()
        all_results['weight_conversion'] = weight_results

        # 2. Forward pass
        forward_results = self.verify_forward_pass()
        all_results['forward_pass'] = forward_results

        # 3. Activation extraction
        activation_results = self.verify_activation_extraction()
        all_results['activation_extraction'] = activation_results

        # 4. Sharding (if multiple devices available)
        if len(jax.devices()) > 1:
            sharding_results = self.verify_sharding()
            all_results['sharding'] = sharding_results

        # Summary
        print("\n" + "="*70)
        print("VERIFICATION SUMMARY")
        print("="*70)

        all_passed = True
        for category, results in all_results.items():
            category_passed = all(v for v in results.values() if isinstance(v, bool))
            all_passed = all_passed and category_passed
            status = "✓ PASS" if category_passed else "✗ FAIL"
            print(f"{category}: {status}")

        print("="*70)
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print("="*70)

        # Save report
        if save_report:
            report_file = "verification_report.json"
            with open(report_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nReport saved to: {report_file}")

        return all_results


def main():
    parser = argparse.ArgumentParser(description="Verify JAX model against HuggingFace")
    parser.add_argument('--model_path', type=str, default='KathirKs/qwen-2.5-0.5b',
                       help='Path to HuggingFace model')
    parser.add_argument('--test', type=str, choices=['weights', 'forward', 'activations', 'sharding', 'all'],
                       default='all', help='Which test to run')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    parser.add_argument('--save_report', action='store_true', default=True, help='Save report to file')

    args = parser.parse_args()

    verifier = ModelVerifier(args.model_path, verbose=args.verbose)

    if args.test == 'weights':
        results = verifier.verify_weight_conversion()
    elif args.test == 'forward':
        results = verifier.verify_forward_pass()
    elif args.test == 'activations':
        results = verifier.verify_activation_extraction()
    elif args.test == 'sharding':
        results = verifier.verify_sharding()
    else:  # all
        results = verifier.run_all_tests(save_report=args.save_report)

    # Exit with error code if tests failed
    if isinstance(results, dict) and 'weight_conversion' in results:
        # Full test suite
        all_passed = all(
            all(v for v in category_results.values() if isinstance(v, bool))
            for category_results in results.values()
        )
    else:
        # Single test
        all_passed = all(v for v in results.values() if isinstance(v, bool))

    exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
