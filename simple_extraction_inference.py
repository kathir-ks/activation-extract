"""
Simplified Inference with Activation Extraction
Works with existing qwen2_jax.py - extracts activations at key points
"""

import jax
import jax.numpy as jnp
import json
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass
import argparse
from tqdm import tqdm
import os
import pickle
from datetime import datetime

from transformers import AutoTokenizer
from arc_inference_jax import (
    ARCConfig, create_prompts, validate_grid,
    create_solutions, load_jax_model_and_tokenizer
)


@dataclass
class ExtractionConfig:
    """Configuration for activation extraction"""
    extract_activations: bool = True
    layers_to_extract: List[int] = None  # None = all layers
    activations_dir: str = './activations'
    save_every_n_samples: int = 100
    upload_to_cloud: bool = False
    cloud_bucket: Optional[str] = None


class SimpleActivationExtractor:
    """Simple activation extractor that works with any JAX model"""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.activations = []
        self.sample_count = 0
        self.batch_count = 0

        # Create output directory
        os.makedirs(config.activations_dir, exist_ok=True)

        # Metadata
        self.metadata = {
            'start_time': datetime.now().isoformat(),
            'layers_extracted': config.layers_to_extract or 'all',
            'files': []
        }

    def capture_activation(self, task_id: str, sample_idx: int,
                          input_ids: jnp.ndarray, output_logits: jnp.ndarray):
        """
        Capture activation for a single forward pass

        Args:
            task_id: Task identifier
            sample_idx: Sample index
            input_ids: Input token IDs
            output_logits: Output logits from model
        """
        if not self.config.extract_activations:
            return

        # Store activation data
        activation_data = {
            'task_id': task_id,
            'sample_idx': sample_idx,
            'input_shape': input_ids.shape,
            'output_shape': output_logits.shape,
            'output_logits': np.array(output_logits),  # Convert to numpy
            'timestamp': datetime.now().isoformat()
        }

        self.activations.append(activation_data)
        self.sample_count += 1

        # Save periodically
        if self.sample_count >= self.config.save_every_n_samples:
            self.save_batch()

    def save_batch(self):
        """Save current batch of activations"""
        if not self.activations:
            return

        self.batch_count += 1
        filename = f"activations_batch_{self.batch_count:06d}.pkl"
        filepath = os.path.join(self.config.activations_dir, filename)

        # Save to pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.activations, f)

        print(f"✓ Saved {len(self.activations)} activations to {filename}")

        # Update metadata
        self.metadata['files'].append({
            'batch_id': self.batch_count,
            'filename': filename,
            'n_samples': len(self.activations),
            'timestamp': datetime.now().isoformat()
        })

        # Upload to cloud if configured
        if self.config.upload_to_cloud and self.config.cloud_bucket:
            self._upload_file(filepath)

        # Clear activations
        self.activations = []
        self.sample_count = 0

    def _upload_file(self, filepath: str):
        """Upload file to cloud storage"""
        try:
            from google.cloud import storage

            # Parse bucket name and path
            bucket_path = self.config.cloud_bucket.replace('gs://', '')
            parts = bucket_path.split('/', 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ''

            # Upload
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob_name = os.path.join(prefix, os.path.basename(filepath))
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(filepath)

            print(f"  ↑ Uploaded to gs://{bucket_name}/{blob_name}")

        except Exception as e:
            print(f"  ⚠ Upload failed: {e}")

    def finalize(self):
        """Finalize extraction and save metadata"""
        # Save any remaining activations
        if self.activations:
            self.save_batch()

        # Save metadata
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['total_samples'] = sum(f['n_samples'] for f in self.metadata['files'])
        self.metadata['total_batches'] = self.batch_count

        metadata_path = os.path.join(self.config.activations_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Extraction complete!")
        print(f"  Total samples: {self.metadata['total_samples']}")
        print(f"  Total batches: {self.metadata['total_batches']}")
        print(f"  Metadata: {metadata_path}")
        print(f"{'='*70}")


def generate_with_extraction(model, params, input_ids, max_tokens,
                            extractor, task_id, sample_idx):
    """Generate tokens and extract activations"""
    generated_ids = input_ids

    for _ in range(max_tokens):
        # Forward pass
        logits = model.apply(params, generated_ids)

        # Extract activation (output logits)
        if extractor:
            extractor.capture_activation(task_id, sample_idx, generated_ids, logits)

        # Sample next token
        next_token_logits = logits[:, -1, :]
        next_token_id = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        generated_ids = jnp.concatenate([generated_ids, next_token_id], axis=1)

        # Stop if we've generated enough
        if generated_ids.shape[1] > input_ids.shape[1] + max_tokens:
            break

    return generated_ids


def run_inference_with_extraction(
    model_path: str,
    dataset_path: str,
    output_path: str,
    extraction_config: ExtractionConfig,
    max_tasks: int = None,
    batch_size: int = 1  # Process one at a time for simplicity
):
    """
    Run inference with activation extraction

    This is a simplified version that processes sequentially
    For distributed processing, use the full distributed_inference script
    """
    print("="*70)
    print("INFERENCE WITH ACTIVATION EXTRACTION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Extracting: {extraction_config.extract_activations}")
    if extraction_config.extract_activations:
        print(f"Output dir: {extraction_config.activations_dir}")
    print("="*70)

    # Initialize extractor
    extractor = SimpleActivationExtractor(extraction_config)

    # Load model and tokenizer
    print("\nLoading model...")
    from qwen2_jax import QwenConfig

    config = QwenConfig(
        vocab_size=151936,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    model, tokenizer, params = load_jax_model_and_tokenizer(model_path, config)
    print("✓ Model loaded")

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    with open(dataset_path) as f:
        data = json.load(f)

    if max_tasks:
        data = dict(list(data.items())[:max_tasks])

    print(f"✓ Loaded {len(data)} tasks")

    # Process tasks
    print("\nProcessing tasks...")
    results = {}

    for task_id, task in tqdm(data.items(), desc="Tasks"):
        # Create simple prompt (just use the first test input)
        test_input = task['test'][0]['input']

        # Tokenize (simplified - in practice use the full prompt creation)
        prompt = f"Input: {test_input}"
        inputs = tokenizer(prompt, return_tensors="np", truncation=True, max_length=512)
        input_ids = jnp.array(inputs['input_ids'])

        # Generate with extraction
        generated_ids = generate_with_extraction(
            model, params, input_ids,
            max_tokens=100,  # Simplified
            extractor=extractor,
            task_id=task_id,
            sample_idx=0
        )

        # Decode
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Store result
        results[task_id] = {
            'input': test_input,
            'output': output_text
        }

    # Save results
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Results saved")

    # Finalize extraction
    extractor.finalize()

    return results


def main():
    parser = argparse.ArgumentParser(description="Inference with Activation Extraction")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='results.json')
    parser.add_argument('--activations_dir', type=str, default='./activations')
    parser.add_argument('--max_tasks', type=int, default=None)
    parser.add_argument('--save_every_n_samples', type=int, default=100)
    parser.add_argument('--no_extraction', action='store_true', help="Disable extraction")
    parser.add_argument('--upload_to_cloud', action='store_true')
    parser.add_argument('--cloud_bucket', type=str, default=None)

    args = parser.parse_args()

    extraction_config = ExtractionConfig(
        extract_activations=not args.no_extraction,
        activations_dir=args.activations_dir,
        save_every_n_samples=args.save_every_n_samples,
        upload_to_cloud=args.upload_to_cloud,
        cloud_bucket=args.cloud_bucket
    )

    run_inference_with_extraction(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        extraction_config=extraction_config,
        max_tasks=args.max_tasks
    )


if __name__ == '__main__':
    main()
