"""
Quick validation script to verify the clean pipeline works end-to-end
Run this to make sure everything is working correctly
"""

import sys
import jax
import jax.numpy as jnp

def validate_imports():
    """Validate all required imports work"""
    print("Validating imports...")
    try:
        from inference_clean import (
            ModelConfig, InferenceConfig, QwenModel, QwenModelWithActivations,
            InferencePipeline, create_generation_step, setup_mesh
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def validate_model_creation():
    """Validate model creation works"""
    print("\nValidating model creation...")
    try:
        from inference_clean import ModelConfig, QwenModel

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1
        )

        model = QwenModel(config=config)
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 5), 0, 1000)
        variables = model.init(jax.random.PRNGKey(0), input_ids)

        print("✅ Model creation successful")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_forward_pass():
    """Validate forward pass works"""
    print("\nValidating forward pass...")
    try:
        from inference_clean import ModelConfig, QwenModel

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1
        )

        model = QwenModel(config=config)
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 5), 0, 1000)
        variables = model.init(jax.random.PRNGKey(0), input_ids)

        logits = model.apply(variables, input_ids)

        assert logits.shape == (2, 5, 1000), f"Wrong shape: {logits.shape}"

        print("✅ Forward pass successful")
        return True
    except Exception as e:
        print(f"❌ Forward pass error: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_generation():
    """Validate generation works"""
    print("\nValidating generation...")
    try:
        from inference_clean import (
            ModelConfig, QwenModel, create_generation_step, generate_tokens
        )

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1
        )

        model = QwenModel(config=config)
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 5), 0, 1000)
        variables = model.init(jax.random.PRNGKey(0), input_ids)

        generation_step = create_generation_step(model, extract_activations=False)
        generated = generate_tokens(
            variables, input_ids, max_new_tokens=3,
            generation_step=generation_step, extract_activations=False
        )

        assert generated.shape == (2, 8), f"Wrong shape: {generated.shape}"

        print("✅ Generation successful")
        return True
    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_activation_extraction():
    """Validate activation extraction works"""
    print("\nValidating activation extraction...")
    try:
        from inference_clean import (
            ModelConfig, QwenModelWithActivations,
            create_generation_step, generate_tokens
        )

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1
        )

        model = QwenModelWithActivations(config=config, extract_layers=[1, 2])
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 5), 0, 1000)
        variables = model.init(jax.random.PRNGKey(0), input_ids)

        generation_step = create_generation_step(model, extract_activations=True)
        generated, activations = generate_tokens(
            variables, input_ids, max_new_tokens=2,
            generation_step=generation_step, extract_activations=True
        )

        assert generated.shape == (2, 7), f"Wrong shape: {generated.shape}"
        assert len(activations) == 2, f"Wrong activation count: {len(activations)}"

        print("✅ Activation extraction successful")
        return True
    except Exception as e:
        print(f"❌ Activation extraction error: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_mesh_setup():
    """Validate distributed mesh setup works"""
    print("\nValidating mesh setup...")
    try:
        from inference_clean import InferenceConfig, setup_mesh

        config = InferenceConfig(mesh_shape=(1, 1))
        mesh = setup_mesh(config)

        print("✅ Mesh setup successful")
        return True
    except Exception as e:
        print(f"❌ Mesh setup error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validations"""
    print("="*70)
    print("VALIDATION SUITE FOR CLEAN INFERENCE PIPELINE")
    print("="*70)

    results = []

    results.append(("Imports", validate_imports()))
    results.append(("Model Creation", validate_model_creation()))
    results.append(("Forward Pass", validate_forward_pass()))
    results.append(("Generation", validate_generation()))
    results.append(("Activation Extraction", validate_activation_extraction()))
    results.append(("Mesh Setup", validate_mesh_setup()))

    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:30s} {status}")

    print("="*70)

    all_passed = all(success for _, success in results)

    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("\nThe clean inference pipeline is ready to use.")
        print("See README_CLEAN.md for usage examples.")
        return 0
    else:
        print("\n❌ SOME VALIDATIONS FAILED")
        print("\nPlease check the errors above and fix them.")
        return 1


if __name__ == '__main__':
    sys.exit(main())