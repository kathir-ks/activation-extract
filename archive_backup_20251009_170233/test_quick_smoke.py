#!/usr/bin/env python3
"""
Quick smoke test to verify basic functionality
Run this first to ensure imports and basic operations work
"""

import sys
import traceback


def test_imports():
    """Test all required imports"""
    print("Testing imports...")

    try:
        # Test arc24 modules
        from arc24.encoders import MinimalGridEncoder, create_grid_encoder
        print("  ✓ arc24.encoders")

        from arc24.prompting import create_prompts_from_task, parse_grid_from_response
        print("  ✓ arc24.prompting")

        from arc24.data_augmentation import geometric_augmentation, apply_data_augmentation
        print("  ✓ arc24.data_augmentation")

        # Test main inference module
        from arc_inference_jax import ARCConfig, create_prompts, validate_grid
        print("  ✓ arc_inference_jax")

        print("✅ All imports successful!\n")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_basic_encoder():
    """Test basic encoder functionality"""
    print("Testing basic encoder...")

    try:
        from arc24.encoders import MinimalGridEncoder

        encoder = MinimalGridEncoder()
        test_grid = [[0, 1, 2], [3, 4, 5]]

        # Encode
        text = encoder.to_text(test_grid)
        expected = "012\n345"
        assert text == expected, f"Expected '{expected}', got '{text}'"

        # Decode
        decoded = encoder.to_grid(text)
        assert decoded == test_grid, f"Roundtrip failed"

        print("  ✓ Encoding/decoding works")
        print("✅ Basic encoder test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Encoder test failed: {e}")
        traceback.print_exc()
        return False


def test_data_augmentation():
    """Test data augmentation"""
    print("Testing data augmentation...")

    try:
        from arc24.data_augmentation import geometric_augmentation, revert_geometric_augmentation

        test_grid = [[0, 1], [2, 3]]

        # Test flip
        augmented = geometric_augmentation(test_grid, hflip=True, n_rot90=0)
        reverted = revert_geometric_augmentation(augmented, hflip=True, n_rot90=0)
        assert reverted == test_grid, "Augmentation roundtrip failed"

        print("  ✓ Geometric augmentation works")
        print("✅ Data augmentation test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Data augmentation test failed: {e}")
        traceback.print_exc()
        return False


def test_grid_validation():
    """Test grid validation"""
    print("Testing grid validation...")

    try:
        from arc_inference_jax import validate_grid

        # Valid grid
        valid_grid = [[0, 1, 2], [3, 4, 5]]
        validate_grid(valid_grid)
        print("  ✓ Valid grid accepted")

        # Invalid grid
        try:
            invalid_grid = [[10, 11]]  # Values > 9
            validate_grid(invalid_grid)
            print("  ❌ Invalid grid not caught")
            return False
        except AssertionError:
            print("  ✓ Invalid grid rejected")

        print("✅ Grid validation test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Grid validation test failed: {e}")
        traceback.print_exc()
        return False


def test_config():
    """Test configuration"""
    print("Testing configuration...")

    try:
        from arc_inference_jax import ARCConfig

        config = ARCConfig()
        assert config.predictions_per_task > 0
        assert config.batch_size > 0
        assert config.max_output_tokens > 0

        print("  ✓ Default configuration created")

        # Test with custom params
        custom_config = ARCConfig(
            batch_size=16,
            max_output_tokens=2000
        )
        assert custom_config.batch_size == 16
        assert custom_config.max_output_tokens == 2000

        print("  ✓ Custom configuration works")
        print("✅ Configuration test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests"""
    print("="*70)
    print("Running Smoke Tests for ARC Inference Pipeline")
    print("="*70 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Basic Encoder", test_basic_encoder),
        ("Data Augmentation", test_data_augmentation),
        ("Grid Validation", test_grid_validation),
        ("Configuration", test_config),
    ]

    results = []
    for name, test_func in tests:
        results.append(test_func())

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)

    for (name, _), result in zip(tests, results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:.<50} {status}")

    print("="*70)

    all_passed = all(results)
    if all_passed:
        print("\n🎉 All smoke tests passed! Ready to run full test suite.")
        return 0
    else:
        print("\n⚠️  Some smoke tests failed. Fix issues before running full tests.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
