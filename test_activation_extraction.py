"""
Quick test for activation extraction
"""

import subprocess
import sys

def test_activation_extraction():
    """Test the simplified activation extraction"""

    print("="*70)
    print("TESTING SIMPLIFIED ACTIVATION EXTRACTION")
    print("="*70)

    # Run with minimal settings for quick test
    cmd = [
        sys.executable,
        "extract_activations_arc.py",
        "--dataset_path", "test_data_small.json",
        "--n_tasks", "1",  # Just 1 task for testing
        "--batch_size", "2",
        "--predictions_per_task", "2",  # Just 2 predictions
        "--layers_to_extract", "10", "15", "20",  # Just 3 layers
        "--output_dir", "./test_activations",
        "--verbose"
    ]

    print("\nRunning command:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print("\n✓ Test PASSED!")
        print("Check ./test_activations/ for outputs")
    else:
        print("\n✗ Test FAILED!")
        sys.exit(1)


if __name__ == '__main__':
    test_activation_extraction()
