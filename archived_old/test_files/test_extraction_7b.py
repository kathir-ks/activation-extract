"""
Test script for Qwen 2.5 7B activation extraction with model sharding
Run with small sample size to validate the pipeline
"""

import subprocess
import sys

# Qwen 2.5 7B configuration
model_config = {
    "model_path": "Qwen/Qwen2.5-7B",  # HuggingFace model
    "vocab_size": 152064,
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False
}

# Test configuration
test_config = {
    "machine_id": 0,
    "total_machines": 1,
    "batch_size": 2,  # Small batch for testing
    "max_samples": 10,  # Just 10 samples to test
    "max_seq_length": 512,  # Shorter sequences for testing
    "layers_to_extract": [20, 21, 22, 23, 24, 25, 26, 27],  # Last 8 layers
    "dataset_name": "HuggingFaceFW/fineweb-edu",
    "dataset_config": "sample-10BT",
    "dataset_split": "train",
    "output_dir": "./test_activations_7b",
    "shard_size_gb": 0.1,  # Small shards for testing
    "upload_to_gcs": False,  # Don't upload during testing
    "verbose": True
}

print("="*70)
print("TESTING QWEN 2.5 7B EXTRACTION WITH MODEL SHARDING")
print("="*70)
print(f"\nModel: {model_config['model_path']}")
print(f"Hidden size: {model_config['hidden_size']}")
print(f"Layers: {model_config['num_hidden_layers']}")
print(f"Test samples: {test_config['max_samples']}")
print(f"Batch size: {test_config['batch_size']}")
print("="*70)

# Build command
cmd = [
    sys.executable, "extract_activations_fineweb.py",
    "--machine_id", str(test_config["machine_id"]),
    "--total_machines", str(test_config["total_machines"]),
    "--model_path", model_config["model_path"],
    "--batch_size", str(test_config["batch_size"]),
    "--max_samples", str(test_config["max_samples"]),
    "--max_seq_length", str(test_config["max_seq_length"]),
    "--layers_to_extract", *[str(l) for l in test_config["layers_to_extract"]],
    "--dataset_name", test_config["dataset_name"],
    "--dataset_config", test_config["dataset_config"],
    "--dataset_split", test_config["dataset_split"],
    "--output_dir", test_config["output_dir"],
    "--shard_size_gb", str(test_config["shard_size_gb"]),
    "--verbose"
]

if not test_config["upload_to_gcs"]:
    # Add a dummy bucket to pass validation, but don't actually upload
    cmd.extend(["--no_data_parallel"])  # Start with sequential mode first

print("\nRunning command:")
print(" ".join(cmd))
print("\n" + "="*70)

# Run the extraction
try:
    result = subprocess.run(cmd, check=True, text=True)
    print("\n" + "="*70)
    print("✓ TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
except subprocess.CalledProcessError as e:
    print("\n" + "="*70)
    print("✗ TEST FAILED!")
    print(f"Error code: {e.returncode}")
    print("="*70)
    sys.exit(1)
