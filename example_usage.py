"""
Example usage script for ARC-AGI inference with JAX/TPU
"""

import json
from arc_inference_jax import ARCConfig, inference_main

# Example of how to create a simple ARC task for testing
example_task = {
    "task_1": {
        "train": [
            {
                "input": [[0, 1], [1, 0]],
                "output": [[1, 0], [0, 1]]
            },
            {
                "input": [[0, 2], [2, 0]],
                "output": [[2, 0], [0, 2]]
            }
        ],
        "test": [
            {
                "input": [[0, 3], [3, 0]]
            }
        ]
    }
}

def create_example_data():
    """Create example ARC data file"""
    with open('/home/kathirks_gc/torch_xla/qwen/example_arc_data.json', 'w') as f:
        json.dump(example_task, f, indent=2)
    print("Created example_arc_data.json")

def run_example_inference():
    """Run inference with example configuration"""
    # Create example data
    create_example_data()

    # Set up configuration
    config = ARCConfig(
        dataset_path='/home/kathirks_gc/torch_xla/qwen/example_arc_data.json',
        model_path="Qwen/Qwen2.5-0.5B",  # Replace with your fine-tuned model path
        output_filepath='/home/kathirks_gc/torch_xla/qwen/example_results.json',
        predictions_per_task=2,  # Reduce for quick testing
        batch_size=1,
        max_output_tokens=100,
        verbose=True
    )

    print("Running ARC-AGI inference...")
    print(f"Configuration: {config}")

    # Note: This would run the actual inference
    # inference_main() with the above config

if __name__ == "__main__":
    create_example_data()
    print("Example setup complete!")
    print("To run inference, modify the model_path in run_example_inference() and call it.")