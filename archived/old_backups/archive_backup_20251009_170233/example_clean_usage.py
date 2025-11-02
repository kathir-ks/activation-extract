"""
Example usage of the clean inference pipeline
Shows how to use the pipeline for various scenarios
"""

from inference_clean import InferenceConfig, InferencePipeline


def example_basic_inference():
    """Basic inference example"""
    print("="*70)
    print("EXAMPLE 1: Basic Inference")
    print("="*70)

    config = InferenceConfig(
        model_path="KathirKs/qwen-2.5-0.5b",
        batch_size=4,
        max_new_tokens=100,
        mesh_shape=(1, 1),  # Single device
        extract_activations=False
    )

    pipeline = InferencePipeline(config)
    pipeline.setup()

    prompts = [
        "What is the capital of France?",
        "Explain machine learning in one sentence.",
        "Write a haiku about coding.",
    ]

    results = pipeline.generate(prompts)

    for r in results:
        print(f"\nPrompt {r['prompt_idx']}: {prompts[r['prompt_idx']]}")
        print(f"Output: {r['output'][:200]}...")  # Truncate for display


def example_with_activations():
    """Inference with activation extraction"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Inference with Activation Extraction")
    print("="*70)

    config = InferenceConfig(
        model_path="KathirKs/qwen-2.5-0.5b",
        batch_size=2,
        max_new_tokens=50,
        mesh_shape=(1, 1),
        extract_activations=True,
        layers_to_extract=[10, 15, 20, 23],  # Extract specific layers
        activations_dir="./activations_output"
    )

    pipeline = InferencePipeline(config)
    pipeline.setup()

    prompts = [
        "Translate to French: Hello, how are you?",
        "Write a function to sort a list in Python."
    ]

    results = pipeline.generate(prompts)

    print(f"\nGenerated {len(results)} outputs")
    print(f"Activations saved to: {config.activations_dir}")


def example_distributed():
    """Multi-device distributed inference"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Distributed Inference (Multi-Device)")
    print("="*70)

    # This will use all available TPU/GPU devices
    import jax
    n_devices = len(jax.devices())

    config = InferenceConfig(
        model_path="KathirKs/qwen-2.5-0.5b",
        batch_size=8,
        max_new_tokens=200,
        mesh_shape=(n_devices, 1),  # Data parallelism across all devices
        extract_activations=False
    )

    pipeline = InferencePipeline(config)
    pipeline.setup()

    # Large batch for distributed processing
    prompts = [
        f"Generate a creative story about topic {i}."
        for i in range(16)
    ]

    results = pipeline.generate(prompts)

    print(f"\nProcessed {len(results)} prompts across {n_devices} devices")
    print(f"First output: {results[0]['output'][:100]}...")


def example_batch_processing():
    """Process large dataset efficiently"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Processing Large Dataset")
    print("="*70)

    config = InferenceConfig(
        model_path="KathirKs/qwen-2.5-0.5b",
        batch_size=16,
        max_new_tokens=256,
        mesh_shape=(4, 1),  # Use 4 devices for data parallelism
        extract_activations=False
    )

    pipeline = InferencePipeline(config)
    pipeline.setup()

    # Simulate large dataset
    large_dataset = [
        f"Question {i}: What is {i} + {i}?"
        for i in range(100)
    ]

    results = pipeline.generate(large_dataset)

    print(f"\nProcessed {len(results)} items")
    print(f"Sample outputs:")
    for i in [0, 50, 99]:
        print(f"  [{i}]: {results[i]['output'][:80]}...")


def example_research_activations():
    """Extract activations for research purposes"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Research - Detailed Activation Extraction")
    print("="*70)

    config = InferenceConfig(
        model_path="KathirKs/qwen-2.5-0.5b",
        batch_size=4,
        max_new_tokens=100,
        mesh_shape=(1, 1),
        extract_activations=True,
        layers_to_extract=list(range(24)),  # All layers
        activations_dir="./research_activations"
    )

    pipeline = InferencePipeline(config)
    pipeline.setup()

    # Research prompts
    prompts = [
        "Define: Artificial Intelligence",
        "Define: Machine Learning",
        "Define: Deep Learning",
        "Define: Neural Networks"
    ]

    results = pipeline.generate(prompts)

    print(f"\nExtracted activations for {len(results)} prompts")
    print(f"All 24 layers extracted")
    print(f"Saved to: {config.activations_dir}")
    print("\nUse this data for:")
    print("  - Analyzing model representations")
    print("  - Probing experiments")
    print("  - Feature visualization")
    print("  - Interpretability research")


if __name__ == '__main__':
    # Run examples (comment out ones you don't want to run)

    # Basic examples
    # example_basic_inference()

    # Advanced examples
    example_with_activations()
    # example_distributed()
    # example_batch_processing()

    # Research example
    # example_research_activations()

    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
    print("\nTo run an example, uncomment it in the __main__ block")
