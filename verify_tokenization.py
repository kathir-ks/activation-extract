"""
Verify Tokenization Pipeline for ARC Dataset
Prints formatted prompts to file for manual inspection
"""

import json
import sys
from transformers import AutoTokenizer

# Import ARC formatting utilities
sys.path.append('.')
from arc24.prompting import create_prompts_from_task
from arc24.encoders import GridCodeBlockEncoder, MinimalGridEncoder


def load_arc_tasks(dataset_path, max_tasks=10):
    """Load ARC tasks from JSONL file"""
    tasks = []
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_tasks:
                break
            task = json.loads(line.strip())
            tasks.append(task)
    return tasks


def verify_tokenization(dataset_path, model_path, output_file, max_tasks=10):
    """Verify tokenization pipeline and write to file"""

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading {max_tasks} tasks from {dataset_path}...")
    tasks = load_arc_tasks(dataset_path, max_tasks)

    # Create grid encoder (default configuration)
    grid_encoder = GridCodeBlockEncoder(MinimalGridEncoder())

    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("ARC TOKENIZATION PIPELINE VERIFICATION\n")
        f.write("=" * 100 + "\n\n")

        for i, task in enumerate(tasks):
            f.write(f"\n{'=' * 100}\n")
            f.write(f"TASK {i+1}/{len(tasks)}: {task.get('task_id', 'unknown')}\n")
            f.write(f"{'=' * 100}\n\n")

            # Extract train examples
            train_examples = task.get('train', [])
            test_examples = task.get('test', [])

            f.write(f"Train examples: {len(train_examples)}\n")
            f.write(f"Test examples: {len(test_examples)}\n\n")

            # Show first train example
            if train_examples:
                f.write("-" * 100 + "\n")
                f.write("SAMPLE TRAIN EXAMPLE:\n")
                f.write("-" * 100 + "\n")
                input_grid = train_examples[0].get('input', [])
                output_grid = train_examples[0].get('output', [])

                f.write(f"Input shape: {len(input_grid)}x{len(input_grid[0]) if input_grid else 0}\n")
                f.write(f"Input grid (raw): {input_grid}\n\n")

                # Convert to text using grid encoder
                input_text = grid_encoder.to_text(input_grid)
                f.write("Input grid (encoded):\n")
                f.write(input_text + "\n\n")

                f.write(f"Output shape: {len(output_grid)}x{len(output_grid[0]) if output_grid else 0}\n")
                output_text = grid_encoder.to_text(output_grid)
                f.write("Output grid (encoded):\n")
                f.write(output_text + "\n\n")

            # Format full task prompt using actual API
            f.write("-" * 100 + "\n")
            f.write("FORMATTED TASK PROMPTS (all test cases):\n")
            f.write("-" * 100 + "\n\n")

            # Create prompts (training version)
            try:
                prompts = create_prompts_from_task(
                    task,
                    grid_encoder,
                    tokenizer,
                    is_train_prompt=True,
                    prompt_version='output-from-examples-v1'
                )

                for test_idx, prompt in enumerate(prompts):
                    f.write(f"--- Test case {test_idx + 1} ---\n\n")
                    f.write(prompt + "\n\n")

                # Use first prompt for tokenization analysis
                if prompts:
                    prompt = prompts[0]

                    # Tokenize
                    f.write("-" * 100 + "\n")
                    f.write("TOKENIZATION DETAILS (first prompt):\n")
                    f.write("-" * 100 + "\n")

                    tokens = tokenizer(prompt, return_tensors='pt', truncation=False)
                    input_ids = tokens['input_ids'][0]

                    f.write(f"Number of tokens: {len(input_ids)}\n")
                    f.write(f"First 50 token IDs: {input_ids[:50].tolist()}\n")
                    f.write(f"Last 50 token IDs: {input_ids[-50:].tolist()}\n\n")

                    # Check for special tokens
                    f.write(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})\n")
                    f.write(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})\n")
                    f.write(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})\n\n")

            except Exception as e:
                f.write(f"ERROR creating prompts: {str(e)}\n\n")
                import traceback
                f.write(traceback.format_exc() + "\n")

    print(f"\n✓ Tokenization verification written to: {output_file}")
    print(f"  Please review the file to verify:")
    print(f"  - Prompt formatting is correct")
    print(f"  - Grid encoding is readable")
    print(f"  - Tokenization preserves information")
    print(f"  - Special tokens are handled properly")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify ARC tokenization pipeline")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-0.5B", help="Model path")
    parser.add_argument('--output_file', type=str, default="tokenization_verification.txt", help="Output file")
    parser.add_argument('--max_tasks', type=int, default=10, help="Number of tasks to verify")

    args = parser.parse_args()

    verify_tokenization(args.dataset_path, args.model_path, args.output_file, args.max_tasks)
