"""
Convert HuggingFace dataset to ARC-AGI format for activation extraction

This script converts datasets like barc0/200k_HEAVY to ARC format
WITHOUT slicing training examples (keeps all pairs for richer context)
"""

import json
from tqdm import tqdm
from datasets import load_dataset
import argparse


def convert_hf_dataset_to_arc_format(
    dataset_name: str,
    column_name: str,
    output_filename: str,
    max_tasks: int = None,
    max_train_examples: int = None,
    start_index: int = 0,
    end_index: int = None,
    verbose: bool = True
):
    """
    Loads a Hugging Face dataset and converts it to ARC challenge format.

    Args:
        dataset_name: HF dataset name (e.g., "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems")
        column_name: Column containing task pairs (e.g., "examples")
        output_filename: Output JSONL file path
        max_tasks: Maximum number of tasks to convert (None = all)
        max_train_examples: Maximum training examples per task (None = keep all)
        start_index: Start index in dataset (for sharding across machines)
        end_index: End index in dataset (for sharding across machines)
        verbose: Print progress
    """
    if verbose:
        print("="*70)
        print(f"Converting HuggingFace dataset to ARC format")
        print("="*70)
        print(f"  Dataset: {dataset_name}")
        print(f"  Column: {column_name}")
        print(f"  Output: {output_filename}")
        print(f"  Max tasks: {max_tasks if max_tasks else 'unlimited'}")
        print(f"  Max train examples: {max_train_examples if max_train_examples else 'unlimited'}")
        print(f"  Range: {start_index} to {end_index if end_index else 'end'}")
        print("="*70)

    try:
        # Stream the dataset to avoid loading it all into memory
        dataset = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"❌ Error loading dataset '{dataset_name}'")
        print(f"   {e}")
        return

    task_counter = 0
    current_index = 0
    skipped_invalid = 0

    try:
        with open(output_filename, 'w') as f:
            for sample in tqdm(dataset, desc="Converting", disable=not verbose):
                # Skip until we reach start_index
                if current_index < start_index:
                    current_index += 1
                    continue

                # Stop if we've reached end_index
                if end_index is not None and current_index >= end_index:
                    break

                # Stop if we've reached max_tasks
                if max_tasks is not None and task_counter >= max_tasks:
                    break

                current_index += 1

                # Validate column exists
                if column_name not in sample:
                    skipped_invalid += 1
                    continue

                task_pairs = sample[column_name]

                # Validate format: must be list with at least 2 pairs (train + test)
                if not isinstance(task_pairs, list) or len(task_pairs) < 2:
                    skipped_invalid += 1
                    continue

                # Split into train/test: last pair is test, rest are train
                train_pairs = task_pairs[:-1]
                test_pair = task_pairs[-1]

                # Apply max_train_examples limit if specified
                if max_train_examples is not None:
                    train_pairs = train_pairs[:max_train_examples]

                # Format training data
                formatted_train_list = []
                for pair in train_pairs:
                    if isinstance(pair, list) and len(pair) == 2:
                        try:
                            # Convert to integer grids
                            input_grid = [[int(cell) for cell in row] for row in pair[0]]
                            output_grid = [[int(cell) for cell in row] for row in pair[1]]
                            formatted_train_list.append({
                                "input": input_grid,
                                "output": output_grid
                            })
                        except (ValueError, TypeError):
                            # Skip invalid grids
                            continue

                # Skip if no valid training examples
                if len(formatted_train_list) == 0:
                    skipped_invalid += 1
                    continue

                # Format test data
                if isinstance(test_pair, list) and len(test_pair) > 0:
                    try:
                        test_input_grid = [[int(cell) for cell in row] for row in test_pair[0]]
                        formatted_test_list = [{"input": test_input_grid}]
                    except (ValueError, TypeError):
                        skipped_invalid += 1
                        continue
                else:
                    skipped_invalid += 1
                    continue

                # Create task object (CORRECTED FORMAT)
                task_id = f"task_{task_counter:08x}"
                task_object = {
                    "task_id": task_id,
                    "train": formatted_train_list,
                    "test": formatted_test_list
                }

                # Write as JSON Lines (one task per line)
                f.write(json.dumps(task_object) + '\n')
                task_counter += 1

        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ Conversion complete!")
            print(f"{'='*70}")
            print(f"  Tasks converted: {task_counter}")
            print(f"  Invalid tasks skipped: {skipped_invalid}")
            print(f"  Output file: {output_filename}")
            print(f"  File size: {get_file_size_mb(output_filename):.2f} MB")
            print(f"{'='*70}")

    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()


def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB"""
    import os
    return os.path.getsize(filepath) / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace dataset to ARC-AGI format"
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        default="barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        '--column_name',
        type=str,
        default="examples",
        help="Column containing task pairs"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default="arc_formatted_challenges.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        '--max_tasks',
        type=int,
        default=None,
        help="Maximum number of tasks to convert (default: all)"
    )
    parser.add_argument(
        '--max_train_examples',
        type=int,
        default=None,
        help="Maximum training examples per task (default: keep all)"
    )
    parser.add_argument(
        '--start_index',
        type=int,
        default=0,
        help="Start index in dataset (for sharding)"
    )
    parser.add_argument(
        '--end_index',
        type=int,
        default=None,
        help="End index in dataset (for sharding)"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Print progress"
    )

    args = parser.parse_args()

    convert_hf_dataset_to_arc_format(
        dataset_name=args.dataset_name,
        column_name=args.column_name,
        output_filename=args.output_file,
        max_tasks=args.max_tasks,
        max_train_examples=args.max_train_examples,
        start_index=args.start_index,
        end_index=args.end_index,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
