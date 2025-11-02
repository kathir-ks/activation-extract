"""
Transform HuggingFace dataset to ARC-AGI format
Converts dataset from https://huggingface.co/datasets/barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems
"""

import json
import argparse
from datasets import load_dataset
from tqdm import tqdm
import hashlib
from typing import Dict, List, Any
import os


def generate_task_id(examples: List[Dict], index: int) -> str:
    """Generate a unique task ID from the examples"""
    # Use hash of examples + index for uniqueness
    content = json.dumps(examples, sort_keys=True) + str(index)
    hash_obj = hashlib.md5(content.encode())
    return hash_obj.hexdigest()[:8]


def parse_example(example_entry: Dict) -> Dict[str, List[List[int]]]:
    """
    Parse a single example entry which contains 'input' and 'output' grids

    Expected structure of example_entry:
    {
        'input': [[0, 1, 2], ...],
        'output': [[3, 4, 5], ...]
    }
    """
    if isinstance(example_entry, dict) and 'input' in example_entry and 'output' in example_entry:
        return {
            'input': example_entry['input'],
            'output': example_entry['output']
        }
    else:
        raise ValueError(f"Invalid example format: {example_entry}")


def transform_row_to_arc_format(row: Dict, index: int) -> Dict[str, Any]:
    """
    Transform a single row from HF dataset to ARC-AGI format

    Expected HF row structure:
    {
        'examples': [
            {'input': [[...]], 'output': [[...]]},
            {'input': [[...]], 'output': [[...]]},
            {'input': [[...]], 'output': [[...]]},
            {'input': [[...]], 'output': [[...]]}  # This becomes test input
        ]
    }

    ARC-AGI format:
    {
        'task_id': {
            'train': [
                {'input': [[...]], 'output': [[...]]},
                {'input': [[...]], 'output': [[...]]}
            ],
            'test': [
                {'input': [[...]], 'output': [[...]]}
            ]
        }
    }
    """
    examples = row['examples']

    # Ensure we have at least 2 examples (1 train, 1 test)
    if len(examples) < 2:
        raise ValueError(f"Row {index} has less than 2 examples: {len(examples)}")

    # Split examples: all but last for training, last for test
    train_examples = examples[:-1]  # 2-3 examples
    test_example = examples[-1]     # Last example

    # Generate task ID
    task_id = generate_task_id(examples, index)

    # Parse train examples
    train_data = []
    for train_ex in train_examples:
        parsed = parse_example(train_ex)
        train_data.append(parsed)

    # Parse test example
    test_parsed = parse_example(test_example)

    # Create ARC-AGI format
    # Test should only have input initially (output saved separately for verification)
    arc_task = {
        task_id: {
            'train': train_data,
            'test': [{
                'input': test_parsed['input']
            }]
        }
    }

    # Return both the arc task and the test output for verification
    test_output = test_parsed['output']

    return arc_task, {task_id: {'test': [{'output': test_output}]}}


def transform_dataset(
    dataset_name: str,
    output_dir: str,
    split: str = 'train',
    max_samples: int = None,
    streaming: bool = False
):
    """
    Transform HuggingFace dataset to ARC-AGI format

    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Directory to save outputs
        split: Dataset split to use
        max_samples: Maximum number of samples to process (None for all)
        streaming: Whether to use streaming mode for large datasets
    """
    print(f"Loading dataset: {dataset_name}, split: {split}")

    # Load dataset
    if streaming:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, split=split)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Output files
    arc_data_path = os.path.join(output_dir, f'arc_format_{split}.json')
    test_outputs_path = os.path.join(output_dir, f'test_outputs_{split}.json')

    arc_data = {}
    test_outputs = {}

    # Process dataset
    if max_samples:
        if streaming:
            import itertools
            dataset_iter = itertools.islice(dataset, max_samples)
            total = max_samples
        else:
            dataset_iter = dataset.select(range(min(max_samples, len(dataset))))
            total = len(dataset_iter)
    else:
        dataset_iter = dataset
        total = len(dataset) if not streaming else None

    errors = []
    processed = 0

    for idx, row in enumerate(tqdm(dataset_iter, total=total, desc="Transforming dataset")):
        try:
            arc_task, test_output = transform_row_to_arc_format(row, idx)

            # Merge into main dictionaries
            arc_data.update(arc_task)
            test_outputs.update(test_output)

            processed += 1

        except Exception as e:
            errors.append((idx, str(e)))
            print(f"\nError processing row {idx}: {e}")
            continue

    # Save outputs
    print(f"\nSaving ARC-AGI format data to {arc_data_path}")
    with open(arc_data_path, 'w') as f:
        json.dump(arc_data, f, indent=2)

    print(f"Saving test outputs to {test_outputs_path}")
    with open(test_outputs_path, 'w') as f:
        json.dump(test_outputs, f, indent=2)

    # Print statistics
    print("\n" + "="*70)
    print("TRANSFORMATION SUMMARY")
    print("="*70)
    print(f"Total processed: {processed}")
    print(f"Total errors: {len(errors)}")
    print(f"Output files:")
    print(f"  - ARC format: {arc_data_path}")
    print(f"  - Test outputs: {test_outputs_path}")

    if errors:
        error_log_path = os.path.join(output_dir, 'transformation_errors.json')
        with open(error_log_path, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"  - Errors log: {error_log_path}")

    print("="*70)

    return arc_data_path, test_outputs_path


def main():
    parser = argparse.ArgumentParser(description='Transform HF dataset to ARC-AGI format')
    parser.add_argument(
        '--dataset',
        type=str,
        default='barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems',
        help='HuggingFace dataset name'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/kathirks_gc/torch_xla/qwen/arc_data',
        help='Output directory for transformed data'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Dataset split to process'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (None for all)'
    )
    parser.add_argument(
        '--streaming',
        action='store_true',
        help='Use streaming mode for large datasets'
    )

    args = parser.parse_args()

    transform_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
        streaming=args.streaming
    )


if __name__ == '__main__':
    main()
