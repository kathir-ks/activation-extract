"""
Dataset Conversion Utilities

Converts HuggingFace datasets to ARC-AGI format for activation extraction.
Supports streaming to handle large datasets efficiently.
"""

import json
import os
from typing import Optional, List, Dict
from tqdm import tqdm


def convert_hf_dataset_to_arc_format(
    dataset_name: str,
    column_name: str,
    output_filename: str,
    max_tasks: Optional[int] = None,
    max_train_examples: Optional[int] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
    verbose: bool = True
) -> int:
    """
    Convert a HuggingFace dataset to ARC challenge format.
    
    Args:
        dataset_name: HF dataset name (e.g., "barc0/200k_HEAVY")
        column_name: Column containing task pairs (e.g., "examples")
        output_filename: Output JSONL file path
        max_tasks: Maximum number of tasks to convert (None = all)
        max_train_examples: Maximum training examples per task (None = keep all)
        start_index: Start index in dataset (for sharding across machines)
        end_index: End index in dataset (for sharding across machines)
        verbose: Print progress
        
    Returns:
        Number of tasks converted
    """
    from datasets import load_dataset
    
    if verbose:
        print("="*70)
        print("Converting HuggingFace dataset to ARC format")
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
        print(f"Error loading dataset '{dataset_name}': {e}")
        return 0

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
                formatted_train = _format_pairs(train_pairs)
                if not formatted_train:
                    skipped_invalid += 1
                    continue

                # Format test data
                formatted_test = _format_test_pair(test_pair)
                if not formatted_test:
                    skipped_invalid += 1
                    continue

                # Create task object
                task_id = f"task_{task_counter:08x}"
                task_object = {
                    "task_id": task_id,
                    "train": formatted_train,
                    "test": formatted_test
                }

                # Write as JSON Lines
                f.write(json.dumps(task_object) + '\n')
                task_counter += 1

        if verbose:
            print(f"\n{'='*70}")
            print("Conversion complete!")
            print(f"{'='*70}")
            print(f"  Tasks converted: {task_counter}")
            print(f"  Invalid tasks skipped: {skipped_invalid}")
            print(f"  Output file: {output_filename}")
            print(f"  File size: {get_file_size_mb(output_filename):.2f} MB")
            print(f"{'='*70}")

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

    return task_counter


def _format_pairs(pairs: List) -> List[Dict]:
    """Format train/test pairs to ARC format."""
    formatted = []
    for pair in pairs:
        if isinstance(pair, list) and len(pair) == 2:
            try:
                input_grid = [[int(cell) for cell in row] for row in pair[0]]
                output_grid = [[int(cell) for cell in row] for row in pair[1]]
                formatted.append({
                    "input": input_grid,
                    "output": output_grid
                })
            except (ValueError, TypeError):
                continue
    return formatted


def _format_test_pair(test_pair: List) -> List[Dict]:
    """Format test pair to ARC format."""
    if isinstance(test_pair, list) and len(test_pair) > 0:
        try:
            test_input = [[int(cell) for cell in row] for row in test_pair[0]]
            return [{"input": test_input}]
        except (ValueError, TypeError):
            pass
    return []


def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(filepath) / (1024 * 1024)


def convert_jsonl_to_sharded(
    input_file: str,
    output_dir: str,
    tasks_per_shard: int = 1000,
    tasks_per_chunk: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Convert a JSONL file to sharded format for distributed processing.
    
    Args:
        input_file: Input JSONL file path
        output_dir: Output directory for sharded dataset
        tasks_per_shard: Number of tasks per shard
        tasks_per_chunk: Number of tasks per chunk within a shard
        verbose: Print progress
        
    Returns:
        Dictionary with shard metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Count tasks
    task_count = 0
    with open(input_file, 'r') as f:
        for _ in f:
            task_count += 1
    
    if verbose:
        print(f"Total tasks: {task_count}")
    
    num_shards = (task_count + tasks_per_shard - 1) // tasks_per_shard
    
    shard_metadata = {
        "total_tasks": task_count,
        "num_shards": num_shards,
        "tasks_per_shard": tasks_per_shard,
        "tasks_per_chunk": tasks_per_chunk,
        "shards": []
    }
    
    with open(input_file, 'r') as f:
        for shard_id in range(num_shards):
            shard_dir = os.path.join(output_dir, f"shard_{shard_id:03d}")
            os.makedirs(shard_dir, exist_ok=True)
            
            shard_tasks = []
            for _ in range(tasks_per_shard):
                line = f.readline()
                if not line:
                    break
                shard_tasks.append(json.loads(line.strip()))
            
            # Write chunks
            chunks = []
            for chunk_id, i in enumerate(range(0, len(shard_tasks), tasks_per_chunk)):
                chunk_tasks = shard_tasks[i:i + tasks_per_chunk]
                chunk_file = f"chunk_{chunk_id:03d}.jsonl"
                chunk_path = os.path.join(shard_dir, chunk_file)
                
                with open(chunk_path, 'w') as cf:
                    for task in chunk_tasks:
                        cf.write(json.dumps(task) + '\n')
                
                chunks.append({
                    "file": chunk_file,
                    "num_tasks": len(chunk_tasks)
                })
            
            # Write shard metadata
            shard_info = {
                "shard_id": shard_id,
                "total_tasks": len(shard_tasks),
                "num_chunks": len(chunks),
                "chunks": chunks,
                "status": "available"
            }
            
            with open(os.path.join(shard_dir, "metadata.json"), 'w') as mf:
                json.dump(shard_info, mf, indent=2)
            
            shard_metadata["shards"].append({
                "shard_id": shard_id,
                "directory": f"shard_{shard_id:03d}",
                "total_tasks": len(shard_tasks)
            })
            
            if verbose:
                print(f"  Created shard {shard_id}: {len(shard_tasks)} tasks, {len(chunks)} chunks")
    
    # Write master metadata
    with open(os.path.join(output_dir, "master_metadata.json"), 'w') as f:
        json.dump(shard_metadata, f, indent=2)
    
    return shard_metadata
