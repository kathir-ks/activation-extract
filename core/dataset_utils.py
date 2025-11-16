"""
Dataset Loading and Processing Utilities

This module provides utilities for:
- Loading ARC datasets from JSONL files
- Loading from sharded datasets with automatic shard claiming
- Creating prompts from tasks with data augmentation
"""

import json
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm

from shard_manager import ShardManager, load_shard_chunks


def load_arc_dataset_jsonl(
    dataset_path: str,
    max_tasks: Optional[int] = None,
    machine_id: int = 0,
    total_machines: int = 1,
    verbose: bool = False
) -> Dict:
    """
    Load ARC dataset from JSONL format with machine-based sharding

    Args:
        dataset_path: Path to JSONL file
        max_tasks: Maximum tasks to load (per machine)
        machine_id: This machine's ID (0 to total_machines-1)
        total_machines: Total number of machines
        verbose: Print progress

    Returns:
        Dictionary mapping task_id to task data
    """
    if verbose:
        print(f"\nLoading ARC dataset from {dataset_path}...")
        print(f"  Machine {machine_id}/{total_machines-1}")
        print(f"  Max tasks per machine: {max_tasks if max_tasks else 'unlimited'}")

    tasks = {}
    task_count = 0

    with open(dataset_path, 'r') as f:
        for line_idx, line in enumerate(f):
            # Shard across machines (round-robin)
            if line_idx % total_machines != machine_id:
                continue

            try:
                task_obj = json.loads(line.strip())

                # Handle both formats:
                # 1. {"task_id": "...", "train": [...], "test": [...]}
                # 2. {"train": [...], "test": [...]} (generate task_id)

                if "task_id" in task_obj:
                    task_id = task_obj["task_id"]
                else:
                    task_id = f"task_{line_idx:08x}"

                # Store task data
                task_data = {
                    "train": task_obj["train"],
                    "test": task_obj["test"]
                }

                tasks[task_id] = task_data
                task_count += 1

                if max_tasks is not None and task_count >= max_tasks:
                    break

            except json.JSONDecodeError:
                if verbose:
                    print(f"  Warning: Skipping invalid JSON line {line_idx}")
                continue

    if verbose:
        print(f"  ✓ Loaded {len(tasks)} tasks")

    return tasks


def load_arc_dataset_from_shard(
    sharded_dataset_dir: str,
    worker_id: str,
    preferred_shard_id: Optional[int] = None,
    verbose: bool = False
) -> Tuple[Dict, int, ShardManager]:
    """
    Load ARC dataset from sharded dataset with automatic shard claiming

    Args:
        sharded_dataset_dir: Directory containing sharded dataset
        worker_id: Unique worker identifier
        preferred_shard_id: Preferred shard ID (auto-select if None)
        verbose: Print progress

    Returns:
        Tuple of (tasks_dict, shard_id, shard_manager)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Loading from Sharded Dataset")
        print(f"{'='*70}")
        print(f"  Dataset dir: {sharded_dataset_dir}")
        print(f"  Worker ID: {worker_id}")
        if preferred_shard_id is not None:
            print(f"  Preferred shard: {preferred_shard_id}")

    # Initialize shard manager
    shard_manager = ShardManager(sharded_dataset_dir, worker_id)

    # Claim a shard
    if verbose:
        print(f"\n  Claiming shard...")

    shard_info = shard_manager.claim_shard(preferred_shard_id)

    if shard_info is None:
        raise RuntimeError(
            f"No available shards in {sharded_dataset_dir}. "
            f"All shards may be in use or completed."
        )

    shard_id = shard_info["shard_id"]
    chunk_files = shard_info["chunks"]
    total_tasks = shard_info["metadata"]["total_tasks"]

    if verbose:
        print(f"  ✓ Claimed shard {shard_id}")
        print(f"    Tasks: {total_tasks:,}")
        print(f"    Chunks: {len(chunk_files)}")
        print(f"\n  Loading tasks from chunks...")

    # Load all tasks from chunks
    is_gcs = sharded_dataset_dir.startswith("gs://")
    task_list = load_shard_chunks(chunk_files, is_gcs)

    # Convert to dict format
    tasks = {}
    for task_obj in task_list:
        task_id = task_obj.get("task_id", f"task_{len(tasks):08x}")
        tasks[task_id] = {
            "train": task_obj["train"],
            "test": task_obj["test"]
        }

    if verbose:
        print(f"  ✓ Loaded {len(tasks):,} tasks from shard {shard_id}")
        print(f"{'='*70}\n")

    return tasks, shard_id, shard_manager


def create_prompts_from_dataset(
    tasks: Dict,
    grid_encoder,
    tokenizer,
    prompt_version: str,
    predictions_per_task: int,
    random_seed: Optional[int] = None,
    verbose: bool = False
) -> List[Dict]:
    """
    Create prompts for all tasks with data augmentation

    Args:
        tasks: Dictionary of tasks
        grid_encoder: Grid encoder instance
        tokenizer: Tokenizer instance
        prompt_version: Prompt format version
        predictions_per_task: Number of predictions per task
        random_seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        List of prompt dictionaries with task_id, prompt, and metadata
    """
    # Import ARC utilities (these are specific to ARC dataset)
    from arc24.data_augmentation import (
        apply_data_augmentation,
        get_random_color_map,
        set_random_seed
    )
    from arc24.prompting import create_prompts_from_task

    if random_seed is not None:
        set_random_seed(random_seed)

    prompts = []

    for task_id, task in tqdm(tasks.items(), total=len(tasks),
                              desc='Creating prompts', disable=not verbose):
        # Data augmentation parameters
        from itertools import product
        data_augmentation_params = list(product([False, True], [0, 1, 2, 3]))
        num_augmentations = len(data_augmentation_params)

        # Calculate how many times to repeat each augmentation
        repeats_per_aug = max(1, predictions_per_task // num_augmentations)

        for hflip, n_rot90 in data_augmentation_params:
            for _ in range(repeats_per_aug):
                color_map = get_random_color_map(change_background_probability=0.1)
                data_augmentation_kwargs = dict(
                    hflip=hflip,
                    n_rot90=n_rot90,
                    color_map=color_map
                )
                augmented_task = apply_data_augmentation(task, **data_augmentation_kwargs)

                task_prompts = create_prompts_from_task(
                    augmented_task,
                    grid_encoder=grid_encoder,
                    tokenizer=tokenizer,
                    is_train_prompt=False,
                    prompt_version=prompt_version
                )

                for idx, prompt in enumerate(task_prompts):
                    prompts.append({
                        'task_id': task_id,
                        'data_augmentation_kwargs': data_augmentation_kwargs,
                        'prompt': prompt,
                        'idx': idx
                    })

            # Break early if we have enough
            if len(prompts) >= predictions_per_task * len(task['test']):
                break

    return prompts
