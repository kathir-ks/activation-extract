"""
Continuous grid chunking pipeline for SAE activation extraction.

Instead of using full prompts (system prompt + instructions + grids), this module:

1. Strips ALL text/instructions from the pipeline
2. Converts only grid data (train inputs, train outputs, test inputs) to tokens
3. Concatenates all grid tokens into a continuous stream
4. Splits the stream into fixed-size chunks (default 2048 tokens)

Overflow from one task naturally continues into the next chunk. This maximizes
token utilization (no wasted tokens on prompt text) and reduces padding (only
the very last chunk may need padding).

This is specifically designed for SAE training where continuous grid
representations matter more than prompt structure.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import product
from tqdm.auto import tqdm


# Separator token between grids (newline). Gives the model a signal
# that one grid ended and another begins, without wasting tokens on
# full prompt text.
GRID_SEPARATOR = "\n"


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk in the continuous stream."""
    chunk_idx: int              # Index of this chunk in the stream
    num_tokens: int             # Actual tokens (before padding)
    is_last: bool               # Whether this is the last chunk
    task_boundaries: List[int]  # Token offsets within this chunk where new tasks start


def create_grid_token_stream(
    tasks: Dict,
    grid_encoder,
    tokenizer,
    predictions_per_task: int = 8,
    random_seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[List[int], List[Dict]]:
    """
    Convert all grids from tasks into a continuous token stream.

    For each task (with data augmentation):
    1. Convert each grid (train inputs, train outputs, test inputs) to text
       using the grid encoder
    2. Tokenize the grid text
    3. Append tokens to the continuous stream with a separator between grids

    Args:
        tasks: Dictionary mapping task_id to task data
        grid_encoder: Grid encoder instance (e.g., GridShapeEncoder)
        tokenizer: HuggingFace tokenizer
        predictions_per_task: Number of augmented versions per task
        random_seed: Random seed for reproducibility
        verbose: Print statistics

    Returns:
        Tuple of (token_stream, stream_metadata) where:
        - token_stream: Single flat list of token IDs
        - stream_metadata: List of dicts with task boundaries in the stream
    """
    from arc24.data_augmentation import (
        apply_data_augmentation,
        get_random_color_map,
        set_random_seed,
    )

    if random_seed is not None:
        set_random_seed(random_seed)

    token_stream = []
    stream_metadata = []  # Track where each task's tokens start
    separator_tokens = tokenizer.encode(GRID_SEPARATOR, add_special_tokens=False)

    data_augmentation_params = list(product([False, True], [0, 1, 2, 3]))
    num_augmentations = len(data_augmentation_params)
    repeats_per_aug = max(1, predictions_per_task // num_augmentations)

    total_grids = 0
    total_tasks_processed = 0

    for task_id, task in tqdm(tasks.items(), total=len(tasks),
                               desc="Building grid token stream",
                               disable=not verbose):
        task_augment_count = 0
        target = predictions_per_task * len(task["test"])

        for hflip, n_rot90 in data_augmentation_params:
            for _ in range(repeats_per_aug):
                color_map = get_random_color_map(change_background_probability=0.1)
                augmented_task = apply_data_augmentation(
                    task,
                    hflip=hflip,
                    n_rot90=n_rot90,
                    color_map=color_map,
                )

                # Record where this task's tokens start in the stream
                task_start_offset = len(token_stream)

                # Convert all grids to tokens: train inputs, train outputs, test inputs
                for sample in augmented_task.get("train", []):
                    # Input grid
                    grid_text = grid_encoder.to_text(sample["input"])
                    grid_tokens = tokenizer.encode(grid_text, add_special_tokens=False)
                    token_stream.extend(grid_tokens)
                    token_stream.extend(separator_tokens)
                    total_grids += 1

                    # Output grid
                    grid_text = grid_encoder.to_text(sample["output"])
                    grid_tokens = tokenizer.encode(grid_text, add_special_tokens=False)
                    token_stream.extend(grid_tokens)
                    token_stream.extend(separator_tokens)
                    total_grids += 1

                for test_sample in augmented_task.get("test", []):
                    # Test input grid
                    grid_text = grid_encoder.to_text(test_sample["input"])
                    grid_tokens = tokenizer.encode(grid_text, add_special_tokens=False)
                    token_stream.extend(grid_tokens)
                    token_stream.extend(separator_tokens)
                    total_grids += 1

                    # Test output grid (if available)
                    if "output" in test_sample:
                        grid_text = grid_encoder.to_text(test_sample["output"])
                        grid_tokens = tokenizer.encode(grid_text, add_special_tokens=False)
                        token_stream.extend(grid_tokens)
                        token_stream.extend(separator_tokens)
                        total_grids += 1

                stream_metadata.append({
                    "task_id": task_id,
                    "stream_offset": task_start_offset,
                    "token_count": len(token_stream) - task_start_offset,
                    "augmentation": {"hflip": hflip, "n_rot90": n_rot90},
                })

                task_augment_count += 1

            if task_augment_count >= target:
                break

        total_tasks_processed += 1

    if verbose:
        print(f"\n  Grid token stream statistics:")
        print(f"    Tasks processed: {total_tasks_processed}")
        print(f"    Total grids encoded: {total_grids}")
        print(f"    Total tokens in stream: {len(token_stream):,}")
        avg_per_grid = len(token_stream) / total_grids if total_grids > 0 else 0
        print(f"    Average tokens per grid: {avg_per_grid:.1f}")

    return token_stream, stream_metadata


def chunk_token_stream(
    token_stream: List[int],
    chunk_size: int = 2048,
    pad_token_id: int = 0,
    verbose: bool = False,
) -> Tuple[List[List[int]], List[ChunkMetadata]]:
    """
    Split a continuous token stream into fixed-size chunks.

    Chunks are exactly chunk_size tokens. The last chunk is padded if needed.
    Overflow from one task naturally continues into the next chunk.

    Args:
        token_stream: Flat list of token IDs
        chunk_size: Fixed size for each chunk (default 2048)
        pad_token_id: Token ID used for padding the last chunk
        verbose: Print statistics

    Returns:
        Tuple of (chunks, chunk_metadata_list) where:
        - chunks: List of token ID lists, each of length chunk_size
        - chunk_metadata_list: Metadata for each chunk
    """
    total_tokens = len(token_stream)
    num_full_chunks = total_tokens // chunk_size
    remainder = total_tokens % chunk_size
    num_chunks = num_full_chunks + (1 if remainder > 0 else 0)

    chunks = []
    metadata = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_tokens)
        chunk = token_stream[start:end]
        actual_len = len(chunk)

        # Pad last chunk if needed
        if actual_len < chunk_size:
            chunk = chunk + [pad_token_id] * (chunk_size - actual_len)

        chunks.append(chunk)
        metadata.append(ChunkMetadata(
            chunk_idx=i,
            num_tokens=actual_len,
            is_last=(i == num_chunks - 1),
            task_boundaries=[],  # Could be populated from stream_metadata if needed
        ))

    if verbose:
        padding_tokens = chunk_size - remainder if remainder > 0 else 0
        padding_pct = padding_tokens / (num_chunks * chunk_size) * 100 if num_chunks > 0 else 0
        print(f"\n  Chunking statistics:")
        print(f"    Total tokens: {total_tokens:,}")
        print(f"    Chunk size: {chunk_size}")
        print(f"    Total chunks: {num_chunks}")
        print(f"    Full chunks: {num_full_chunks}")
        print(f"    Padding tokens (last chunk): {padding_tokens}")
        print(f"    Padding waste: {padding_pct:.2f}%")
        print(f"    Token utilization: {100 - padding_pct:.2f}%")

    return chunks, metadata


def create_grid_chunks_from_dataset(
    tasks: Dict,
    grid_encoder,
    tokenizer,
    chunk_size: int = 2048,
    predictions_per_task: int = 8,
    random_seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[List[List[int]], List[ChunkMetadata], List[Dict]]:
    """
    End-to-end pipeline: tasks -> grid token stream -> fixed-size chunks.

    This replaces the standard prompt creation + tokenization + dynamic batching
    pipeline when using the grid chunking approach.

    Args:
        tasks: Dictionary mapping task_id to task data
        grid_encoder: Grid encoder instance
        tokenizer: HuggingFace tokenizer
        chunk_size: Fixed chunk size (default 2048)
        predictions_per_task: Augmentation count per task
        random_seed: Random seed
        verbose: Print statistics

    Returns:
        Tuple of (chunks, chunk_metadata, stream_metadata) where:
        - chunks: List of token ID lists, each of length chunk_size
        - chunk_metadata: Per-chunk metadata
        - stream_metadata: Per-task metadata (offsets into the original stream)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Grid Chunking Pipeline")
        print(f"{'='*60}")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Predictions per task: {predictions_per_task}")

    # Step 1: Build continuous grid token stream
    token_stream, stream_metadata = create_grid_token_stream(
        tasks=tasks,
        grid_encoder=grid_encoder,
        tokenizer=tokenizer,
        predictions_per_task=predictions_per_task,
        random_seed=random_seed,
        verbose=verbose,
    )

    # Step 2: Split into fixed-size chunks
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    chunks, chunk_metadata = chunk_token_stream(
        token_stream=token_stream,
        chunk_size=chunk_size,
        pad_token_id=pad_token_id,
        verbose=verbose,
    )

    if verbose:
        print(f"{'='*60}\n")

    return chunks, chunk_metadata, stream_metadata
