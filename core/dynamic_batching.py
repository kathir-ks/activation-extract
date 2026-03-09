"""
Dynamic length batching for activation extraction.

Instead of padding all sequences to a fixed max_seq_length (wasting compute
on padding and truncating long sequences), this module:

1. Filters out sequences exceeding a hard cap (model's max_position_embeddings)
2. Sorts sequences by length for efficient batching
3. Groups sequences into length buckets with adaptive batch sizes
4. Returns batches where all sequences are padded to the bucket size

This keeps memory usage roughly constant across batches while supporting
sequences of vastly different lengths.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# Fixed bucket boundaries. Sequences are padded to the smallest bucket
# that fits them. JIT compiles once per bucket size.
DEFAULT_LENGTH_BUCKETS = [512, 1024, 2048, 4096, 8192, 16384, 32768]

# Adaptive batch sizes per bucket to keep memory roughly constant.
# Longer sequences use smaller batches. Base is calibrated for
# v5litepod-64 with 0.5B model (896 hidden_dim, bfloat16).
DEFAULT_BATCH_SIZES = {
    512: 64,
    1024: 64,
    2048: 32,
    4096: 16,
    8192: 8,
    16384: 4,
    32768: 2,
}


def get_bucket_size(seq_len: int, buckets: List[int] = None) -> int:
    """Return the smallest bucket that fits this sequence length."""
    if buckets is None:
        buckets = DEFAULT_LENGTH_BUCKETS
    for bucket in buckets:
        if seq_len <= bucket:
            return bucket
    return buckets[-1]


def get_batch_size_for_bucket(
    bucket_size: int,
    base_batch_sizes: Dict[int, int] = None,
    num_hosts: int = 1,
) -> int:
    """
    Return the global batch size for a given bucket size.
    Must be divisible by num_hosts for FSDP sharding.
    """
    if base_batch_sizes is None:
        base_batch_sizes = DEFAULT_BATCH_SIZES

    # Find the matching or next-larger bucket
    batch_size = base_batch_sizes.get(bucket_size)
    if batch_size is None:
        # Fallback: find the largest bucket <= this size
        for b in sorted(base_batch_sizes.keys(), reverse=True):
            if b <= bucket_size:
                batch_size = base_batch_sizes[b]
                break
        if batch_size is None:
            batch_size = 2  # safe minimum

    # Ensure divisible by num_hosts
    batch_size = max(num_hosts, (batch_size // num_hosts) * num_hosts)
    return batch_size


@dataclass
class DynamicBatch:
    """A batch of sequences with uniform padded length."""
    sequences: List[List[int]]       # Token ID lists (unpadded)
    original_indices: List[int]      # Index into the original (pre-sort) sequence list
    actual_lengths: List[int]        # Actual (unpadded) length of each sequence
    bucket_size: int                 # Padded length for this batch
    batch_size: int                  # Global batch size (may include padding sequences)


def create_dynamic_batches(
    sequences: List[List[int]],
    prompts_data: List[Dict],
    max_seq_length: int = 32768,
    length_buckets: List[int] = None,
    batch_sizes: Dict[int, int] = None,
    num_hosts: int = 1,
    verbose: bool = False,
) -> Tuple[List[DynamicBatch], List[int], List[Dict]]:
    """
    Create dynamically-batched groups from variable-length sequences.

    Args:
        sequences: List of token ID lists (variable length)
        prompts_data: Corresponding prompt metadata
        max_seq_length: Hard cap - filter out sequences exceeding this
        length_buckets: Bucket boundaries (default: DEFAULT_LENGTH_BUCKETS)
        batch_sizes: Batch size per bucket (default: DEFAULT_BATCH_SIZES)
        num_hosts: Number of hosts for FSDP (batch must be divisible)
        verbose: Print statistics

    Returns:
        Tuple of:
        - List of DynamicBatch objects
        - Filtered/sorted sequence list (for checkpoint compatibility)
        - Filtered/sorted prompts_data list
    """
    if length_buckets is None:
        length_buckets = DEFAULT_LENGTH_BUCKETS
    if batch_sizes is None:
        batch_sizes = DEFAULT_BATCH_SIZES

    # 1. Compute lengths and filter
    seq_lengths = [len(s) for s in sequences]
    total_before = len(sequences)

    valid_indices = [i for i, l in enumerate(seq_lengths) if l <= max_seq_length]
    filtered_count = total_before - len(valid_indices)

    if verbose and filtered_count > 0:
        print(f"  Filtered out {filtered_count} sequences exceeding {max_seq_length} tokens "
              f"({filtered_count/total_before*100:.1f}%)")

    # 2. Sort valid sequences by length (deterministic for multihost consistency)
    valid_indices_sorted = sorted(valid_indices, key=lambda i: len(sequences[i]))

    sorted_sequences = [sequences[i] for i in valid_indices_sorted]
    sorted_prompts = [prompts_data[i] for i in valid_indices_sorted]
    sorted_lengths = [len(s) for s in sorted_sequences]

    # 3. Create batches grouped by bucket
    batches = []
    i = 0
    while i < len(sorted_sequences):
        # Determine bucket for this batch (based on longest sequence starting here)
        # Look ahead to find batch boundary
        bucket = get_bucket_size(sorted_lengths[i], length_buckets)
        batch_sz = get_batch_size_for_bucket(bucket, batch_sizes, num_hosts)

        # Collect sequences that fit in this bucket
        batch_end = i
        while batch_end < len(sorted_sequences) and batch_end - i < batch_sz:
            if sorted_lengths[batch_end] <= bucket:
                batch_end += 1
            else:
                break

        if batch_end == i:
            # Current sequence doesn't fit in its bucket (shouldn't happen with proper buckets)
            # Move to next larger bucket
            bucket = get_bucket_size(sorted_lengths[i], length_buckets)
            batch_end = i + 1

        batch = DynamicBatch(
            sequences=sorted_sequences[i:batch_end],
            original_indices=list(range(i, batch_end)),
            actual_lengths=sorted_lengths[i:batch_end],
            bucket_size=bucket,
            batch_size=batch_sz,
        )
        batches.append(batch)
        i = batch_end

    if verbose:
        print(f"\n  Dynamic batching summary:")
        print(f"    Total sequences: {len(sorted_sequences)} (filtered {filtered_count})")
        print(f"    Total batches: {len(batches)}")
        # Stats per bucket
        bucket_stats = {}
        for b in batches:
            bs = b.bucket_size
            if bs not in bucket_stats:
                bucket_stats[bs] = {'batches': 0, 'sequences': 0}
            bucket_stats[bs]['batches'] += 1
            bucket_stats[bs]['sequences'] += len(b.sequences)

        print(f"    {'Bucket':>8} {'Batches':>8} {'Sequences':>10} {'Batch Size':>10}")
        for bucket in sorted(bucket_stats.keys()):
            stats = bucket_stats[bucket]
            bs = get_batch_size_for_bucket(bucket, batch_sizes, num_hosts)
            print(f"    {bucket:>8} {stats['batches']:>8} {stats['sequences']:>10} {bs:>10}")

    return batches, sorted_sequences, sorted_prompts


def pad_batch_to_bucket(
    sequences: List[List[int]],
    bucket_size: int,
    batch_size: int,
    pad_token_id: int = 0,
) -> Tuple[List[List[int]], int]:
    """
    Pad sequences to bucket_size and pad batch dimension to batch_size.

    Returns:
        Tuple of (padded_sequences, actual_count)
    """
    actual_count = len(sequences)

    # Pad sequences to bucket length
    padded = []
    for seq in sequences:
        if len(seq) < bucket_size:
            padded.append(seq + [pad_token_id] * (bucket_size - len(seq)))
        else:
            # Truncate to bucket (shouldn't happen if bucketing is correct)
            padded.append(seq[:bucket_size])

    # Pad batch dimension if needed
    if actual_count < batch_size:
        pad_count = batch_size - actual_count
        padded.extend([padded[-1]] * pad_count)

    return padded, actual_count
