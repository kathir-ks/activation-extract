#!/usr/bin/env python3
"""Test the grid chunking pipeline."""

import sys
sys.path.insert(0, '.')

from core.grid_chunking import (
    create_grid_token_stream,
    chunk_token_stream,
    create_grid_chunks_from_dataset,
)


def test_chunk_token_stream():
    """Test basic chunking of a token stream."""
    # Simple stream of 10 tokens
    stream = list(range(10))
    chunks, metadata = chunk_token_stream(stream, chunk_size=4, pad_token_id=0, verbose=True)

    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    assert chunks[0] == [0, 1, 2, 3], f"Chunk 0: {chunks[0]}"
    assert chunks[1] == [4, 5, 6, 7], f"Chunk 1: {chunks[1]}"
    assert chunks[2] == [8, 9, 0, 0], f"Chunk 2 (padded): {chunks[2]}"

    assert metadata[0].num_tokens == 4
    assert metadata[1].num_tokens == 4
    assert metadata[2].num_tokens == 2
    assert metadata[2].is_last == True
    assert metadata[0].is_last == False

    print("PASS: test_chunk_token_stream")


def test_chunk_exact_fit():
    """Test when stream divides evenly into chunks."""
    stream = list(range(8))
    chunks, metadata = chunk_token_stream(stream, chunk_size=4, pad_token_id=0)

    assert len(chunks) == 2
    assert chunks[0] == [0, 1, 2, 3]
    assert chunks[1] == [4, 5, 6, 7]
    assert metadata[1].num_tokens == 4
    assert metadata[1].is_last == True

    print("PASS: test_chunk_exact_fit")


def test_chunk_empty():
    """Test empty stream."""
    chunks, metadata = chunk_token_stream([], chunk_size=4, pad_token_id=0)
    assert len(chunks) == 0
    assert len(metadata) == 0
    print("PASS: test_chunk_empty")


def test_grid_chunks_from_dataset():
    """Test end-to-end pipeline with mock tasks."""
    # Create simple mock tasks
    tasks = {
        "task_1": {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]},
            ],
            "test": [
                {"input": [[9, 0], [1, 2]], "output": [[3, 4], [5, 6]]},
            ],
        },
        "task_2": {
            "train": [
                {"input": [[0, 0], [0, 0]], "output": [[1, 1], [1, 1]]},
            ],
            "test": [
                {"input": [[2, 2], [2, 2]]},
            ],
        },
    }

    # Create a simple grid encoder (mock)
    class SimpleGridEncoder:
        def to_text(self, grid):
            return '\n'.join([''.join(str(x) for x in row) for row in grid])

    # Create a simple tokenizer (mock)
    class SimpleTokenizer:
        pad_token_id = 0
        def encode(self, text, add_special_tokens=False):
            # Each character becomes a token (ASCII value)
            return [ord(c) for c in text]

    grid_encoder = SimpleGridEncoder()
    tokenizer = SimpleTokenizer()

    chunks, chunk_meta, stream_meta = create_grid_chunks_from_dataset(
        tasks=tasks,
        grid_encoder=grid_encoder,
        tokenizer=tokenizer,
        chunk_size=64,
        predictions_per_task=1,
        random_seed=42,
        verbose=True,
    )

    print(f"\n  Results:")
    print(f"    Chunks created: {len(chunks)}")
    print(f"    Each chunk size: {len(chunks[0]) if chunks else 'N/A'}")
    print(f"    Stream metadata entries: {len(stream_meta)}")

    # All chunks should be exactly chunk_size
    for i, chunk in enumerate(chunks):
        assert len(chunk) == 64, f"Chunk {i} has {len(chunk)} tokens, expected 64"

    # Stream metadata should have entries for tasks
    assert len(stream_meta) > 0

    print("PASS: test_grid_chunks_from_dataset")


def test_grid_token_stream():
    """Test that grid token stream builds correctly."""
    tasks = {
        "task_1": {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]},
            ],
            "test": [
                {"input": [[9, 0], [1, 2]]},
            ],
        },
    }

    class SimpleGridEncoder:
        def to_text(self, grid):
            return '\n'.join([''.join(str(x) for x in row) for row in grid])

    class SimpleTokenizer:
        pad_token_id = 0
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    stream, metadata = create_grid_token_stream(
        tasks=tasks,
        grid_encoder=SimpleGridEncoder(),
        tokenizer=SimpleTokenizer(),
        predictions_per_task=1,
        random_seed=42,
        verbose=True,
    )

    assert len(stream) > 0, "Stream should not be empty"
    assert len(metadata) > 0, "Metadata should not be empty"
    assert metadata[0]['task_id'] == 'task_1'
    assert metadata[0]['stream_offset'] == 0

    print("PASS: test_grid_token_stream")


if __name__ == '__main__':
    test_chunk_token_stream()
    print()
    test_chunk_exact_fit()
    print()
    test_chunk_empty()
    print()
    test_grid_token_stream()
    print()
    test_grid_chunks_from_dataset()
    print("\nAll tests passed!")
