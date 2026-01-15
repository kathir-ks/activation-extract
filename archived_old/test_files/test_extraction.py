"""
Unit tests for activation extraction pipeline
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
import tempfile
import os
import pickle
import gzip
import json
from pathlib import Path

# Import functions to test
from extract_activations_fineweb_multihost import (
    pad_sequences,
    create_device_mesh,
    create_sharding_strategy,
    ActivationStorage,
    QwenConfig,
)


class TestPadSequences:
    """Test sequence padding functionality"""

    def test_pad_to_max_in_batch(self):
        """Test padding to max length in batch (no fixed_length)"""
        seq1 = np.array([1, 2, 3])
        seq2 = np.array([4, 5, 6, 7, 8])
        sequences = [seq1, seq2]

        result = pad_sequences(sequences, pad_token_id=0)

        assert result.shape == (2, 5)
        assert np.array_equal(result[0], [1, 2, 3, 0, 0])
        assert np.array_equal(result[1], [4, 5, 6, 7, 8])

    def test_pad_to_fixed_length(self):
        """Test padding to fixed length"""
        seq1 = np.array([1, 2, 3])
        seq2 = np.array([4, 5, 6, 7, 8])
        sequences = [seq1, seq2]

        result = pad_sequences(sequences, pad_token_id=0, fixed_length=10)

        assert result.shape == (2, 10)
        assert np.array_equal(result[0], [1, 2, 3, 0, 0, 0, 0, 0, 0, 0])
        assert np.array_equal(result[1], [4, 5, 6, 7, 8, 0, 0, 0, 0, 0])

    def test_truncate_when_longer_than_fixed_length(self):
        """Test truncation when sequences exceed fixed_length"""
        seq1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        sequences = [seq1]

        result = pad_sequences(sequences, pad_token_id=0, fixed_length=5)

        assert result.shape == (1, 5)
        assert np.array_equal(result[0], [1, 2, 3, 4, 5])

    def test_custom_pad_token(self):
        """Test with custom pad token ID"""
        seq1 = np.array([1, 2, 3])
        sequences = [seq1]

        result = pad_sequences(sequences, pad_token_id=999, fixed_length=5)

        assert np.array_equal(result[0], [1, 2, 3, 999, 999])

    def test_single_sequence(self):
        """Test with single sequence"""
        seq1 = np.array([1, 2, 3])
        sequences = [seq1]

        result = pad_sequences(sequences, pad_token_id=0, fixed_length=5)

        assert result.shape == (1, 5)
        assert np.array_equal(result[0], [1, 2, 3, 0, 0])

    def test_all_same_length(self):
        """Test when all sequences already same length"""
        seq1 = np.array([1, 2, 3])
        seq2 = np.array([4, 5, 6])
        sequences = [seq1, seq2]

        result = pad_sequences(sequences, pad_token_id=0)

        assert result.shape == (2, 3)
        assert np.array_equal(result[0], [1, 2, 3])
        assert np.array_equal(result[1], [4, 5, 6])


class TestDeviceMesh:
    """Test device mesh creation"""

    def test_1d_mesh_creation(self):
        """Test 1D mesh creation"""
        devices = jax.devices()
        num_devices = len(devices)

        mesh, sharding = create_device_mesh(num_devices, mesh_type='1d')

        # mesh.shape returns OrderedDict, so check axis names and devices
        assert mesh.axis_names == ('model',)
        assert len(mesh.devices.flatten()) == num_devices
        assert isinstance(sharding, NamedSharding)

    def test_1d_mesh_shape(self):
        """Test that 1D mesh has correct shape"""
        devices = jax.devices()
        num_devices = len(devices)

        mesh, _ = create_device_mesh(num_devices, mesh_type='1d')

        # Mesh should be 1D array of devices
        assert len(mesh.devices.shape) == 1
        assert mesh.devices.shape[0] == num_devices


class TestShardingStrategy:
    """Test sharding strategy creation"""

    def test_1d_sharding_rules(self):
        """Test 1D sharding rules creation"""
        devices = jax.devices()
        num_devices = len(devices)
        mesh, _ = create_device_mesh(num_devices, mesh_type='1d')

        rules = create_sharding_strategy(mesh)

        assert isinstance(rules, dict)
        assert 'embed_tokens' in rules
        assert 'q_proj' in rules
        assert 'k_proj' in rules
        assert 'v_proj' in rules
        assert 'o_proj' in rules

        # Check that rules are PartitionSpec objects
        for key, spec in rules.items():
            assert isinstance(spec, P)


class TestActivationStorage:
    """Test activation storage functionality"""

    def test_add_and_retrieve_activation(self):
        """Test adding and retrieving activations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ActivationStorage(
                output_dir=tmpdir,
                shard_size_gb=0.001,  # 1 MB
                compress_shards=False,
                verbose=False
            )

            # Add activation
            activation = np.random.randn(128, 256).astype(np.float32)
            storage.add_activation(
                layer_idx=0,
                activation=activation,
                sample_idx=0,
                text_preview="test text"
            )

            # Check it's in buffer (buffer uses integer keys, not strings)
            assert 0 in storage.buffer
            assert len(storage.buffer[0]) == 1

    def test_auto_shard_on_size_limit(self):
        """Test automatic shard creation when size limit reached"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ActivationStorage(
                output_dir=tmpdir,
                shard_size_gb=0.000001,  # Very small: ~1 KB
                compress_shards=False,
                verbose=False
            )

            # Add large activation that exceeds limit
            activation = np.random.randn(128, 256).astype(np.float32)
            storage.add_activation(
                layer_idx=0,
                activation=activation,
                sample_idx=0,
                text_preview="test"
            )

            # Add another to trigger shard save
            storage.add_activation(
                layer_idx=0,
                activation=activation,
                sample_idx=1,
                text_preview="test"
            )

            # Should have created a shard file
            shard_files = list(Path(tmpdir).glob("shard_*.pkl"))
            assert len(shard_files) >= 1

    def test_compression(self):
        """Test that compression works"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ActivationStorage(
                output_dir=tmpdir,
                shard_size_gb=0.001,  # 1 MB
                compress_shards=True,
                verbose=False
            )

            activation = np.random.randn(128, 256).astype(np.float32)
            storage.add_activation(
                layer_idx=0,
                activation=activation,
                sample_idx=0,
                text_preview="test"
            )

            # Force save using private method
            storage._save_shard()

            # Check compressed file exists
            shard_files = list(Path(tmpdir).glob("shard_*.pkl.gz"))
            assert len(shard_files) == 1

            # Check can load compressed file
            with gzip.open(shard_files[0], 'rb') as f:
                data = pickle.load(f)
                # Buffer uses integer keys, but saved shards use string keys
                assert 'layer_0' in data or 0 in data

    def test_metadata_creation(self):
        """Test metadata file creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ActivationStorage(
                output_dir=tmpdir,
                shard_size_gb=0.001,  # 1 MB
                compress_shards=False,
                verbose=False
            )

            activation = np.random.randn(128, 256).astype(np.float32)
            storage.add_activation(
                layer_idx=0,
                activation=activation,
                sample_idx=0,
                text_preview="test"
            )

            # Finalize
            storage.finalize()

            # Check metadata exists
            metadata_file = Path(tmpdir) / "metadata.json"
            assert metadata_file.exists()

            # Check metadata content
            with open(metadata_file) as f:
                metadata = json.load(f)
                assert 'total_shards' in metadata
                assert 'total_samples' in metadata

    def test_multiple_layers(self):
        """Test storing activations from multiple layers"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ActivationStorage(
                output_dir=tmpdir,
                shard_size_gb=0.001,  # 1 MB
                compress_shards=False,
                verbose=False
            )

            # Add activations from different layers
            for layer_idx in [0, 1, 2]:
                activation = np.random.randn(128, 256).astype(np.float32)
                storage.add_activation(
                    layer_idx=layer_idx,
                    activation=activation,
                    sample_idx=0,
                    text_preview="test"
                )

            # Check all layers in buffer (uses integer keys)
            assert 0 in storage.buffer
            assert 1 in storage.buffer
            assert 2 in storage.buffer


class TestQwenConfig:
    """Test Qwen configuration"""

    def test_config_creation(self):
        """Test creating config with valid parameters"""
        config = QwenConfig(
            vocab_size=151936,
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
        )

        assert config.vocab_size == 151936
        assert config.hidden_size == 896
        assert config.num_hidden_layers == 24

    def test_config_defaults(self):
        """Test config has sensible defaults"""
        config = QwenConfig(
            vocab_size=151936,
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
        )

        # Check defaults
        assert config.max_position_embeddings == 32768
        assert config.rope_theta == 1000000.0


class TestEndToEndExtraction:
    """End-to-end integration tests"""

    def test_small_extraction_pipeline(self):
        """Test extracting activations from a small synthetic model"""
        # This would require mocking the model or using a tiny real model
        # Skip if no devices available
        if len(jax.devices()) == 0:
            pytest.skip("No JAX devices available")

        # Create simple test case
        batch_size = 2
        seq_len = 8
        hidden_size = 16

        # Simulate input
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        # Simulate activation output
        fake_activation = jnp.ones((batch_size, seq_len, hidden_size))

        # Test storage
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ActivationStorage(
                output_dir=tmpdir,
                shard_size_gb=0.001,  # 1 MB
                compress_shards=True,
                verbose=False
            )

            # Store fake activations
            for i in range(batch_size):
                storage.add_activation(
                    layer_idx=0,
                    activation=np.array(fake_activation[i]),
                    sample_idx=i,
                    text_preview=f"Sample {i}"
                )

            storage.finalize()

            # Verify files created
            assert (Path(tmpdir) / "metadata.json").exists()
            shard_files = list(Path(tmpdir).glob("shard_*.pkl.gz"))
            assert len(shard_files) > 0


class TestBatchProcessing:
    """Test batch processing logic"""

    def test_batch_padding_consistency(self):
        """Test that batch padding produces consistent shapes"""
        sequences = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6, 7]),
        ]

        batch_size = 4
        max_seq_length = 10

        # Simulate batch padding
        actual_batch_size = len(sequences)
        if actual_batch_size < batch_size:
            pad_count = batch_size - actual_batch_size
            sequences = sequences + [sequences[-1]] * pad_count

        # Pad sequences
        padded = pad_sequences(sequences, pad_token_id=0, fixed_length=max_seq_length)

        # Check shape is consistent
        assert padded.shape == (batch_size, max_seq_length)

    def test_partial_batch_handling(self):
        """Test handling of partial batches"""
        sequences = [
            np.array([1, 2, 3]),
        ]

        batch_size = 4
        max_seq_length = 10

        # Pad batch dimension
        actual_batch_size = len(sequences)
        if actual_batch_size < batch_size:
            pad_count = batch_size - actual_batch_size
            sequences = sequences + [sequences[-1]] * pad_count

        # Pad sequences
        padded = pad_sequences(sequences, pad_token_id=0, fixed_length=max_seq_length)

        assert padded.shape == (batch_size, max_seq_length)
        # First sample should be real data
        assert np.array_equal(padded[0][:3], [1, 2, 3])
        # Remaining should be duplicates (padded)
        for i in range(1, batch_size):
            assert np.array_equal(padded[i][:3], [1, 2, 3])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
