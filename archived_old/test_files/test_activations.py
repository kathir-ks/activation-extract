#!/usr/bin/env python3
"""
Test suite for validating activation extraction outputs.

This script validates:
1. Shard structure and format
2. Activation shapes and data integrity
3. Metadata consistency
4. Layer coverage
5. Sample indexing correctness
"""

import pickle
import gzip
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


class ActivationValidator:
    """Validates activation extraction outputs"""

    def __init__(self, output_dir: str = "activations_arc"):
        self.output_dir = Path(output_dir)
        self.metadata_path = self.output_dir / "metadata.json"
        self.errors = []
        self.warnings = []

    def log_error(self, test_name: str, message: str):
        """Log an error"""
        self.errors.append(f"[ERROR] {test_name}: {message}")

    def log_warning(self, test_name: str, message: str):
        """Log a warning"""
        self.warnings.append(f"[WARN] {test_name}: {message}")

    def log_success(self, test_name: str):
        """Log a success"""
        print(f"✓ {test_name}")

    def load_metadata(self) -> Dict:
        """Load and validate metadata file"""
        test_name = "Metadata Loading"
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            self.log_success(test_name)
            return metadata
        except Exception as e:
            self.log_error(test_name, f"Failed to load metadata: {e}")
            return None

    def validate_metadata_structure(self, metadata: Dict):
        """Validate metadata has expected structure"""
        test_name = "Metadata Structure"

        required_fields = ['total_shards', 'total_samples', 'shards']
        missing = [f for f in required_fields if f not in metadata]

        if missing:
            self.log_error(test_name, f"Missing required fields: {missing}")
            return

        if not isinstance(metadata['shards'], list):
            self.log_error(test_name, "shards field is not a list")
            return

        if len(metadata['shards']) != metadata['total_shards']:
            self.log_error(test_name,
                f"Shard count mismatch: {len(metadata['shards'])} != {metadata['total_shards']}")
            return

        self.log_success(test_name)

    def validate_shard_files_exist(self, metadata: Dict):
        """Validate all shard files exist"""
        test_name = "Shard Files Existence"

        missing_files = []
        for shard_info in metadata['shards']:
            shard_path = self.output_dir / shard_info['filename']
            if not shard_path.exists():
                missing_files.append(shard_info['filename'])

        if missing_files:
            self.log_error(test_name, f"Missing {len(missing_files)} shard files: {missing_files[:5]}")
        else:
            self.log_success(test_name)

    def load_shard(self, filename: str) -> Dict:
        """Load a single shard file"""
        shard_path = self.output_dir / filename
        with gzip.open(shard_path, 'rb') as f:
            return pickle.load(f)

    def validate_shard_structure(self, shard_data: Dict, filename: str, expected_layers: List[int]):
        """Validate structure of a single shard"""
        test_name = f"Shard Structure ({filename})"

        # Check it's a dict with layer keys
        if not isinstance(shard_data, dict):
            self.log_error(test_name, f"Shard is not a dict, got {type(shard_data)}")
            return False

        # Check layers match metadata
        shard_layers = set(shard_data.keys())
        expected_layers_set = set(expected_layers)
        if shard_layers != expected_layers_set:
            self.log_error(test_name,
                f"Layer mismatch: {shard_layers} != {expected_layers_set}")
            return False

        # Check each layer has list of activation records
        for layer_idx, records in shard_data.items():
            if not isinstance(records, list):
                self.log_error(test_name,
                    f"Layer {layer_idx} data is not a list, got {type(records)}")
                return False

            # Check each record has required fields
            required_fields = ['task_id', 'sample_idx', 'activation', 'shape', 'prompt']
            for i, record in enumerate(records):
                if not isinstance(record, dict):
                    self.log_error(test_name,
                        f"Layer {layer_idx} record {i} is not a dict")
                    return False

                missing = [f for f in required_fields if f not in record]
                if missing:
                    self.log_error(test_name,
                        f"Layer {layer_idx} record {i} missing fields: {missing}")
                    return False

        return True

    def validate_activation_data(self, shard_data: Dict, filename: str):
        """Validate activation data integrity"""
        test_name = f"Activation Data ({filename})"

        for layer_idx, records in shard_data.items():
            for i, record in enumerate(records):
                activation = record['activation']

                # Check it's a numpy array
                if not isinstance(activation, np.ndarray):
                    self.log_error(test_name,
                        f"Layer {layer_idx} record {i} activation is not ndarray")
                    continue

                # Check dtype
                if activation.dtype != np.float32:
                    self.log_warning(test_name,
                        f"Layer {layer_idx} record {i} has dtype {activation.dtype}, expected float32")

                # Check for NaN
                if np.isnan(activation).any():
                    self.log_error(test_name,
                        f"Layer {layer_idx} record {i} contains NaN values")

                # Check for Inf
                if np.isinf(activation).any():
                    self.log_error(test_name,
                        f"Layer {layer_idx} record {i} contains Inf values")

                # Check shape matches metadata
                expected_shape = tuple(record['shape'])
                if activation.shape != expected_shape:
                    self.log_error(test_name,
                        f"Layer {layer_idx} record {i} shape mismatch: {activation.shape} != {expected_shape}")

                # Check 2D shape (seq_len, hidden_dim)
                if len(activation.shape) != 2:
                    self.log_error(test_name,
                        f"Layer {layer_idx} record {i} not 2D: {activation.shape}")
                else:
                    seq_len, hidden_dim = activation.shape

                    # Qwen 0.5B has 896 hidden dim
                    if hidden_dim != 896:
                        self.log_error(test_name,
                            f"Layer {layer_idx} record {i} unexpected hidden_dim: {hidden_dim} != 896")

                    # Sequence length should be reasonable (> 0, < 4096)
                    if seq_len <= 0 or seq_len > 4096:
                        self.log_warning(test_name,
                            f"Layer {layer_idx} record {i} unusual seq_len: {seq_len}")

    def validate_layer_coverage(self, metadata: Dict, expected_layers: Set[int]):
        """Validate that all expected layers are present"""
        test_name = "Layer Coverage"

        # Collect all layers across all shards
        all_layers = set()
        for shard_info in metadata['shards']:
            all_layers.update(shard_info['layers'])

        missing_layers = expected_layers - all_layers
        extra_layers = all_layers - expected_layers

        if missing_layers:
            self.log_error(test_name, f"Missing layers: {sorted(missing_layers)}")
        if extra_layers:
            self.log_warning(test_name, f"Extra layers: {sorted(extra_layers)}")

        if not missing_layers and not extra_layers:
            self.log_success(test_name)

    def validate_sample_consistency(self, metadata: Dict):
        """Validate sample counts are consistent"""
        test_name = "Sample Consistency"

        # Count samples per layer from metadata
        samples_per_layer = defaultdict(int)
        for shard_info in metadata['shards']:
            for layer, count in shard_info.get('samples_per_layer', {}).items():
                samples_per_layer[int(layer)] += count

        # All layers should have same number of samples
        sample_counts = list(samples_per_layer.values())
        if len(set(sample_counts)) > 1:
            self.log_error(test_name,
                f"Inconsistent sample counts across layers: {dict(samples_per_layer)}")
        else:
            self.log_success(test_name)

    def validate_sample_indices(self):
        """Validate sample indices are correct by loading actual shards"""
        test_name = "Sample Indices"

        # Load metadata
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        # Group shards by layer
        shards_by_layer = defaultdict(list)
        for shard_info in metadata['shards']:
            for layer in shard_info['layers']:
                shards_by_layer[layer].append(shard_info['filename'])

        # For each layer, load all shards and check sample_idx continuity
        for layer, shard_files in sorted(shards_by_layer.items()):
            sample_indices = []
            for shard_file in shard_files:
                shard_data = self.load_shard(shard_file)
                if layer in shard_data:
                    for record in shard_data[layer]:
                        sample_indices.append(record['sample_idx'])

            # Check for duplicates
            if len(sample_indices) != len(set(sample_indices)):
                self.log_error(test_name,
                    f"Layer {layer} has duplicate sample indices: {sample_indices}")
                continue

            # Check indices are sequential (0, 1, 2, ...)
            expected_indices = list(range(len(sample_indices)))
            if sorted(sample_indices) != expected_indices:
                self.log_error(test_name,
                    f"Layer {layer} has non-sequential indices: {sorted(sample_indices)} != {expected_indices}")
                continue

        self.log_success(test_name)

    def validate_cross_layer_consistency(self):
        """Validate same sample_idx across layers has same task_id and prompt"""
        test_name = "Cross-Layer Consistency"

        # Load metadata
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        # Collect all records grouped by sample_idx
        records_by_sample = defaultdict(list)

        for shard_info in metadata['shards']:
            shard_data = self.load_shard(shard_info['filename'])
            for layer, records in shard_data.items():
                for record in records:
                    records_by_sample[record['sample_idx']].append({
                        'layer': layer,
                        'task_id': record['task_id'],
                        'prompt': record['prompt']
                    })

        # Check consistency within each sample_idx
        for sample_idx, records in records_by_sample.items():
            # All records for same sample should have same task_id
            task_ids = set(r['task_id'] for r in records)
            if len(task_ids) > 1:
                self.log_error(test_name,
                    f"Sample {sample_idx} has multiple task_ids: {task_ids}")

            # All records for same sample should have same prompt
            prompts = set(r['prompt'] for r in records)
            if len(prompts) > 1:
                self.log_error(test_name,
                    f"Sample {sample_idx} has multiple prompts")

        self.log_success(test_name)

    def run_all_tests(self, expected_layers: Set[int] = set(range(10, 24)),
                      sample_shards_to_validate: int = 5):
        """Run all validation tests"""
        print("=" * 60)
        print("ACTIVATION EXTRACTION VALIDATION TESTS")
        print("=" * 60)
        print()

        # Test 1: Load metadata
        print("1. METADATA TESTS")
        print("-" * 60)
        metadata = self.load_metadata()
        if metadata is None:
            print("\n❌ Cannot proceed without metadata")
            return False

        # Test 2: Validate metadata structure
        self.validate_metadata_structure(metadata)

        # Test 3: Check shard files exist
        self.validate_shard_files_exist(metadata)

        print()
        print("2. LAYER AND SAMPLE TESTS")
        print("-" * 60)

        # Test 4: Validate layer coverage
        self.validate_layer_coverage(metadata, expected_layers)

        # Test 5: Validate sample consistency
        self.validate_sample_consistency(metadata)

        print()
        print("3. SHARD STRUCTURE TESTS (sampling random shards)")
        print("-" * 60)

        # Test 6: Validate shard structure (sample a few shards)
        import random
        shard_sample = random.sample(metadata['shards'],
                                     min(sample_shards_to_validate, len(metadata['shards'])))

        for shard_info in shard_sample:
            try:
                shard_data = self.load_shard(shard_info['filename'])

                # Validate structure
                if self.validate_shard_structure(shard_data, shard_info['filename'],
                                                  shard_info['layers']):
                    self.log_success(f"Shard Structure ({shard_info['filename']})")

                # Validate activation data
                self.validate_activation_data(shard_data, shard_info['filename'])
                self.log_success(f"Activation Data ({shard_info['filename']})")

            except Exception as e:
                self.log_error("Shard Validation",
                    f"Failed to validate {shard_info['filename']}: {e}")

        print()
        print("4. SAMPLE INDEX TESTS")
        print("-" * 60)

        # Test 7: Validate sample indices
        self.validate_sample_indices()

        # Test 8: Validate cross-layer consistency
        self.validate_cross_layer_consistency()

        # Print summary
        print()
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        if self.errors:
            print(f"\n❌ {len(self.errors)} ERRORS:")
            for error in self.errors:
                print(f"  {error}")
        else:
            print("\n✓ All tests passed!")

        if self.warnings:
            print(f"\n⚠ {len(self.warnings)} WARNINGS:")
            for warning in self.warnings:
                print(f"  {warning}")

        print()

        # Print statistics
        print("STATISTICS:")
        print(f"  Total shards: {metadata['total_shards']}")
        print(f"  Total samples: {metadata['total_samples']}")
        print(f"  Expected layers: {sorted(expected_layers)}")
        print(f"  Shard size: {metadata.get('shard_size_gb', 'N/A')} GB")

        return len(self.errors) == 0


def main():
    """Main test runner"""
    validator = ActivationValidator("activations_arc")

    # Expected layers for this extraction (10-23 inclusive)
    expected_layers = set(range(10, 24))

    # Run all tests
    success = validator.run_all_tests(
        expected_layers=expected_layers,
        sample_shards_to_validate=10  # Validate 10 random shards in detail
    )

    if success:
        print("✓ All validation tests passed!")
        return 0
    else:
        print("❌ Some validation tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
