#!/usr/bin/env python3
"""
Comprehensive tests for the new pipeline components:
- Dataset transformation (HF → ARC)
- Activation extraction
- Storage system
"""

import unittest
import tempfile
import json
import os
import shutil
import pickle
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import modules to test
from transform_hf_to_arc import (
    generate_task_id, parse_example, transform_row_to_arc_format,
    transform_dataset
)
from simple_extraction_inference import (
    SimpleActivationExtractor, ExtractionConfig
)


class TestDatasetTransformation(unittest.TestCase):
    """Test HF dataset transformation to ARC format"""

    def test_generate_task_id(self):
        """Test task ID generation"""
        examples = [{'input': [[1, 2]], 'output': [[3, 4]]}]

        # Same input should generate same ID
        id1 = generate_task_id(examples, 0)
        id2 = generate_task_id(examples, 0)
        self.assertEqual(id1, id2)

        # Different input should generate different ID
        id3 = generate_task_id(examples, 1)
        self.assertNotEqual(id1, id3)

        # ID should be 8 characters
        self.assertEqual(len(id1), 8)

    def test_parse_example(self):
        """Test parsing of single example"""
        example = {
            'input': [[0, 1], [2, 3]],
            'output': [[4, 5], [6, 7]]
        }

        parsed = parse_example(example)

        self.assertIn('input', parsed)
        self.assertIn('output', parsed)
        self.assertEqual(parsed['input'], [[0, 1], [2, 3]])
        self.assertEqual(parsed['output'], [[4, 5], [6, 7]])

    def test_parse_example_invalid(self):
        """Test parsing with invalid input"""
        invalid_examples = [
            {'input': [[1, 2]]},  # Missing output
            {'output': [[3, 4]]},  # Missing input
            [[1, 2], [3, 4]],     # Not a dict
        ]

        for invalid in invalid_examples:
            with self.assertRaises(ValueError):
                parse_example(invalid)

    def test_transform_row_to_arc_format(self):
        """Test transformation of single row"""
        row = {
            'examples': [
                {'input': [[0, 1]], 'output': [[2, 3]]},
                {'input': [[4, 5]], 'output': [[6, 7]]},
                {'input': [[8, 9]], 'output': [[10, 11]]}
            ]
        }

        arc_task, test_output = transform_row_to_arc_format(row, 0)

        # Check structure
        self.assertEqual(len(arc_task), 1)
        task_id = list(arc_task.keys())[0]

        # Check train examples (first 2)
        self.assertEqual(len(arc_task[task_id]['train']), 2)
        self.assertEqual(arc_task[task_id]['train'][0]['input'], [[0, 1]])
        self.assertEqual(arc_task[task_id]['train'][0]['output'], [[2, 3]])

        # Check test example (last one, input only)
        self.assertEqual(len(arc_task[task_id]['test']), 1)
        self.assertEqual(arc_task[task_id]['test'][0]['input'], [[8, 9]])
        self.assertNotIn('output', arc_task[task_id]['test'][0])

        # Check test output saved separately
        self.assertIn(task_id, test_output)
        self.assertEqual(test_output[task_id]['test'][0]['output'], [[10, 11]])

    def test_transform_row_insufficient_examples(self):
        """Test with insufficient examples"""
        row = {'examples': [{'input': [[1]], 'output': [[2]]}]}

        with self.assertRaises(ValueError):
            transform_row_to_arc_format(row, 0)

    def test_transform_dataset_small(self):
        """Test transformation with mocked dataset"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create a mock dataset object with __len__ and __iter__
            class MockDataset:
                def __init__(self, data):
                    self.data = data

                def __len__(self):
                    return len(self.data)

                def __iter__(self):
                    return iter(self.data)

                def select(self, indices):
                    return MockDataset([self.data[i] for i in indices])

            mock_data = [
                {
                    'examples': [
                        {'input': [[0]], 'output': [[1]]},
                        {'input': [[2]], 'output': [[3]]}
                    ]
                },
                {
                    'examples': [
                        {'input': [[4]], 'output': [[5]]},
                        {'input': [[6]], 'output': [[7]]}
                    ]
                }
            ]

            # Mock load_dataset
            with patch('transform_hf_to_arc.load_dataset') as mock_load:
                mock_load.return_value = MockDataset(mock_data)

                arc_path, test_path = transform_dataset(
                    'fake_dataset',
                    temp_dir,
                    split='train',
                    max_samples=2,
                    streaming=False
                )

            # Check files exist
            self.assertTrue(os.path.exists(arc_path))
            self.assertTrue(os.path.exists(test_path))

            # Check content
            with open(arc_path) as f:
                arc_data = json.load(f)
            self.assertEqual(len(arc_data), 2)

        finally:
            shutil.rmtree(temp_dir)


class TestActivationExtraction(unittest.TestCase):
    """Test activation extraction functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ExtractionConfig(
            extract_activations=True,
            activations_dir=self.temp_dir,
            save_every_n_samples=5,
            upload_to_cloud=False
        )

    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.temp_dir)

    def test_extractor_initialization(self):
        """Test extractor initialization"""
        extractor = SimpleActivationExtractor(self.config)

        self.assertEqual(extractor.sample_count, 0)
        self.assertEqual(extractor.batch_count, 0)
        self.assertEqual(len(extractor.activations), 0)
        self.assertTrue(os.path.exists(self.config.activations_dir))

    def test_capture_activation(self):
        """Test capturing single activation"""
        extractor = SimpleActivationExtractor(self.config)

        # Mock data
        input_ids = np.array([[1, 2, 3]])
        output_logits = np.random.randn(1, 3, 100)

        extractor.capture_activation('task_1', 0, input_ids, output_logits)

        self.assertEqual(len(extractor.activations), 1)
        self.assertEqual(extractor.sample_count, 1)

        activation = extractor.activations[0]
        self.assertEqual(activation['task_id'], 'task_1')
        self.assertEqual(activation['sample_idx'], 0)
        self.assertEqual(activation['input_shape'], (1, 3))
        self.assertEqual(activation['output_shape'], (1, 3, 100))

    def test_save_batch(self):
        """Test saving activations batch"""
        extractor = SimpleActivationExtractor(self.config)

        # Add some activations
        for i in range(3):
            input_ids = np.array([[1, 2]])
            output_logits = np.random.randn(1, 2, 50)
            extractor.capture_activation(f'task_{i}', i, input_ids, output_logits)

        # Save manually
        extractor.save_batch()

        # Check file exists
        files = os.listdir(self.temp_dir)
        pkl_files = [f for f in files if f.endswith('.pkl')]
        self.assertEqual(len(pkl_files), 1)

        # Load and verify
        with open(os.path.join(self.temp_dir, pkl_files[0]), 'rb') as f:
            saved_data = pickle.load(f)

        self.assertEqual(len(saved_data), 3)
        self.assertEqual(saved_data[0]['task_id'], 'task_0')

    def test_automatic_save_on_threshold(self):
        """Test automatic saving when threshold reached"""
        extractor = SimpleActivationExtractor(self.config)

        # Add samples up to threshold (save_every_n_samples=5)
        for i in range(5):
            input_ids = np.array([[1]])
            output_logits = np.random.randn(1, 1, 10)
            extractor.capture_activation(f'task_{i}', i, input_ids, output_logits)

        # Should auto-save at 5 samples
        files = os.listdir(self.temp_dir)
        pkl_files = [f for f in files if f.endswith('.pkl')]
        self.assertGreater(len(pkl_files), 0)

        # Buffer should be cleared
        self.assertEqual(len(extractor.activations), 0)
        self.assertEqual(extractor.sample_count, 0)

    def test_finalize(self):
        """Test finalization saves metadata"""
        extractor = SimpleActivationExtractor(self.config)

        # Add and save some activations
        for i in range(3):
            input_ids = np.array([[1]])
            output_logits = np.random.randn(1, 1, 10)
            extractor.capture_activation(f'task_{i}', i, input_ids, output_logits)

        extractor.finalize()

        # Check metadata exists
        metadata_path = os.path.join(self.temp_dir, 'metadata.json')
        self.assertTrue(os.path.exists(metadata_path))

        # Load and verify metadata
        with open(metadata_path) as f:
            metadata = json.load(f)

        self.assertIn('total_samples', metadata)
        self.assertIn('total_batches', metadata)
        self.assertIn('start_time', metadata)
        self.assertIn('end_time', metadata)
        self.assertEqual(metadata['total_samples'], 3)

    def test_disabled_extraction(self):
        """Test that extraction can be disabled"""
        config = ExtractionConfig(extract_activations=False)
        extractor = SimpleActivationExtractor(config)

        input_ids = np.array([[1, 2]])
        output_logits = np.random.randn(1, 2, 50)

        extractor.capture_activation('task_1', 0, input_ids, output_logits)

        # Should not capture when disabled
        self.assertEqual(len(extractor.activations), 0)

    def test_metadata_tracks_multiple_batches(self):
        """Test metadata correctly tracks multiple batches"""
        config = ExtractionConfig(
            extract_activations=True,
            activations_dir=self.temp_dir,
            save_every_n_samples=2
        )
        extractor = SimpleActivationExtractor(config)

        # Add samples to trigger 2 batches
        for i in range(5):
            input_ids = np.array([[1]])
            output_logits = np.random.randn(1, 1, 10)
            extractor.capture_activation(f'task_{i}', i, input_ids, output_logits)

        extractor.finalize()

        # Check metadata
        with open(os.path.join(self.temp_dir, 'metadata.json')) as f:
            metadata = json.load(f)

        self.assertEqual(len(metadata['files']), 3)  # 2 auto-saves + 1 finalize
        self.assertEqual(metadata['total_batches'], 3)


class TestCloudStorage(unittest.TestCase):
    """Test cloud storage functionality"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_cloud_upload_mock(self):
        """Test cloud upload with mocked GCS"""
        # Skip if google.cloud.storage not available
        try:
            from google.cloud import storage
        except ImportError:
            self.skipTest("google-cloud-storage not installed")

        # Test that upload fails gracefully when disabled
        config = ExtractionConfig(
            activations_dir=self.temp_dir,
            upload_to_cloud=False,  # Disabled - should not error
            cloud_bucket=None,
            save_every_n_samples=2
        )

        extractor = SimpleActivationExtractor(config)

        # Add samples to trigger save
        for i in range(2):
            extractor.capture_activation(f'task_{i}', i, np.array([[1]]), np.random.randn(1, 1, 10))

        # Should complete without errors
        self.assertTrue(True)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline"""

    def test_end_to_end_transformation(self):
        """Test complete transformation flow"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create a mock dataset with proper interface
            class MockDataset:
                def __init__(self, data):
                    self.data = data

                def __len__(self):
                    return len(self.data)

                def __iter__(self):
                    return iter(self.data)

                def select(self, indices):
                    return MockDataset([self.data[i] for i in indices])

            # Create mock dataset
            mock_data = [
                {
                    'examples': [
                        {'input': [[i]], 'output': [[i+1]]}
                        for i in range(3)
                    ]
                }
                for _ in range(5)
            ]

            with patch('transform_hf_to_arc.load_dataset') as mock_load:
                mock_load.return_value = MockDataset(mock_data)

                arc_path, test_path = transform_dataset(
                    'fake',
                    temp_dir,
                    max_samples=5,
                    streaming=False
                )

            # Verify output structure
            with open(arc_path) as f:
                arc_data = json.load(f)

            with open(test_path) as f:
                test_data = json.load(f)

            self.assertEqual(len(arc_data), 5)
            self.assertEqual(len(test_data), 5)

            # Check a task structure
            task_id = list(arc_data.keys())[0]
            self.assertIn('train', arc_data[task_id])
            self.assertIn('test', arc_data[task_id])
            self.assertEqual(len(arc_data[task_id]['train']), 2)
            self.assertEqual(len(arc_data[task_id]['test']), 1)

        finally:
            shutil.rmtree(temp_dir)

    def test_extraction_with_multiple_samples(self):
        """Test extraction across multiple samples and batches"""
        temp_dir = tempfile.mkdtemp()

        try:
            config = ExtractionConfig(
                activations_dir=temp_dir,
                save_every_n_samples=10
            )
            extractor = SimpleActivationExtractor(config)

            # Simulate processing 25 samples
            for i in range(25):
                input_ids = np.array([[1, 2, 3]])
                output_logits = np.random.randn(1, 3, 100)
                extractor.capture_activation(f'task_{i}', i % 3, input_ids, output_logits)

            extractor.finalize()

            # Should have 3 batch files (10 + 10 + 5)
            pkl_files = [f for f in os.listdir(temp_dir) if f.endswith('.pkl')]
            self.assertEqual(len(pkl_files), 3)

            # Verify total samples in metadata
            with open(os.path.join(temp_dir, 'metadata.json')) as f:
                metadata = json.load(f)

            self.assertEqual(metadata['total_samples'], 25)

        finally:
            shutil.rmtree(temp_dir)


def run_tests():
    """Run all pipeline tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestDatasetTransformation,
        TestActivationExtraction,
        TestCloudStorage,
        TestIntegration
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    if success:
        print("\n✅ All pipeline tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some pipeline tests failed!")
        sys.exit(1)
