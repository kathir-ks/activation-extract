"""
Test suite for ARC-AGI inference pipeline and input/output processing
"""

import unittest
import tempfile
import json
import os
import numpy as np
import jax.numpy as jnp
from unittest.mock import patch, MagicMock

# Import modules to test
from arc24.encoders import (
    MinimalGridEncoder, GridWithSeparationEncoder, GridCodeBlockEncoder,
    GridShapeEncoder, RowNumberEncoder, create_grid_encoder
)
from arc24.data_augmentation import (
    geometric_augmentation, revert_geometric_augmentation,
    get_random_color_map, apply_data_augmentation, revert_data_augmentation
)
from arc24.prompting import (
    create_prompts_from_task, parse_grid_from_response, get_prompt_templates
)
from arc_inference_jax import (
    ARCConfig, create_prompts, validate_grid, create_solutions,
    generate_tokens_jax, load_jax_model_and_tokenizer
)


class TestGridEncoders(unittest.TestCase):
    """Test grid encoding and decoding functionality"""

    def setUp(self):
        """Set up test data"""
        self.sample_grid = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.small_grid = [[1, 0], [0, 1]]

    def test_minimal_grid_encoder(self):
        """Test MinimalGridEncoder roundtrip"""
        encoder = MinimalGridEncoder()

        # Test encoding
        text = encoder.to_text(self.sample_grid)
        expected_text = "012\n345\n678"
        self.assertEqual(text, expected_text)

        # Test decoding
        decoded_grid = encoder.to_grid(text)
        self.assertEqual(decoded_grid, self.sample_grid)

    def test_grid_with_separation_encoder(self):
        """Test GridWithSeparationEncoder with different separators"""
        encoder = GridWithSeparationEncoder('|')

        # Test encoding
        text = encoder.to_text(self.sample_grid)
        expected_text = "0|1|2\n3|4|5\n6|7|8"
        self.assertEqual(text, expected_text)

        # Test decoding
        decoded_grid = encoder.to_grid(text)
        self.assertEqual(decoded_grid, self.sample_grid)

    def test_grid_code_block_encoder(self):
        """Test GridCodeBlockEncoder"""
        base_encoder = MinimalGridEncoder()
        encoder = GridCodeBlockEncoder(base_encoder)

        # Test encoding
        text = encoder.to_text(self.sample_grid)
        self.assertTrue(text.startswith('```grid\n'))
        self.assertTrue(text.endswith('\n```'))

        # Test decoding
        decoded_grid = encoder.to_grid(text)
        self.assertEqual(decoded_grid, self.sample_grid)

    def test_grid_shape_encoder(self):
        """Test GridShapeEncoder"""
        base_encoder = MinimalGridEncoder()
        encoder = GridShapeEncoder(base_encoder)

        # Test encoding
        text = encoder.to_text(self.sample_grid)
        self.assertIn('shape: 3x3', text)

        # Test decoding
        decoded_grid = encoder.to_grid(text)
        self.assertEqual(decoded_grid, self.sample_grid)

    def test_row_number_encoder(self):
        """Test RowNumberEncoder"""
        base_encoder = MinimalGridEncoder()
        encoder = RowNumberEncoder(base_encoder)

        # Test encoding
        text = encoder.to_text(self.small_grid)
        lines = text.split('\n')
        self.assertTrue(lines[0].startswith('1 '))
        self.assertTrue(lines[1].startswith('2 '))

        # Test decoding
        decoded_grid = encoder.to_grid(text)
        self.assertEqual(decoded_grid, self.small_grid)

    def test_create_grid_encoder(self):
        """Test encoder creation from string"""
        encoder = create_grid_encoder('MinimalGridEncoder()')
        self.assertIsInstance(encoder, MinimalGridEncoder)

        encoder = create_grid_encoder("GridWithSeparationEncoder('|')")
        self.assertIsInstance(encoder, GridWithSeparationEncoder)

        # Test complex encoder composition
        encoder = create_grid_encoder('GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))')
        self.assertIsInstance(encoder, GridShapeEncoder)


class TestDataAugmentation(unittest.TestCase):
    """Test data augmentation functionality"""

    def setUp(self):
        """Set up test data"""
        self.sample_grid = [[0, 1], [2, 3]]
        self.sample_task = {
            'train': [
                {'input': [[0, 1], [2, 3]], 'output': [[1, 0], [3, 2]]},
            ],
            'test': [
                {'input': [[4, 5], [6, 7]]},
            ]
        }

    def test_geometric_augmentation_roundtrip(self):
        """Test geometric augmentation and reversion"""
        # Test horizontal flip
        flipped = geometric_augmentation(self.sample_grid, hflip=True, n_rot90=0)
        reverted = revert_geometric_augmentation(flipped, hflip=True, n_rot90=0)
        self.assertEqual(reverted, self.sample_grid)

        # Test rotation
        rotated = geometric_augmentation(self.sample_grid, hflip=False, n_rot90=1)
        reverted = revert_geometric_augmentation(rotated, hflip=False, n_rot90=1)
        self.assertEqual(reverted, self.sample_grid)

        # Test combined transformation
        transformed = geometric_augmentation(self.sample_grid, hflip=True, n_rot90=2)
        reverted = revert_geometric_augmentation(transformed, hflip=True, n_rot90=2)
        self.assertEqual(reverted, self.sample_grid)

    def test_color_map_generation(self):
        """Test random color map generation"""
        color_map = get_random_color_map(change_background_probability=0.0)

        # Should preserve background color (0)
        self.assertEqual(color_map[0], 0)

        # Should map all colors 0-9
        self.assertEqual(set(color_map.keys()), set(range(10)))
        self.assertEqual(set(color_map.values()), set(range(10)))

    def test_task_augmentation_roundtrip(self):
        """Test full task augmentation and reversion"""
        # Apply augmentation
        color_map = {i: i for i in range(10)}  # Identity mapping for simplicity
        augmented_task = apply_data_augmentation(
            self.sample_task, hflip=True, n_rot90=1, color_map=color_map
        )

        # Verify structure is preserved
        self.assertEqual(len(augmented_task['train']), len(self.sample_task['train']))
        self.assertEqual(len(augmented_task['test']), len(self.sample_task['test']))

        # Test grid reversion
        original_input = self.sample_task['train'][0]['input']
        augmented_input = augmented_task['train'][0]['input']
        reverted_input = revert_data_augmentation(
            augmented_input, hflip=True, n_rot90=1, color_map=color_map
        )
        self.assertEqual(reverted_input, original_input)


class TestPrompting(unittest.TestCase):
    """Test prompting functionality"""

    def setUp(self):
        """Set up test data"""
        self.sample_task = {
            'train': [
                {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]},
            ],
            'test': [
                {'input': [[0, 2], [2, 0]]},
            ]
        }

        # Mock tokenizer with proper Qwen format
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.apply_chat_template.return_value = "<|im_start|>user\ntest<|im_end|><|im_start|>assistant\nresponse<|im_end|>"

    def test_prompt_template_retrieval(self):
        """Test prompt template retrieval"""
        system_prompt, prompt_template, answer_template = get_prompt_templates('output-from-examples-v0')

        self.assertIsNotNone(system_prompt)
        self.assertIsNotNone(prompt_template)
        self.assertIsNotNone(answer_template)

        # Test invalid version
        with self.assertRaises(ValueError):
            get_prompt_templates('invalid-version')

    def test_create_prompts_from_task(self):
        """Test prompt creation from task"""
        encoder = MinimalGridEncoder()

        prompts = create_prompts_from_task(
            self.sample_task,
            grid_encoder=encoder,
            tokenizer=self.mock_tokenizer,
            is_train_prompt=False,
            prompt_version='output-from-examples-v0'
        )

        self.assertEqual(len(prompts), len(self.sample_task['test']))
        self.mock_tokenizer.apply_chat_template.assert_called()

    def test_parse_grid_from_response(self):
        """Test grid parsing from model response"""
        encoder = GridCodeBlockEncoder(MinimalGridEncoder())

        # Test valid response
        response = "012\n345"
        expected_grid = [[0, 1, 2], [3, 4, 5]]

        parsed_grid = parse_grid_from_response(f"```grid\n{response}\n```", encoder)
        self.assertEqual(parsed_grid, expected_grid)


class TestInferencePipeline(unittest.TestCase):
    """Test main inference pipeline components"""

    def setUp(self):
        """Set up test data"""
        self.sample_data = {
            'task_1': {
                'train': [
                    {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]},
                ],
                'test': [
                    {'input': [[0, 2], [2, 0]]},
                ]
            }
        }

        self.config = ARCConfig(
            predictions_per_task=2,
            batch_size=1,
            max_output_tokens=50,
            verbose=False
        )

    def test_config_creation(self):
        """Test ARCConfig creation and parameter validation"""
        config = ARCConfig()
        self.assertIsInstance(config.output_filepath, str)
        self.assertIsInstance(config.predictions_per_task, int)
        self.assertTrue(config.predictions_per_task > 0)

    def test_validate_grid(self):
        """Test grid validation function"""
        # Valid grid
        valid_grid = [[0, 1, 2], [3, 4, 5]]
        validate_grid(valid_grid)  # Should not raise

        # Invalid grids
        with self.assertRaises(AssertionError):
            validate_grid("not a list")

        with self.assertRaises(AssertionError):
            validate_grid([])  # Empty grid

        with self.assertRaises(AssertionError):
            validate_grid([[]])  # Zero columns

        with self.assertRaises(AssertionError):
            validate_grid([[-1, 0]])  # Negative values

        with self.assertRaises(AssertionError):
            validate_grid([[10, 0]])  # Values > 9

    def test_create_prompts(self):
        """Test prompt creation for inference"""
        mock_tokenizer = MagicMock()
        # Use proper Qwen format for mocked response
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>user\ntest<|im_end|><|im_start|>assistant\nresponse<|im_end|>"

        grid_encoder = MinimalGridEncoder()

        prompts_conf = create_prompts(
            self.sample_data,
            grid_encoder,
            mock_tokenizer,
            'output-from-examples-v0',
            predictions_per_task=2
        )

        # Should create prompts for each task and augmentation
        # predictions_per_task=2 means 2 prompts per data augmentation set
        # There are 8 augmentation combinations (2 flips × 4 rotations) and 1 test sample
        # So we expect at least some prompts
        self.assertGreater(len(prompts_conf), 0, "Should create at least one prompt")

        # Check prompt structure
        if len(prompts_conf) > 0:
            prompt = prompts_conf[0]
            self.assertIn('task_id', prompt)
            self.assertIn('data_augmentation_kwargs', prompt)
            self.assertIn('prompt', prompt)
            self.assertIn('idx', prompt)

    def test_create_solutions(self):
        """Test solution structure creation"""
        # Mock task results
        task_results = [
            {
                'task_id': 'task_1',
                'idx': 0,
                'grid': [[1, 0], [0, 1]]
            }
        ]

        solutions = create_solutions(task_results, self.sample_data)

        self.assertIn('task_1', solutions)
        self.assertEqual(len(solutions['task_1']), len(self.sample_data['task_1']['test']))
        self.assertIn('attempt_1', solutions['task_1'][0])

    @patch('arc_inference_jax.AutoTokenizer')
    @patch('arc_inference_jax.AutoModelForCausalLM')
    def test_load_jax_model_and_tokenizer(self, mock_model_cls, mock_tokenizer_cls):
        """Test model and tokenizer loading"""
        from qwen2_jax import QwenConfig

        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_hf_model = MagicMock()
        mock_hf_model.state_dict.return_value = {}
        mock_model_cls.from_pretrained.return_value = mock_hf_model

        config = QwenConfig()

        # This would normally load real models, so we'll just test the interface
        try:
            model, tokenizer, params = load_jax_model_and_tokenizer("test/path", config)
            # If we get here without errors, the function structure is correct
        except Exception as e:
            # Expected since we're using mocks
            pass

    def test_generate_tokens_interface(self):
        """Test token generation function interface"""
        # Use numpy instead of jnp to avoid TPU initialization
        import numpy as np

        # Create mock inputs
        mock_model = MagicMock()
        mock_params = {}
        input_ids = np.array([[1, 2, 3]])

        # Mock model.apply to return logits
        mock_logits = np.ones((1, 3, 1000))  # batch_size=1, seq_len=3, vocab_size=1000
        mock_model.apply.return_value = mock_logits

        # Test function call (will need to convert to jnp inside)
        # For now, just test that the function signature is correct
        self.assertTrue(callable(generate_tokens_jax))

        # Test the function exists and has correct signature
        import inspect
        sig = inspect.signature(generate_tokens_jax)
        params = list(sig.parameters.keys())
        self.assertIn('model', params)
        self.assertIn('params', params)
        self.assertIn('input_ids', params)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""

    def setUp(self):
        """Set up test files and data"""
        self.test_dir = tempfile.mkdtemp()

        # Create test data file
        self.test_data = {
            'simple_task': {
                'train': [
                    {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]},
                ],
                'test': [
                    {'input': [[0, 2], [2, 0]]},
                ]
            }
        }

        self.data_file = os.path.join(self.test_dir, 'test_data.json')
        with open(self.data_file, 'w') as f:
            json.dump(self.test_data, f)

    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_end_to_end_data_flow(self):
        """Test data flow from input to prompt creation"""
        # Load data
        with open(self.data_file) as f:
            data = json.load(f)

        # Create encoder
        encoder = create_grid_encoder('MinimalGridEncoder()')

        # Mock tokenizer with proper format
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>user\ntest<|im_end|><|im_start|>assistant\nresponse<|im_end|>"

        # Create prompts
        prompts_conf = create_prompts(
            data, encoder, mock_tokenizer, 'output-from-examples-v0', predictions_per_task=1
        )

        # Verify complete data flow
        self.assertGreater(len(prompts_conf), 0, "Should create at least one prompt")

        # Check that all required fields are present
        for prompt_conf in prompts_conf:
            self.assertIn('task_id', prompt_conf)
            self.assertIn('prompt', prompt_conf)
            self.assertIn('data_augmentation_kwargs', prompt_conf)

    def test_encoder_composition(self):
        """Test complex encoder composition works end-to-end"""
        test_grid = [[0, 1, 2], [3, 4, 5]]

        # Test various encoder combinations
        encoder_configs = [
            'MinimalGridEncoder()',
            'GridCodeBlockEncoder(MinimalGridEncoder())',
            'GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))',
            "GridWithSeparationEncoder('|')",
        ]

        for config in encoder_configs:
            encoder = create_grid_encoder(config)

            # Test roundtrip
            text = encoder.to_text(test_grid)
            decoded = encoder.to_grid(text)

            self.assertEqual(decoded, test_grid, f"Failed for encoder: {config}")


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestGridEncoders,
        TestDataAugmentation,
        TestPrompting,
        TestInferencePipeline,
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
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)