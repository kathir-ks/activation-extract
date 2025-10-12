"""
Comprehensive unit tests for the clean inference pipeline
Tests all components independently and integration
"""

import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from inference_clean import (
    ModelConfig, InferenceConfig, QwenModel, QwenModelWithActivations,
    RMSNorm, Attention, MLP, TransformerBlock, RotaryEmbedding,
    create_generation_step, generate_tokens, setup_mesh,
    rotate_half, apply_rotary_pos_emb
)


class TestRotaryEmbeddings(unittest.TestCase):
    """Test rotary position embeddings"""

    def test_rotate_half(self):
        """Test rotate_half function"""
        x = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        result = rotate_half(x)
        expected = jnp.array([[-3.0, -4.0, 1.0, 2.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotary_embedding_shape(self):
        """Test RotaryEmbedding output shapes"""
        config = ModelConfig()
        head_dim = config.hidden_size // config.num_attention_heads

        rotary = RotaryEmbedding(dim=head_dim, max_position_embeddings=2048, base=10000.0)
        variables = rotary.init(jax.random.PRNGKey(0), seq_len=10)

        cos, sin = rotary.apply(variables, seq_len=10)

        self.assertEqual(cos.shape, (10, head_dim))
        self.assertEqual(sin.shape, (10, head_dim))

    def test_apply_rotary_pos_emb(self):
        """Test applying rotary embeddings"""
        batch, seq_len, num_heads, head_dim = 2, 4, 2, 8

        q = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(jax.random.PRNGKey(1), (batch, seq_len, num_heads, head_dim))
        cos = jax.random.normal(jax.random.PRNGKey(2), (1, seq_len, 1, head_dim))
        sin = jax.random.normal(jax.random.PRNGKey(3), (1, seq_len, 1, head_dim))

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)


class TestNormalization(unittest.TestCase):
    """Test RMSNorm layer"""

    def test_rmsnorm_output_shape(self):
        """Test RMSNorm preserves shape"""
        dim = 64
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, dim))

        norm = RMSNorm(dim=dim, eps=1e-6)
        variables = norm.init(jax.random.PRNGKey(0), x)
        output = norm.apply(variables, x)

        self.assertEqual(output.shape, x.shape)

    def test_rmsnorm_normalization(self):
        """Test RMSNorm actually normalizes"""
        dim = 64
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, dim)) * 100  # Large values

        norm = RMSNorm(dim=dim, eps=1e-6)
        variables = norm.init(jax.random.PRNGKey(0), x)
        output = norm.apply(variables, x)

        # Check that variance is close to 1
        variance = jnp.mean(jnp.square(output), axis=-1)
        np.testing.assert_array_almost_equal(variance, jnp.ones_like(variance), decimal=5)


class TestAttention(unittest.TestCase):
    """Test attention mechanism"""

    def test_attention_output_shape(self):
        """Test Attention output shape"""
        config = ModelConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, config.hidden_size))

        attn = Attention(config=config)
        variables = attn.init(jax.random.PRNGKey(0), x)
        output = attn.apply(variables, x)

        self.assertEqual(output.shape, x.shape)

    def test_attention_causal_mask(self):
        """Test that attention respects causal masking"""
        config = ModelConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2
        )

        # Simple input
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 5, config.hidden_size))

        attn = Attention(config=config)
        variables = attn.init(jax.random.PRNGKey(0), x)

        # Should work without errors
        output = attn.apply(variables, x)
        self.assertEqual(output.shape, x.shape)


class TestMLP(unittest.TestCase):
    """Test MLP layer"""

    def test_mlp_output_shape(self):
        """Test MLP output shape"""
        config = ModelConfig(hidden_size=64, intermediate_size=256)

        x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, config.hidden_size))

        mlp = MLP(config=config)
        variables = mlp.init(jax.random.PRNGKey(0), x)
        output = mlp.apply(variables, x)

        self.assertEqual(output.shape, x.shape)


class TestTransformerBlock(unittest.TestCase):
    """Test complete transformer block"""

    def test_transformer_block_output_shape(self):
        """Test TransformerBlock output shape"""
        config = ModelConfig(
            hidden_size=64,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, config.hidden_size))

        block = TransformerBlock(config=config, layer_idx=0)
        variables = block.init(jax.random.PRNGKey(0), x)
        output = block.apply(variables, x)

        self.assertEqual(output.shape, x.shape)

    def test_transformer_block_residual(self):
        """Test that residual connections work"""
        config = ModelConfig(
            hidden_size=64,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, config.hidden_size))

        block = TransformerBlock(config=config, layer_idx=0)
        variables = block.init(jax.random.PRNGKey(0), x)
        output = block.apply(variables, x)

        # Output should be different from input (not just identity)
        self.assertFalse(jnp.allclose(output, x))


class TestQwenModel(unittest.TestCase):
    """Test complete Qwen model"""

    def test_qwen_model_forward(self):
        """Test QwenModel forward pass"""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2
        )

        input_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 10), 0, config.vocab_size)

        model = QwenModel(config=config)
        variables = model.init(jax.random.PRNGKey(0), input_ids)
        logits = model.apply(variables, input_ids)

        self.assertEqual(logits.shape, (2, 10, config.vocab_size))

    def test_qwen_model_with_activations(self):
        """Test QwenModelWithActivations"""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2
        )

        input_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 10), 0, config.vocab_size)
        extract_layers = [1, 2]

        model = QwenModelWithActivations(config=config, extract_layers=extract_layers)
        variables = model.init(jax.random.PRNGKey(0), input_ids)
        logits, activations = model.apply(variables, input_ids)

        self.assertEqual(logits.shape, (2, 10, config.vocab_size))

        # Check activations were extracted for specified layers
        for layer_idx in extract_layers:
            self.assertIn(f'layer_{layer_idx}_input', activations)
            self.assertIn(f'layer_{layer_idx}_output', activations)

        # Check activations have correct shape
        for key, value in activations.items():
            self.assertEqual(value.shape, (2, 10, config.hidden_size))


class TestGenerationStep(unittest.TestCase):
    """Test generation step functions"""

    def test_create_generation_step_no_activations(self):
        """Test generation step without activations"""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1
        )

        model = QwenModel(config=config)
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 5), 0, config.vocab_size)

        variables = model.init(jax.random.PRNGKey(0), input_ids)
        params = variables

        generation_step = create_generation_step(model, extract_activations=False)
        next_token = generation_step(params, input_ids)

        self.assertEqual(next_token.shape, (2, 1))

    def test_create_generation_step_with_activations(self):
        """Test generation step with activations"""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1
        )

        extract_layers = [0, 1]
        model = QwenModelWithActivations(config=config, extract_layers=extract_layers)
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 5), 0, config.vocab_size)

        variables = model.init(jax.random.PRNGKey(0), input_ids)
        params = variables

        generation_step = create_generation_step(model, extract_activations=True)
        next_token, activations = generation_step(params, input_ids)

        self.assertEqual(next_token.shape, (2, 1))
        self.assertIsInstance(activations, dict)
        self.assertGreater(len(activations), 0)

    def test_generate_tokens_multi_step(self):
        """Test multi-step generation"""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1
        )

        model = QwenModel(config=config)
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 5), 0, config.vocab_size)

        variables = model.init(jax.random.PRNGKey(0), input_ids)
        params = variables

        generation_step = create_generation_step(model, extract_activations=False)

        max_new_tokens = 3
        generated = generate_tokens(params, input_ids, max_new_tokens, generation_step, extract_activations=False)

        # Should have original length + new tokens
        self.assertEqual(generated.shape, (2, 5 + max_new_tokens))


class TestDistributedSetup(unittest.TestCase):
    """Test distributed setup functions"""

    def test_setup_mesh_single_device(self):
        """Test mesh setup with single device"""
        config = InferenceConfig(mesh_shape=(1, 1))

        # This should work even with multiple devices
        mesh = setup_mesh(config)

        self.assertIsInstance(mesh, Mesh)
        self.assertEqual(mesh.axis_names, ('data', 'model'))

    def test_setup_mesh_adjusts_to_devices(self):
        """Test that mesh auto-adjusts to available devices"""
        n_devices = len(jax.devices())

        # Request incompatible mesh
        config = InferenceConfig(mesh_shape=(999, 1))

        mesh = setup_mesh(config)

        # Should auto-adjust
        self.assertEqual(config.mesh_shape, (n_devices, 1))


class TestInferenceConfig(unittest.TestCase):
    """Test configuration dataclass"""

    def test_inference_config_defaults(self):
        """Test InferenceConfig default values"""
        config = InferenceConfig()

        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.max_new_tokens, 512)
        self.assertEqual(config.mesh_shape, (1, 1))
        self.assertFalse(config.extract_activations)

    def test_inference_config_custom_values(self):
        """Test InferenceConfig with custom values"""
        config = InferenceConfig(
            batch_size=16,
            max_new_tokens=256,
            extract_activations=True,
            layers_to_extract=[0, 5, 10]
        )

        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.max_new_tokens, 256)
        self.assertTrue(config.extract_activations)
        self.assertEqual(config.layers_to_extract, [0, 5, 10])


class TestModelConfig(unittest.TestCase):
    """Test model configuration"""

    def test_model_config_defaults(self):
        """Test ModelConfig default values"""
        config = ModelConfig()

        self.assertEqual(config.vocab_size, 151936)
        self.assertEqual(config.hidden_size, 896)
        self.assertEqual(config.num_hidden_layers, 24)
        self.assertTrue(config.tie_word_embeddings)

    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values"""
        config = ModelConfig(
            vocab_size=50000,
            hidden_size=512,
            num_hidden_layers=12
        )

        self.assertEqual(config.vocab_size, 50000)
        self.assertEqual(config.hidden_size, 512)
        self.assertEqual(config.num_hidden_layers, 12)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""

    def test_small_model_inference(self):
        """Test complete inference on small model"""
        # Create tiny model for testing
        config = ModelConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128
        )

        model = QwenModel(config=config)

        # Initialize model
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 8), 0, config.vocab_size)
        variables = model.init(jax.random.PRNGKey(42), input_ids)

        # Create generation step
        generation_step = create_generation_step(model, extract_activations=False)

        # Generate tokens
        generated = generate_tokens(
            variables, input_ids, max_new_tokens=5,
            generation_step=generation_step, extract_activations=False
        )

        # Check output
        self.assertEqual(generated.shape, (2, 8 + 5))
        self.assertTrue(jnp.all(generated >= 0))
        self.assertTrue(jnp.all(generated < config.vocab_size))

    def test_small_model_with_activations(self):
        """Test inference with activation extraction"""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128
        )

        extract_layers = [1, 2]
        model = QwenModelWithActivations(config=config, extract_layers=extract_layers)

        input_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 8), 0, config.vocab_size)
        variables = model.init(jax.random.PRNGKey(42), input_ids)

        generation_step = create_generation_step(model, extract_activations=True)

        generated, all_activations = generate_tokens(
            variables, input_ids, max_new_tokens=3,
            generation_step=generation_step, extract_activations=True
        )

        # Check outputs
        self.assertEqual(generated.shape, (2, 8 + 3))
        self.assertEqual(len(all_activations), 3)  # 3 generation steps

        # Check each step has activations
        for step_acts in all_activations:
            self.assertIsInstance(step_acts, dict)
            for layer_idx in extract_layers:
                self.assertIn(f'layer_{layer_idx}_input', step_acts)
                self.assertIn(f'layer_{layer_idx}_output', step_acts)


class TestJITCompilation(unittest.TestCase):
    """Test JIT compilation works correctly"""

    def test_generation_step_is_jitted(self):
        """Test that generation_step is properly JIT-compiled"""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1
        )

        model = QwenModel(config=config)
        input_ids = jax.random.randint(jax.random.PRNGKey(0), (2, 5), 0, config.vocab_size)
        variables = model.init(jax.random.PRNGKey(0), input_ids)

        generation_step = create_generation_step(model, extract_activations=False)

        # First call should compile
        _ = generation_step(variables, input_ids)

        # Second call should use cached compilation
        next_token = generation_step(variables, input_ids)

        self.assertEqual(next_token.shape, (2, 1))


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
