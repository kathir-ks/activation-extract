"""Quick test of parallel generation"""

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_fixed import QwenModelFixed
from generate_parallel import generate_parallel
from transformers import AutoModelForCausalLM
import torch

print("Loading model...")
model_path = "KathirKs/qwen-2.5-0.5b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load HF model for weight conversion
hf_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    trust_remote_code=True
)

# Initialize JAX model
config = QwenConfig(
    vocab_size=151936,
    hidden_size=896,
    intermediate_size=4864,
    num_hidden_layers=24,
    num_attention_heads=14,
    num_key_value_heads=2,
    max_position_embeddings=32768,
    rope_theta=1000000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True
)

jax_model = QwenModelFixed(config)

# Initialize parameters
dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
rng = jax.random.PRNGKey(0)
params = jax_model.init(rng, dummy_input)

# Convert weights
print("Converting weights...")
converted_params = convert_hf_to_jax_weights(hf_model, config)
params = {'params': converted_params}

# Test prompts
prompts = [
    "Hello, how are you?",
    "What is 2+2?",
    "The capital of France is",
    "Python is a"
]

print(f"\nTesting with {len(prompts)} prompts, 50 tokens each")
print(f"Devices: {jax.devices()}")

# Generate
outputs = generate_parallel(
    jax_model,
    params,
    tokenizer,
    prompts,
    max_tokens=100,
    temperature=0.0,
    verbose=True
)

print("\n" + "="*60)
print("Results:")
for i, output in enumerate(outputs):
    print(f"\nPrompt {i}: {prompts[i]}")
    print(f"Output: {output[:100]}...")  # First 100 chars
