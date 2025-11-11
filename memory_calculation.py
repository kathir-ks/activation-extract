"""
Memory calculation for TPU v4-8 with Qwen 7B model extraction
"""

print("="*70)
print("TPU v4-8 MEMORY CALCULATION FOR QWEN 7B EXTRACTION")
print("="*70)

# TPU v4 specs
tpu_v4_cores = 4  # v4-8 has 4 chips (8 cores, but we use 4 for 1D mesh)
hbm_per_core_gb = 32  # Each TPU v4 chip has 32 GB HBM
total_hbm_gb = tpu_v4_cores * hbm_per_core_gb

print(f"\nTPU Configuration:")
print(f"  Chips: {tpu_v4_cores}")
print(f"  HBM per chip: {hbm_per_core_gb} GB")
print(f"  Total HBM: {total_hbm_gb} GB")

# Qwen 7B model parameters (bfloat16)
print(f"\n{'='*70}")
print("QWEN 7B MODEL MEMORY")
print("="*70)

# Model architecture
hidden_size = 4096
intermediate_size = 11008
num_layers = 28
num_attention_heads = 32
num_key_value_heads = 4
vocab_size = 151936

# Calculate parameter counts
embedding_params = vocab_size * hidden_size  # ~622M
per_layer_params = (
    # Self-attention
    hidden_size * hidden_size +  # q_proj
    hidden_size * (num_key_value_heads / num_attention_heads) * hidden_size +  # k_proj (GQA)
    hidden_size * (num_key_value_heads / num_attention_heads) * hidden_size +  # v_proj (GQA)
    hidden_size * hidden_size +  # o_proj
    # MLP
    hidden_size * intermediate_size +  # gate_proj
    hidden_size * intermediate_size +  # up_proj
    intermediate_size * hidden_size +  # down_proj
    # Layer norms
    hidden_size * 2  # input_layernorm + post_attention_layernorm
)

total_layer_params = per_layer_params * num_layers
lm_head_params = vocab_size * hidden_size  # ~622M
final_norm_params = hidden_size

total_params = embedding_params + total_layer_params + lm_head_params + final_norm_params

print(f"\nParameter counts:")
print(f"  Embeddings: {embedding_params/1e9:.2f}B")
print(f"  Per layer: {per_layer_params/1e9:.2f}B")
print(f"  Total layers ({num_layers}): {total_layer_params/1e9:.2f}B")
print(f"  LM head: {lm_head_params/1e9:.2f}B")
print(f"  Total: {total_params/1e9:.2f}B parameters")

# Memory for parameters (bfloat16 = 2 bytes)
bytes_per_param = 2  # bfloat16
model_memory_gb = (total_params * bytes_per_param) / (1024**3)

print(f"\nModel memory (bfloat16):")
print(f"  Parameters: {model_memory_gb:.2f} GB")

# With FSDP sharding across 4 chips
sharded_model_memory_gb = model_memory_gb / tpu_v4_cores
print(f"  Sharded across {tpu_v4_cores} chips: {sharded_model_memory_gb:.2f} GB per chip")

# User confirmed: 16 GB available per chip after loading model
print(f"\n  *** User-specified: 16 GB available per chip after model load ***")
available_per_chip = 16.0  # User's actual measurement
total_model_per_chip = hbm_per_core_gb - available_per_chip

print(f"  Model + overhead per chip: {total_model_per_chip:.2f} GB")
print(f"  Available per chip: {available_per_chip:.2f} GB")

# Available memory for activations
total_available = available_per_chip * tpu_v4_cores

print(f"\n{'='*70}")
print("AVAILABLE MEMORY FOR ACTIVATIONS")
print("="*70)
print(f"  Per chip: {available_per_chip:.2f} GB")
print(f"  Total: {total_available:.2f} GB")

# Activation memory calculation
print(f"\n{'='*70}")
print("ACTIVATION MEMORY PER SAMPLE")
print("="*70)

# For a single sequence, we extract activations at one layer
# Activation shape: [seq_len, hidden_size]
# Memory per activation: seq_len * hidden_size * 4 bytes (float32)

bytes_per_activation = 4  # float32

def calc_activation_size(seq_len, layers=1):
    """Calculate activation memory for one sample"""
    elements = seq_len * hidden_size * layers
    bytes_total = elements * bytes_per_activation
    mb = bytes_total / (1024**2)
    gb = bytes_total / (1024**3)
    return mb, gb

print(f"\nActivation size per sample (1 layer extracted):")
for seq_len in [256, 512, 1024, 2048]:
    mb, gb = calc_activation_size(seq_len, layers=1)
    print(f"  seq_len={seq_len:4d}: {mb:7.2f} MB ({gb:.4f} GB)")

print(f"\nActivation size per sample (ALL 28 layers extracted):")
for seq_len in [256, 512, 1024, 2048]:
    mb, gb = calc_activation_size(seq_len, layers=28)
    print(f"  seq_len={seq_len:4d}: {mb:7.2f} MB ({gb:.4f} GB)")

# Batch processing calculation
print(f"\n{'='*70}")
print("BATCH SIZE CALCULATION")
print("="*70)

print(f"\nWith available memory: {total_available:.2f} GB")
print(f"\nMaximum batch sizes (1 layer):")
for seq_len in [256, 512, 1024, 2048]:
    mb, gb = calc_activation_size(seq_len, layers=1)
    # During processing, we hold: input batch + output activations
    # Input: batch_size * seq_len * 4 bytes (int32)
    input_gb = (seq_len * 4) / (1024**3)
    total_per_sample = gb + input_gb
    max_batch = int(total_available / total_per_sample * 0.8)  # 80% utilization
    print(f"  seq_len={seq_len:4d}: batch_size={max_batch:3d} ({total_per_sample:.4f} GB per sample)")

print(f"\nMaximum batch sizes (ALL 28 layers):")
for seq_len in [256, 512, 1024, 2048]:
    mb, gb = calc_activation_size(seq_len, layers=28)
    input_gb = (seq_len * 4) / (1024**3)
    total_per_sample = gb + input_gb
    max_batch = int(total_available / total_per_sample * 0.8)
    print(f"  seq_len={seq_len:4d}: batch_size={max_batch:3d} ({total_per_sample:.4f} GB per sample)")

# Recommended configuration
print(f"\n{'='*70}")
print("RECOMMENDED CONFIGURATION")
print("="*70)

# Conservative recommendations
recommendations = [
    (256, 4, 1, "Fast extraction, minimal memory"),
    (512, 2, 1, "Balanced - good for most use cases"),
    (1024, 1, 1, "Long sequences, extract 1 layer at a time"),
    (256, 2, 28, "Extract ALL layers, shorter sequences"),
    (512, 1, 28, "Extract ALL layers, medium sequences"),
]

print("\nConfiguration: (seq_len, batch_size, num_layers)")
for seq_len, batch, layers, desc in recommendations:
    mb, gb = calc_activation_size(seq_len, layers=layers)
    total_batch_gb = gb * batch
    print(f"  seq_len={seq_len:4d}, batch={batch}, layers={layers:2d}: {total_batch_gb:.2f} GB - {desc}")

print(f"\n{'='*70}")
print("EXTRACTION SPEED ESTIMATE")
print("="*70)

# Based on our test: 7.43s/it for batch_size=2, seq_len=256, 2 layers, 0.5B model
# 7B model is ~14x larger, so roughly 14x slower per forward pass
# But with sharding, we're still limited by communication overhead

time_per_sample_0_5b = 7.43 / 2  # 3.7s per sample
scaling_factor = 2.0  # Conservative estimate (not 14x due to sharding efficiency)
time_per_sample_7b = time_per_sample_0_5b * scaling_factor

print(f"\nEstimated extraction time (Qwen 7B):")
print(f"  Per sample: ~{time_per_sample_7b:.1f} seconds")
print(f"  100 samples: ~{time_per_sample_7b * 100 / 60:.1f} minutes")
print(f"  1000 samples: ~{time_per_sample_7b * 1000 / 3600:.1f} hours")
print(f"  10000 samples: ~{time_per_sample_7b * 10000 / 3600:.1f} hours")

print(f"\nWith batch_size=2:")
time_per_batch = time_per_sample_7b * 0.7  # Batching efficiency
print(f"  Per batch: ~{time_per_batch:.1f} seconds")
print(f"  1000 samples (500 batches): ~{time_per_batch * 500 / 3600:.1f} hours")
print(f"  10000 samples (5000 batches): ~{time_per_batch * 5000 / 3600:.1f} hours")

print(f"\n{'='*70}")
