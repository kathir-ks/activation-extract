#!/usr/bin/env python3
"""
TPU Verification Tests for Refactored Codebase

Tests the refactored modules on TPU v5-8 hardware.
"""

import sys
import time
import jax
import jax.numpy as jnp
import numpy as np

def print_header(title):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def print_result(name, passed, details=""):
    status = "PASSED" if passed else "FAILED"
    icon = "✓" if passed else "✗"
    print(f"  {icon} {name}: {status}")
    if details:
        print(f"      {details}")

def test_jax_tpu():
    """Test JAX TPU availability"""
    print_header("Test 1: JAX TPU Availability")
    
    devices = jax.devices()
    device_count = jax.device_count()
    
    passed = device_count > 0 and 'tpu' in str(devices[0]).lower()
    print_result("TPU devices available", passed, f"{device_count} TPU chips")
    
    # Simple computation on TPU
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    result = float(y[0, 0])
    
    passed_compute = result == 1000.0
    print_result("TPU computation", passed_compute, f"dot product result: {result}")
    
    return passed and passed_compute

def test_model_imports():
    """Test model module imports"""
    print_header("Test 2: Model Module Imports")
    
    try:
        from model import (
            QwenConfig,
            get_default_config,
            get_qwen_0_5b_config,
            config_from_hf,
            RMSNorm,
            QwenMLP,
            QwenAttention,
            QwenModel,
            convert_hf_to_jax_weights,
            QwenModelWithActivations,
            create_model_with_hooks,
            DEFAULT_SAE_LAYERS,
            KVCacheConfig,
            create_kv_cache_buffers,
            initialize_multihost,
            create_device_mesh,
        )
        print_result("All model imports", True)
        return True
    except Exception as e:
        print_result("Model imports", False, str(e))
        return False

def test_arc_imports():
    """Test ARC module imports"""
    print_header("Test 3: ARC Module Imports")
    
    try:
        from arc import (
            GridEncoder,
            create_grid_encoder,
            MinimalGridEncoder,
            GridShapeEncoder,
            apply_data_augmentation,
            revert_data_augmentation,
            get_random_color_map,
        )
        print_result("All ARC imports", True)
        return True
    except Exception as e:
        print_result("ARC imports", False, str(e))
        return False

def test_data_imports():
    """Test data module imports"""
    print_header("Test 4: Data Module Imports")
    
    try:
        from data import (
            load_arc_dataset_jsonl,
            convert_hf_dataset_to_arc_format,
            ShardManager,
            create_sharded_dataset,
        )
        print_result("All data imports", True)
        return True
    except Exception as e:
        print_result("Data imports", False, str(e))
        return False

def test_extraction_imports():
    """Test extraction module imports"""
    print_header("Test 5: Extraction Module Imports")
    
    try:
        from extraction import (
            ActivationStorage,
            load_activation_shard,
            ExtractionConfig,
            run_extraction,
        )
        print_result("All extraction imports", True)
        return True
    except Exception as e:
        print_result("Extraction imports", False, str(e))
        return False

def test_model_config():
    """Test model configuration"""
    print_header("Test 6: Model Configuration")
    
    from model import QwenConfig, get_default_config, get_qwen_0_5b_config
    
    # Test default config
    config = get_default_config()
    passed = config.hidden_size == 896 and config.num_hidden_layers == 24
    print_result("Default config", passed, 
                 f"hidden={config.hidden_size}, layers={config.num_hidden_layers}")
    
    # Test 0.5B config
    config_05b = get_qwen_0_5b_config()
    passed_05b = config_05b.hidden_size == 896
    print_result("0.5B config", passed_05b, f"hidden={config_05b.hidden_size}")
    
    return passed and passed_05b

def test_grid_encoder():
    """Test grid encoder functionality"""
    print_header("Test 7: Grid Encoder")
    
    from arc import MinimalGridEncoder
    
    # Test minimal encoder directly
    encoder = MinimalGridEncoder()
    
    test_grid = [[1, 2, 3], [4, 5, 6]]
    text = encoder.to_text(test_grid)
    
    passed = len(text) > 0 and "1" in text
    print_result("Grid to text", passed, f"'{text[:50]}...'")
    
    return passed

def test_data_augmentation():
    """Test data augmentation"""
    print_header("Test 8: Data Augmentation")
    
    from arc import apply_data_augmentation, revert_data_augmentation
    
    task = {
        'train': [{'input': [[1, 2], [3, 4]], 'output': [[5, 6], [7, 8]]}],
        'test': [{'input': [[9, 0]], 'output': [[1, 2]]}]
    }
    
    # Apply augmentation
    aug_task = apply_data_augmentation(task, hflip=True, n_rot90=1)
    
    # Check it was modified
    passed = aug_task != task
    print_result("Augmentation applied", passed)
    
    return passed

def test_rmsnorm_on_tpu():
    """Test RMSNorm on TPU"""
    print_header("Test 9: RMSNorm on TPU")
    
    from model import RMSNorm, get_default_config
    import flax.linen as nn
    
    config = get_default_config()
    
    # Create RMSNorm
    norm = RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps)
    
    # Initialize
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 16, config.hidden_size))
    
    params = norm.init(key, x)
    
    # Forward pass
    start = time.time()
    y = norm.apply(params, x)
    jax.block_until_ready(y)
    elapsed = time.time() - start
    
    passed = y.shape == x.shape
    print_result("RMSNorm forward", passed, f"shape={y.shape}, time={elapsed*1000:.2f}ms")
    
    return passed

def test_qwen_mlp_on_tpu():
    """Test QwenMLP on TPU"""
    print_header("Test 10: QwenMLP on TPU")
    
    from model import QwenMLP, get_default_config
    
    config = get_default_config()
    
    # Create MLP - uses config not individual params
    mlp = QwenMLP(config=config)
    
    # Initialize
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 16, config.hidden_size))
    
    params = mlp.init(key, x)
    
    # Forward pass
    start = time.time()
    y = mlp.apply(params, x)
    jax.block_until_ready(y)
    elapsed = time.time() - start
    
    passed = y.shape == x.shape
    print_result("QwenMLP forward", passed, f"shape={y.shape}, time={elapsed*1000:.2f}ms")
    
    return passed

def test_kv_cache():
    """Test KV cache utilities"""
    print_header("Test 11: KV Cache Utilities")
    
    from model import KVCacheConfig, create_kv_cache_buffers, get_default_config
    
    config = get_default_config()
    batch_size = 2
    
    # KVCacheConfig uses max_prefill_length/max_decode_length
    cache_config = KVCacheConfig(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        max_prefill_length=128,
        max_decode_length=64,
        dtype=jnp.float32
    )
    
    # create_kv_cache_buffers takes config and batch_size
    kv_cache = create_kv_cache_buffers(cache_config, batch_size=batch_size)
    
    # New API returns dict with 'prefill' and 'ar' keys
    passed = kv_cache is not None and 'prefill' in kv_cache and 'ar' in kv_cache
    print_result("KV cache creation", passed, f"keys: {list(kv_cache.keys())}")
    
    # Check prefill shape [layers, batch, max_prefill_len, heads, dim]
    prefill_k = kv_cache['prefill']['k']
    expected_shape = (config.num_hidden_layers, batch_size, 128, 
                      config.num_key_value_heads, 
                      config.hidden_size // config.num_attention_heads)
    shape_ok = prefill_k.shape == expected_shape
    print_result("KV cache shape", shape_ok, f"shape={prefill_k.shape}")
    
    return passed and shape_ok

def test_activation_storage():
    """Test activation storage"""
    print_header("Test 12: Activation Storage")
    
    import tempfile
    import os
    from extraction import ActivationStorage
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ActivationStorage(
            output_dir=tmpdir,
            shard_size_gb=0.001,  # Small for testing
            verbose=False
        )
        
        # Add some activations
        for i in range(5):
            act = np.random.randn(16, 128).astype(np.float32)
            storage.add_activation(
                layer_idx=0,
                activation=act,
                sample_idx=i,
                text_preview=f"Sample {i}"
            )
        
        # Finalize
        summary = storage.finalize()
        
        passed = summary['total_samples'] == 5
        print_result("Activation storage", passed, f"stored {summary['total_samples']} samples")
        
        # Check files exist
        files = os.listdir(tmpdir)
        has_metadata = 'metadata.json' in files
        print_result("Metadata saved", has_metadata)
        
        return passed and has_metadata

def test_device_mesh():
    """Test device mesh creation"""
    print_header("Test 13: Device Mesh")
    
    from model import create_device_mesh
    
    try:
        mesh = create_device_mesh()
        
        passed = mesh is not None
        print_result("Device mesh created", passed, f"devices: {jax.device_count()}")
        return passed
    except Exception as e:
        print_result("Device mesh", False, str(e))
        return False

def run_all_tests():
    """Run all tests and report summary"""
    print("\n" + "="*70)
    print(" REFACTORED CODEBASE - TPU V5-8 VERIFICATION TESTS")
    print("="*70)
    print(f" JAX version: {jax.__version__}")
    print(f" Devices: {jax.device_count()} TPU chips")
    print("="*70)
    
    tests = [
        ("JAX TPU", test_jax_tpu),
        ("Model Imports", test_model_imports),
        ("ARC Imports", test_arc_imports),
        ("Data Imports", test_data_imports),
        ("Extraction Imports", test_extraction_imports),
        ("Model Config", test_model_config),
        ("Grid Encoder", test_grid_encoder),
        ("Data Augmentation", test_data_augmentation),
        ("RMSNorm on TPU", test_rmsnorm_on_tpu),
        ("QwenMLP on TPU", test_qwen_mlp_on_tpu),
        ("KV Cache", test_kv_cache),
        ("Activation Storage", test_activation_storage),
        ("Device Mesh", test_device_mesh),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ✗ {name}: EXCEPTION - {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    for name, passed in results:
        icon = "✓" if passed else "✗"
        status = "PASSED" if passed else "FAILED"
        print(f"  {icon} {name}: {status}")
    
    print("="*70)
    print(f" TOTAL: {passed_count}/{total_count} tests passed")
    print("="*70)
    
    return passed_count == total_count

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
