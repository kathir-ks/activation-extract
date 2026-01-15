"""
Activation Extraction Utilities

Provides unified extraction pipelines for:
- Single-host TPU extraction
- Multi-host TPU extraction
- Various dataset formats (ARC, FineWeb, etc.)
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class ExtractionConfig:
    """Configuration for activation extraction."""
    
    # Model settings
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    layers_to_extract: Optional[List[int]] = None
    max_seq_length: int = 2048
    batch_size: int = 8
    
    # Dataset settings
    dataset_path: Optional[str] = None
    dataset_type: str = "arc"  # "arc", "fineweb", "jsonl"
    max_samples: Optional[int] = None
    
    # Output settings
    output_dir: str = "./activations"
    shard_size_gb: float = 1.0
    compress_shards: bool = True
    
    # GCS settings
    upload_to_gcs: bool = False
    gcs_bucket: Optional[str] = None
    gcs_prefix: str = "activations"
    delete_local_after_upload: bool = False
    
    # Multi-host settings
    use_multihost: bool = False
    machine_id: int = 0
    total_machines: int = 1
    
    # Processing settings
    verbose: bool = True
    random_seed: Optional[int] = None


def run_extraction(config: ExtractionConfig) -> Dict[str, Any]:
    """
    Run activation extraction based on configuration.
    
    Automatically selects single-host or multi-host based on config.
    
    Args:
        config: Extraction configuration
        
    Returns:
        Dictionary with extraction results
    """
    if config.use_multihost:
        return run_extraction_multi_host(config)
    else:
        return run_extraction_single_host(config)


def run_extraction_single_host(config: ExtractionConfig) -> Dict[str, Any]:
    """
    Run activation extraction on a single host.
    
    Args:
        config: Extraction configuration
        
    Returns:
        Dictionary with extraction results
    """
    import jax
    from transformers import AutoTokenizer
    
    from ..model import (
        QwenConfig,
        QwenModelWithActivations,
        create_model_with_hooks,
        config_from_hf,
        convert_hf_to_jax_weights,
        DEFAULT_SAE_LAYERS,
    )
    from .storage import ActivationStorage
    
    if config.verbose:
        print("="*70)
        print("Activation Extraction - Single Host Mode")
        print("="*70)
        print(f"  Model: {config.model_path}")
        print(f"  Dataset: {config.dataset_path}")
        print(f"  Output: {config.output_dir}")
        print(f"  Layers: {config.layers_to_extract or 'default'}")
        print("="*70)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    
    # Load model config
    model_config = config_from_hf(config.model_path)
    
    # Determine layers to extract
    layers = config.layers_to_extract or DEFAULT_SAE_LAYERS
    
    # Create model with hooks
    model = create_model_with_hooks(model_config, layers)
    
    # Load weights
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(config.model_path)
    params = convert_hf_to_jax_weights(hf_model, model_config)
    del hf_model
    
    # Initialize storage
    storage = ActivationStorage(
        output_dir=config.output_dir,
        upload_to_gcs=config.upload_to_gcs,
        gcs_bucket=config.gcs_bucket,
        gcs_prefix=config.gcs_prefix,
        shard_size_gb=config.shard_size_gb,
        compress_shards=config.compress_shards,
        delete_local_after_upload=config.delete_local_after_upload,
        verbose=config.verbose
    )
    
    # Load dataset
    samples = _load_dataset(config)
    
    # Process samples
    total_processed = 0
    for batch_start in range(0, len(samples), config.batch_size):
        batch_end = min(batch_start + config.batch_size, len(samples))
        batch = samples[batch_start:batch_end]
        
        # Tokenize
        texts = [s['text'] for s in batch]
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors='np'
        )
        
        # Forward pass with activation extraction
        input_ids = inputs['input_ids']
        activations, _ = model.apply(params, input_ids, extract_activations=True)
        
        # Store activations
        for i, sample in enumerate(batch):
            for layer_idx, layer_acts in activations.items():
                storage.add_activation(
                    layer_idx=layer_idx,
                    activation=np.array(layer_acts[i]),
                    sample_idx=batch_start + i,
                    text_preview=texts[i]
                )
        
        total_processed += len(batch)
        if config.verbose and total_processed % 100 == 0:
            print(f"  Processed {total_processed}/{len(samples)} samples")
    
    # Finalize
    summary = storage.finalize()
    summary['total_processed'] = total_processed
    
    return summary


def run_extraction_multi_host(config: ExtractionConfig) -> Dict[str, Any]:
    """
    Run activation extraction on multiple TPU hosts.
    
    Args:
        config: Extraction configuration
        
    Returns:
        Dictionary with extraction results
    """
    import jax
    from transformers import AutoTokenizer
    
    from ..model import (
        QwenConfig,
        QwenModelWithActivations,
        create_model_with_hooks,
        config_from_hf,
        convert_hf_to_jax_weights,
        DEFAULT_SAE_LAYERS,
        initialize_multihost,
        create_device_mesh,
        shard_params,
        extract_activations_sharded,
        pad_sequences,
    )
    from .storage import ActivationStorage
    
    if config.verbose:
        print("="*70)
        print("Activation Extraction - Multi-Host Mode")
        print("="*70)
        print(f"  Model: {config.model_path}")
        print(f"  Dataset: {config.dataset_path}")
        print(f"  Machine: {config.machine_id}/{config.total_machines}")
        print(f"  Output: {config.output_dir}")
        print("="*70)
    
    # Initialize multi-host
    process_id, num_processes = initialize_multihost()
    
    if config.verbose:
        print(f"  Process ID: {process_id}/{num_processes}")
    
    # Create device mesh
    mesh = create_device_mesh()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    
    # Load model config
    model_config = config_from_hf(config.model_path)
    
    # Determine layers to extract
    layers = config.layers_to_extract or DEFAULT_SAE_LAYERS
    
    # Create model with hooks
    model = create_model_with_hooks(model_config, layers)
    
    # Load and shard weights
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(config.model_path)
    params = convert_hf_to_jax_weights(hf_model, model_config)
    del hf_model
    
    # Shard parameters
    sharded_params = shard_params(params, mesh)
    
    # Initialize storage (with machine-specific prefix)
    machine_prefix = f"{config.gcs_prefix}/machine_{config.machine_id:03d}"
    storage = ActivationStorage(
        output_dir=f"{config.output_dir}/machine_{config.machine_id:03d}",
        upload_to_gcs=config.upload_to_gcs,
        gcs_bucket=config.gcs_bucket,
        gcs_prefix=machine_prefix,
        shard_size_gb=config.shard_size_gb,
        compress_shards=config.compress_shards,
        delete_local_after_upload=config.delete_local_after_upload,
        verbose=config.verbose
    )
    
    # Load dataset (with machine-based sharding)
    samples = _load_dataset(config)
    
    # Process samples with sharded extraction
    total_processed = 0
    for batch_start in range(0, len(samples), config.batch_size):
        batch_end = min(batch_start + config.batch_size, len(samples))
        batch = samples[batch_start:batch_end]
        
        # Tokenize
        texts = [s['text'] for s in batch]
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors='np'
        )
        
        # Pad to batch size
        input_ids = pad_sequences(inputs['input_ids'], config.batch_size)
        
        # Sharded forward pass
        activations = extract_activations_sharded(
            model, sharded_params, input_ids, mesh, layers
        )
        
        # Store activations (only for actual samples, not padding)
        for i, sample in enumerate(batch):
            for layer_idx, layer_acts in activations.items():
                storage.add_activation(
                    layer_idx=layer_idx,
                    activation=np.array(layer_acts[i]),
                    sample_idx=batch_start + i,
                    text_preview=texts[i]
                )
        
        total_processed += len(batch)
        if config.verbose and total_processed % 100 == 0:
            print(f"  Machine {config.machine_id}: Processed {total_processed}/{len(samples)}")
    
    # Finalize
    summary = storage.finalize()
    summary['total_processed'] = total_processed
    summary['machine_id'] = config.machine_id
    
    return summary


def _load_dataset(config: ExtractionConfig) -> List[Dict[str, Any]]:
    """Load dataset based on configuration."""
    samples = []
    
    if config.dataset_type == "arc":
        from ..data import load_arc_dataset_jsonl
        from ..arc import create_grid_encoder
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        encoder = create_grid_encoder("minimal")
        
        tasks = load_arc_dataset_jsonl(
            config.dataset_path,
            max_tasks=config.max_samples,
            machine_id=config.machine_id,
            total_machines=config.total_machines,
            verbose=config.verbose
        )
        
        # Convert tasks to samples
        from ..arc import create_prompts_from_task
        for task_id, task in tasks.items():
            prompts = create_prompts_from_task(
                task, encoder, tokenizer,
                is_train_prompt=True,
                prompt_version='output-from-examples-v0'
            )
            for prompt in prompts:
                samples.append({
                    'text': prompt,
                    'task_id': task_id
                })
    
    elif config.dataset_type == "jsonl":
        import json
        with open(config.dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if config.max_samples and i >= config.max_samples:
                    break
                # Shard across machines
                if i % config.total_machines != config.machine_id:
                    continue
                data = json.loads(line.strip())
                samples.append({
                    'text': data.get('text', data.get('content', '')),
                    'id': data.get('id', i)
                })
    
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")
    
    return samples
