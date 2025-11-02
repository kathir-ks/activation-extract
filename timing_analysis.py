#!/usr/bin/env python3
"""
Timing analysis for activation extraction pipeline.

Measures performance metrics:
- Time per task
- Time per sample
- Time per batch
- Throughput (samples/second)
- Memory usage
- Bottlenecks (tokenization, model inference, saving)
"""

import time
import json
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
import jax
from transformers import AutoTokenizer

# Import from extraction script
from extract_activations_arc import (
    load_arc_dataset,
    create_prompts_from_dataset,
    pad_sequences,
    extract_activations_batch_parallel,
    extract_activations_batch_single_device,
    ActivationStorage,
    ActivationExtractionConfig
)
from qwen2_jax import QwenConfig, convert_hf_to_jax_weights
from qwen2_jax_with_hooks import create_model_with_hooks
from transformers import AutoModelForCausalLM


@dataclass
class TimingMetrics:
    """Timing metrics for each phase"""
    phase_name: str
    duration_seconds: float
    throughput: float = 0.0  # items/second
    memory_mb: float = 0.0


@dataclass
class ExtractionTimingReport:
    """Complete timing report"""
    # Dataset info
    num_tasks: int
    num_prompts: int
    num_samples: int
    num_layers: int

    # Timing breakdown
    dataset_loading_time: float
    prompt_creation_time: float
    tokenization_time: float
    model_loading_time: float
    inference_time: float
    saving_time: float
    total_time: float

    # Throughput metrics
    samples_per_second: float
    prompts_per_second: float
    tasks_per_second: float

    # Per-item timings
    time_per_task: float
    time_per_prompt: float
    time_per_sample: float
    time_per_batch: float

    # Memory metrics
    peak_memory_mb: float
    avg_memory_mb: float

    # TPU info
    num_devices: int
    device_type: str
    use_data_parallel: bool

    # Bottleneck analysis
    slowest_phase: str
    bottleneck_percentage: float


class TimingAnalyzer:
    """Analyzes timing of activation extraction pipeline"""

    def __init__(self, config: ActivationExtractionConfig):
        self.config = config
        self.metrics: List[TimingMetrics] = []
        self.memory_samples: List[float] = []

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def record_memory(self):
        """Record current memory usage"""
        self.memory_samples.append(self.get_memory_usage())

    def time_phase(self, phase_name: str):
        """Context manager for timing a phase"""
        return PhaseTimer(self, phase_name)

    def add_metric(self, metric: TimingMetrics):
        """Add a timing metric"""
        self.metrics.append(metric)
        print(f"  ⏱ {metric.phase_name}: {metric.duration_seconds:.2f}s", end="")
        if metric.throughput > 0:
            print(f" ({metric.throughput:.2f} items/s)", end="")
        if metric.memory_mb > 0:
            print(f" [Mem: {metric.memory_mb:.0f}MB]", end="")
        print()

    def generate_report(self, num_tasks: int, num_prompts: int, num_samples: int,
                       num_layers: int, num_batches: int) -> ExtractionTimingReport:
        """Generate comprehensive timing report"""

        # Extract timings by phase
        timings = {m.phase_name: m.duration_seconds for m in self.metrics}

        dataset_loading = timings.get('dataset_loading', 0)
        prompt_creation = timings.get('prompt_creation', 0)
        tokenization = timings.get('tokenization', 0)
        model_loading = timings.get('model_loading', 0)
        inference = timings.get('inference', 0)
        saving = timings.get('saving', 0)

        total_time = sum(timings.values())

        # Calculate throughput
        samples_per_second = num_samples / total_time if total_time > 0 else 0
        prompts_per_second = num_prompts / total_time if total_time > 0 else 0
        tasks_per_second = num_tasks / total_time if total_time > 0 else 0

        # Per-item timings
        time_per_task = total_time / num_tasks if num_tasks > 0 else 0
        time_per_prompt = total_time / num_prompts if num_prompts > 0 else 0
        time_per_sample = total_time / num_samples if num_samples > 0 else 0
        time_per_batch = inference / num_batches if num_batches > 0 else 0

        # Memory metrics
        peak_memory = max(self.memory_samples) if self.memory_samples else 0
        avg_memory = np.mean(self.memory_samples) if self.memory_samples else 0

        # TPU info
        devices = jax.devices()
        num_devices = len(devices)
        device_type = devices[0].device_kind if devices else "Unknown"

        # Bottleneck analysis
        phase_times = [
            ('dataset_loading', dataset_loading),
            ('prompt_creation', prompt_creation),
            ('tokenization', tokenization),
            ('model_loading', model_loading),
            ('inference', inference),
            ('saving', saving)
        ]
        slowest_phase, slowest_time = max(phase_times, key=lambda x: x[1])
        bottleneck_percentage = (slowest_time / total_time * 100) if total_time > 0 else 0

        return ExtractionTimingReport(
            num_tasks=num_tasks,
            num_prompts=num_prompts,
            num_samples=num_samples,
            num_layers=num_layers,
            dataset_loading_time=dataset_loading,
            prompt_creation_time=prompt_creation,
            tokenization_time=tokenization,
            model_loading_time=model_loading,
            inference_time=inference,
            saving_time=saving,
            total_time=total_time,
            samples_per_second=samples_per_second,
            prompts_per_second=prompts_per_second,
            tasks_per_second=tasks_per_second,
            time_per_task=time_per_task,
            time_per_prompt=time_per_prompt,
            time_per_sample=time_per_sample,
            time_per_batch=time_per_batch,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            num_devices=num_devices,
            device_type=device_type,
            use_data_parallel=self.config.use_data_parallel,
            slowest_phase=slowest_phase,
            bottleneck_percentage=bottleneck_percentage
        )


class PhaseTimer:
    """Context manager for timing phases"""

    def __init__(self, analyzer: TimingAnalyzer, phase_name: str):
        self.analyzer = analyzer
        self.phase_name = phase_name
        self.start_time = None
        self.start_memory = None

    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = self.analyzer.get_memory_usage()
        self.analyzer.record_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        end_memory = self.analyzer.get_memory_usage()
        memory_delta = end_memory - self.start_memory

        metric = TimingMetrics(
            phase_name=self.phase_name,
            duration_seconds=duration,
            memory_mb=end_memory
        )
        self.analyzer.add_metric(metric)
        self.analyzer.record_memory()


def run_timing_analysis(config: ActivationExtractionConfig, output_file: str = "timing_report.json"):
    """Run complete timing analysis of extraction pipeline"""

    print("=" * 70)
    print("ACTIVATION EXTRACTION TIMING ANALYSIS")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Dataset: {config.dataset_path}")
    print(f"  Tasks: {config.n_tasks}")
    print(f"  Predictions per task: {config.predictions_per_task}")
    print(f"  Layers: {config.layers_to_extract}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Data parallel: {config.use_data_parallel}")
    print()

    analyzer = TimingAnalyzer(config)

    # Phase 1: Load dataset
    print("Phase 1: Loading dataset...")
    with analyzer.time_phase('dataset_loading'):
        data = load_arc_dataset(
            config.dataset_path,
            n_tasks=config.n_tasks,
            random_seed=config.random_seed
        )

    # Phase 2: Create prompts
    print("Phase 2: Creating prompts...")
    with analyzer.time_phase('prompt_creation'):
        prompts_data = create_prompts_from_dataset(
            data,
            grid_encoder=config.grid_encoder,
            predictions_per_task=config.predictions_per_task,
            prompt_version=config.prompt_version,
            verbose=config.verbose
        )
    num_prompts = len(prompts_data)

    # Phase 3: Tokenization
    print("Phase 3: Tokenizing prompts...")
    with analyzer.time_phase('tokenization'):
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)

        tokenized_prompts = []
        for prompt_data in prompts_data:
            inputs = tokenizer(
                prompt_data['prompt'],
                return_tensors="np",
                truncation=True,
                max_length=config.max_seq_length
            )
            tokenized_prompts.append(inputs['input_ids'][0])

    # Phase 4: Load model
    print("Phase 4: Loading model...")
    with analyzer.time_phase('model_loading'):
        print("  Loading HF model...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            dtype=torch.float32
        )

        print("  Creating JAX model with hooks...")
        jax_model = create_model_with_hooks(
            hf_model.config,
            layers_to_extract=config.layers_to_extract
        )

        print("  Converting weights...")
        converted_params = convert_hf_to_jax_weights(hf_model, hf_model.config)
        params = {'params': converted_params}
        del hf_model

    # Phase 5: Inference
    print("Phase 5: Running inference...")

    # Create batches
    batch_size = config.batch_size
    num_batches = (len(tokenized_prompts) + batch_size - 1) // batch_size

    with analyzer.time_phase('inference'):
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(tokenized_prompts))
            batch_prompts = tokenized_prompts[start_idx:end_idx]

            # Pad sequences
            input_ids = pad_sequences(batch_prompts, pad_token_id=tokenizer.pad_token_id)

            # Run inference
            if config.use_data_parallel:
                activations = extract_activations_batch_parallel(jax_model, params, input_ids)
            else:
                activations = extract_activations_batch_single_device(jax_model, params, input_ids)

    # Phase 6: Saving (simulate)
    print("Phase 6: Saving activations...")
    with analyzer.time_phase('saving'):
        # Simulate saving by actually saving to temp directory
        temp_storage = ActivationStorage(
            output_dir="temp_timing_test",
            upload_to_gcs=False,
            shard_size_gb=config.shard_size_gb,
            compress_shards=config.compress_shards,
            verbose=False
        )

        # Add all activations
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(tokenized_prompts))
            batch_prompts = tokenized_prompts[start_idx:end_idx]

            input_ids = pad_sequences(batch_prompts, pad_token_id=tokenizer.pad_token_id)

            if config.use_data_parallel:
                activations = extract_activations_batch_parallel(jax_model, params, input_ids)
            else:
                activations = extract_activations_batch_single_device(jax_model, params, input_ids)

            for sample_idx in range(input_ids.shape[0]):
                global_sample_idx = batch_idx * batch_size + sample_idx
                if global_sample_idx >= len(prompts_data):
                    break

                prompt_data = prompts_data[global_sample_idx]

                for layer_idx in config.layers_to_extract:
                    layer_key = f'layer_{layer_idx}'
                    if layer_key in activations:
                        layer_act = activations[layer_key][sample_idx]
                        layer_act_np = np.array(layer_act)

                        temp_storage.add_activation(
                            layer_idx=layer_idx,
                            activation=layer_act_np,
                            task_id=prompt_data['task_id'],
                            sample_idx=global_sample_idx,
                            prompt=prompt_data['prompt']
                        )

        temp_storage.finalize()

    # Generate report
    print()
    print("Generating timing report...")
    num_samples = len(prompts_data) * len(config.layers_to_extract)

    report = analyzer.generate_report(
        num_tasks=len(data),
        num_prompts=num_prompts,
        num_samples=num_samples,
        num_layers=len(config.layers_to_extract),
        num_batches=num_batches
    )

    # Print report
    print()
    print("=" * 70)
    print("TIMING ANALYSIS REPORT")
    print("=" * 70)
    print()

    print("DATASET METRICS:")
    print(f"  Tasks: {report.num_tasks}")
    print(f"  Prompts: {report.num_prompts}")
    print(f"  Samples (prompts × layers): {report.num_samples}")
    print(f"  Layers: {report.num_layers}")
    print()

    print("TIMING BREAKDOWN:")
    print(f"  Dataset loading:  {report.dataset_loading_time:7.2f}s  ({report.dataset_loading_time/report.total_time*100:5.1f}%)")
    print(f"  Prompt creation:  {report.prompt_creation_time:7.2f}s  ({report.prompt_creation_time/report.total_time*100:5.1f}%)")
    print(f"  Tokenization:     {report.tokenization_time:7.2f}s  ({report.tokenization_time/report.total_time*100:5.1f}%)")
    print(f"  Model loading:    {report.model_loading_time:7.2f}s  ({report.model_loading_time/report.total_time*100:5.1f}%)")
    print(f"  Inference:        {report.inference_time:7.2f}s  ({report.inference_time/report.total_time*100:5.1f}%)")
    print(f"  Saving:           {report.saving_time:7.2f}s  ({report.saving_time/report.total_time*100:5.1f}%)")
    print(f"  {'─' * 60}")
    print(f"  TOTAL:            {report.total_time:7.2f}s")
    print()

    print("THROUGHPUT:")
    print(f"  Tasks/second:     {report.tasks_per_second:.3f}")
    print(f"  Prompts/second:   {report.prompts_per_second:.3f}")
    print(f"  Samples/second:   {report.samples_per_second:.3f}")
    print()

    print("PER-ITEM TIMINGS:")
    print(f"  Time per task:    {report.time_per_task:.2f}s")
    print(f"  Time per prompt:  {report.time_per_prompt:.2f}s")
    print(f"  Time per sample:  {report.time_per_sample:.3f}s")
    print(f"  Time per batch:   {report.time_per_batch:.2f}s")
    print()

    print("MEMORY USAGE:")
    print(f"  Peak memory:      {report.peak_memory_mb:.0f} MB")
    print(f"  Average memory:   {report.avg_memory_mb:.0f} MB")
    print()

    print("TPU CONFIGURATION:")
    print(f"  Devices:          {report.num_devices}")
    print(f"  Device type:      {report.device_type}")
    print(f"  Data parallel:    {report.use_data_parallel}")
    print()

    print("BOTTLENECK ANALYSIS:")
    print(f"  Slowest phase:    {report.slowest_phase}")
    print(f"  Bottleneck %:     {report.bottleneck_percentage:.1f}%")
    print()

    # Save report to JSON
    report_dict = asdict(report)
    with open(output_file, 'w') as f:
        json.dump(report_dict, f, indent=2)

    print(f"Report saved to: {output_file}")
    print()

    # Cleanup
    import shutil
    if Path("temp_timing_test").exists():
        shutil.rmtree("temp_timing_test")

    return report


def main():
    """Main timing analysis"""
    import argparse
    import torch

    parser = argparse.ArgumentParser(description='Timing analysis for activation extraction')
    parser.add_argument('--dataset_path', type=str, default='test_data_small.json',
                       help='Path to dataset')
    parser.add_argument('--n_tasks', type=int, default=5,
                       help='Number of tasks to process for timing')
    parser.add_argument('--predictions_per_task', type=int, default=10,
                       help='Predictions per task')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--output_file', type=str, default='timing_report.json',
                       help='Output file for timing report')
    parser.add_argument('--no_data_parallel', action='store_true',
                       help='Disable data parallelism')

    args = parser.parse_args()

    # Create config
    config = ActivationExtractionConfig(
        dataset_path=args.dataset_path,
        n_tasks=args.n_tasks,
        predictions_per_task=args.predictions_per_task,
        batch_size=args.batch_size,
        use_data_parallel=not args.no_data_parallel,
        shard_size_gb=0.001,
        compress_shards=True,
        verbose=True
    )

    # Run timing analysis
    report = run_timing_analysis(config, args.output_file)

    # Print scaling projections
    print("=" * 70)
    print("SCALING PROJECTIONS")
    print("=" * 70)
    print()

    # Project for different dataset sizes
    for num_tasks in [100, 500, 1000, 5000, 10000]:
        scaling_factor = num_tasks / report.num_tasks
        projected_time = report.total_time * scaling_factor

        hours = projected_time / 3600
        days = hours / 24

        print(f"{num_tasks:5d} tasks: {projected_time:8.0f}s ({hours:6.1f}h / {days:5.2f}d)")

    print()


if __name__ == "__main__":
    main()
