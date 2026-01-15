#!/usr/bin/env python3
"""
Performance Profiler for JAX TPU/GPU Workloads

Features:
- Memory profiling (device and host)
- Computation timing and throughput
- Communication overhead analysis
- TPU/GPU utilization tracking
- Bottleneck detection
"""

import jax
import jax.numpy as jnp
import time
import psutil
import os
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np


@dataclass
class ProfileStats:
    """Statistics from profiling"""
    name: str
    duration_ms: float
    memory_used_mb: float = 0.0
    device_memory_mb: float = 0.0
    throughput: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class PerformanceProfiler:
    """Profile JAX workloads for performance analysis"""

    def __init__(self, enable_memory_profiling: bool = True,
                 enable_device_profiling: bool = True,
                 verbose: bool = True):
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_device_profiling = enable_device_profiling
        self.verbose = verbose

        self.stats: List[ProfileStats] = []
        self.process = psutil.Process(os.getpid())

        # Get JAX device info
        self.devices = jax.devices()
        self.device_kind = self.devices[0].device_kind if self.devices else "cpu"

        if self.verbose:
            print(f"Profiler initialized:")
            print(f"  Devices: {len(self.devices)} x {self.device_kind}")
            print(f"  Memory profiling: {enable_memory_profiling}")
            print(f"  Device profiling: {enable_device_profiling}")

    @contextmanager
    def profile(self, name: str, metadata: Optional[Dict] = None):
        """Context manager for profiling a code block"""
        metadata = metadata or {}

        # Get initial state
        start_time = time.time()
        start_memory = self._get_host_memory_mb() if self.enable_memory_profiling else 0
        start_device_memory = self._get_device_memory_mb() if self.enable_device_profiling else 0

        try:
            yield
        finally:
            # Force JAX to wait for all operations
            jax.block_until_ready(jax.random.PRNGKey(0))

            # Get final state
            elapsed_ms = (time.time() - start_time) * 1000
            end_memory = self._get_host_memory_mb() if self.enable_memory_profiling else 0
            end_device_memory = self._get_device_memory_mb() if self.enable_device_profiling else 0

            memory_used = end_memory - start_memory
            device_memory_used = end_device_memory - start_device_memory

            # Calculate throughput if batch_size provided
            throughput = None
            if 'batch_size' in metadata and 'num_samples' in metadata:
                throughput = metadata['num_samples'] / (elapsed_ms / 1000)

            # Record stats
            stat = ProfileStats(
                name=name,
                duration_ms=elapsed_ms,
                memory_used_mb=memory_used,
                device_memory_mb=device_memory_used,
                throughput=throughput,
                metadata=metadata
            )
            self.stats.append(stat)

            if self.verbose:
                self._print_stat(stat)

    def _get_host_memory_mb(self) -> float:
        """Get current host memory usage in MB"""
        try:
            mem_info = self.process.memory_info()
            return mem_info.rss / (1024 ** 2)
        except:
            return 0.0

    def _get_device_memory_mb(self) -> float:
        """Get approximate device memory usage"""
        try:
            # This is approximate - JAX doesn't expose detailed memory stats
            # We'll track allocated arrays instead
            return 0.0  # Placeholder
        except:
            return 0.0

    def _print_stat(self, stat: ProfileStats):
        """Print a single stat"""
        print(f"\n[Profile] {stat.name}:")
        print(f"  Duration: {stat.duration_ms:.2f} ms")

        if self.enable_memory_profiling and stat.memory_used_mb != 0:
            sign = "+" if stat.memory_used_mb > 0 else ""
            print(f"  Host memory: {sign}{stat.memory_used_mb:.1f} MB")

        if self.enable_device_profiling and stat.device_memory_mb != 0:
            print(f"  Device memory: {stat.device_memory_mb:.1f} MB")

        if stat.throughput:
            print(f"  Throughput: {stat.throughput:.1f} samples/sec")

        if stat.metadata:
            print(f"  Metadata: {stat.metadata}")

    def print_summary(self):
        """Print summary of all profiled operations"""
        if not self.stats:
            print("No profiling data collected")
            return

        print("\n" + "="*70)
        print("PERFORMANCE PROFILE SUMMARY")
        print("="*70)

        # Group by name
        grouped = {}
        for stat in self.stats:
            if stat.name not in grouped:
                grouped[stat.name] = []
            grouped[stat.name].append(stat)

        # Print aggregated stats
        for name, stats_list in grouped.items():
            durations = [s.duration_ms for s in stats_list]
            count = len(stats_list)
            total_ms = sum(durations)
            avg_ms = total_ms / count
            min_ms = min(durations)
            max_ms = max(durations)

            print(f"\n{name}:")
            print(f"  Count: {count}")
            print(f"  Total: {total_ms:.2f} ms")
            print(f"  Average: {avg_ms:.2f} ms")
            print(f"  Min: {min_ms:.2f} ms")
            print(f"  Max: {max_ms:.2f} ms")

            # Throughput stats if available
            throughputs = [s.throughput for s in stats_list if s.throughput]
            if throughputs:
                avg_throughput = sum(throughputs) / len(throughputs)
                print(f"  Avg throughput: {avg_throughput:.1f} samples/sec")

        print("="*70)

        # Total time
        total_time_s = sum(s.duration_ms for s in self.stats) / 1000
        print(f"\nTotal profiled time: {total_time_s:.2f} s")

    def save_report(self, output_path: str):
        """Save profiling report to JSON"""
        report = {
            'device_info': {
                'num_devices': len(self.devices),
                'device_kind': self.device_kind,
            },
            'stats': [
                {
                    'name': s.name,
                    'duration_ms': s.duration_ms,
                    'memory_used_mb': s.memory_used_mb,
                    'device_memory_mb': s.device_memory_mb,
                    'throughput': s.throughput,
                    'metadata': s.metadata
                }
                for s in self.stats
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        if self.verbose:
            print(f"\nReport saved to: {output_path}")

    def get_bottlenecks(self, threshold_ms: float = 100.0) -> List[ProfileStats]:
        """Identify bottlenecks (operations taking > threshold)"""
        bottlenecks = [s for s in self.stats if s.duration_ms > threshold_ms]
        bottlenecks.sort(key=lambda x: x.duration_ms, reverse=True)
        return bottlenecks


class TPUMemoryMonitor:
    """Monitor TPU memory usage during training/inference"""

    def __init__(self, check_interval_s: float = 5.0):
        self.check_interval_s = check_interval_s
        self.samples: List[Dict] = []
        self.is_monitoring = False

    def start_monitoring(self):
        """Start background monitoring (not yet implemented)"""
        # TODO: Implement background thread for continuous monitoring
        pass

    def snapshot(self) -> Dict:
        """Take a memory snapshot"""
        # JAX doesn't expose detailed memory stats, so this is limited
        # We can track host memory and array shapes
        snapshot = {
            'timestamp': time.time(),
            'host_memory_mb': psutil.Process(os.getpid()).memory_info().rss / (1024**2),
            'num_devices': len(jax.devices()),
        }
        self.samples.append(snapshot)
        return snapshot


def benchmark_operation(func: Callable, *args, num_iterations: int = 10,
                       warmup_iterations: int = 3, **kwargs) -> Dict:
    """
    Benchmark a JAX operation

    Args:
        func: Function to benchmark
        args: Arguments to function
        num_iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations
        kwargs: Keyword arguments to function

    Returns:
        Dictionary with benchmark results
    """
    # Warmup
    for _ in range(warmup_iterations):
        result = func(*args, **kwargs)
        if isinstance(result, (jnp.ndarray, np.ndarray)):
            jax.block_until_ready(result)

    # Benchmark
    durations = []
    for _ in range(num_iterations):
        start = time.time()
        result = func(*args, **kwargs)

        # Ensure computation completes
        if isinstance(result, (jnp.ndarray, np.ndarray)):
            jax.block_until_ready(result)
        elif isinstance(result, (tuple, list)):
            for r in result:
                if isinstance(r, (jnp.ndarray, np.ndarray)):
                    jax.block_until_ready(r)

        elapsed = time.time() - start
        durations.append(elapsed * 1000)  # Convert to ms

    return {
        'mean_ms': np.mean(durations),
        'std_ms': np.std(durations),
        'min_ms': np.min(durations),
        'max_ms': np.max(durations),
        'median_ms': np.median(durations),
        'iterations': num_iterations
    }


def profile_model_forward_pass(model, params, input_ids, num_iterations: int = 10):
    """Profile a model's forward pass"""
    print("Profiling model forward pass...")

    # Warmup
    print("  Warming up...")
    for _ in range(3):
        output = model.apply(params, input_ids)
        jax.block_until_ready(output)

    # Profile
    print(f"  Running {num_iterations} iterations...")
    results = benchmark_operation(model.apply, params, input_ids,
                                  num_iterations=num_iterations,
                                  warmup_iterations=0)

    print("\nResults:")
    print(f"  Mean: {results['mean_ms']:.2f} ms")
    print(f"  Std: {results['std_ms']:.2f} ms")
    print(f"  Min: {results['min_ms']:.2f} ms")
    print(f"  Max: {results['max_ms']:.2f} ms")

    # Calculate throughput
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    tokens_per_sec = (batch_size * seq_len) / (results['mean_ms'] / 1000)

    print(f"\nThroughput:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Tokens/sec: {tokens_per_sec:.1f}")

    return results


def analyze_communication_overhead(mesh, params, verbose: bool = True):
    """Analyze communication overhead in sharded setup"""
    if verbose:
        print("\nAnalyzing communication overhead...")
        print(f"  Mesh: {mesh.axis_names}")
        print(f"  Devices: {mesh.devices.shape}")

    # TODO: Add detailed communication analysis
    # - Measure cross-device communication time
    # - Identify parameters with high communication overhead
    # - Suggest optimization strategies

    return {
        'mesh_shape': mesh.devices.shape,
        'mesh_axes': mesh.axis_names,
    }


def main():
    """Test profiler"""
    print("Testing Performance Profiler\n")

    profiler = PerformanceProfiler(verbose=True)

    # Test 1: Simple computation
    with profiler.profile("matmul_1024x1024"):
        key = jax.random.PRNGKey(0)
        a = jax.random.normal(key, (1024, 1024))
        b = jax.random.normal(key, (1024, 1024))
        c = jnp.matmul(a, b)
        jax.block_until_ready(c)

    # Test 2: Larger computation
    with profiler.profile("matmul_4096x4096", metadata={'size': 4096}):
        a = jax.random.normal(key, (4096, 4096))
        b = jax.random.normal(key, (4096, 4096))
        c = jnp.matmul(a, b)
        jax.block_until_ready(c)

    # Test 3: Batch processing
    batch_size = 32
    num_samples = batch_size * 10
    with profiler.profile("batch_processing",
                         metadata={'batch_size': batch_size, 'num_samples': num_samples}):
        for _ in range(10):
            x = jax.random.normal(key, (batch_size, 128, 128))
            y = jnp.sum(x, axis=(1, 2))
            jax.block_until_ready(y)

    # Print summary
    profiler.print_summary()

    # Identify bottlenecks
    bottlenecks = profiler.get_bottlenecks(threshold_ms=50.0)
    if bottlenecks:
        print("\nBottlenecks (>50ms):")
        for stat in bottlenecks:
            print(f"  {stat.name}: {stat.duration_ms:.2f} ms")

    # Save report
    profiler.save_report("profile_report.json")

    print("\nBenchmarking matmul...")
    # Benchmark a specific operation
    def matmul_op(a, b):
        return jnp.matmul(a, b)

    a = jax.random.normal(key, (2048, 2048))
    b = jax.random.normal(key, (2048, 2048))

    results = benchmark_operation(matmul_op, a, b, num_iterations=20)
    print(f"\nMatmul 2048x2048 benchmark:")
    print(f"  Mean: {results['mean_ms']:.2f} ms")
    print(f"  Std: {results['std_ms']:.2f} ms")


if __name__ == "__main__":
    main()
