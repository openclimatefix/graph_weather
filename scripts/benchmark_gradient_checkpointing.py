#!/usr/bin/env python3
"""Benchmark gradient checkpointing memory savings.

This script measures GPU memory usage and training time with different
checkpointing strategies to quantify the memory-compute tradeoff.

Usage:
    python scripts/benchmark_gradient_checkpointing.py

The script will:
1. Test multiple grid resolutions (5.0°, 2.5°, 1.0° if GPU allows)
2. Test multiple batch sizes
3. Compare different checkpointing strategies
4. Generate a detailed report with memory savings and performance impact
"""

import gc
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from graph_weather.models import GraphCast, GraphCastConfig


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    strategy: str
    grid_resolution: float
    batch_size: int
    num_grid_points: int
    peak_memory_mb: float
    allocated_memory_mb: float
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    success: bool
    error: Optional[str] = None


class MemoryTracker:
    """Track GPU memory usage during model execution."""

    def __init__(self):
        self.peak_memory = 0
        self.allocated_memory = 0

    def reset(self):
        """Reset memory tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        gc.collect()
        self.peak_memory = 0
        self.allocated_memory = 0

    def update(self):
        """Update memory statistics."""
        if torch.cuda.is_available():
            self.peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            self.allocated_memory = torch.cuda.memory_allocated() / 1024**2  # MB

    def get_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        return {
            "peak_memory_mb": self.peak_memory,
            "allocated_memory_mb": self.allocated_memory,
        }


def create_grid(resolution_degrees: float) -> list:
    """Create lat/lon grid at specified resolution.

    Parameters
    ----------
    resolution_degrees : float
        Grid resolution in degrees

    Returns
    -------
    list
        List of (lat, lon) tuples
    """
    lat_lons = []
    lat_step = int(resolution_degrees)
    lon_step = int(resolution_degrees)

    for lat in range(-90, 90, lat_step):
        for lon in range(0, 360, lon_step):
            lat_lons.append((lat, lon))

    return lat_lons


def setup_model(
    lat_lons: list,
    strategy: str,
    device: torch.device,
    use_efficient_batching: bool = True,
) -> GraphCast:
    """Create and configure model with specified checkpointing strategy.

    Parameters
    ----------
    lat_lons : list
        Grid points
    strategy : str
        Checkpointing strategy name
    device : torch.device
        Device to use
    use_efficient_batching : bool
        Whether to use efficient batching

    Returns
    -------
    GraphCast
        Configured model
    """
    # Create model with fine-grained checkpointing disabled initially
    model = GraphCast(
        lat_lons=lat_lons,
        resolution=2,
        input_dim=78,
        output_dim=78,
        hidden_dim=256,
        num_processor_blocks=9,
        hidden_layers=2,
        mlp_norm_type="LayerNorm",
        use_checkpointing=False,  # Controlled hierarchically
        efficient_batching=use_efficient_batching,
    ).to(device)

    # Apply checkpointing strategy
    if strategy == "none":
        GraphCastConfig.no_checkpointing(model)
    elif strategy == "full":
        GraphCastConfig.full_checkpointing(model)
    elif strategy == "balanced":
        GraphCastConfig.balanced_checkpointing(model)
    elif strategy == "processor_only":
        GraphCastConfig.processor_only_checkpointing(model)
    elif strategy == "fine_grained":
        # Use layer-level checkpointing
        model = GraphCast(
            lat_lons=lat_lons,
            resolution=2,
            input_dim=78,
            output_dim=78,
            hidden_dim=256,
            num_processor_blocks=9,
            hidden_layers=2,
            mlp_norm_type="LayerNorm",
            use_checkpointing=True,  # Enable fine-grained
            efficient_batching=use_efficient_batching,
        ).to(device)
        GraphCastConfig.no_checkpointing(model)  # Disable hierarchical
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return model


def benchmark_strategy(
    strategy: str,
    grid_resolution: float,
    batch_size: int,
    device: torch.device,
    num_iterations: int = 3,
) -> BenchmarkResult:
    """Benchmark a single checkpointing strategy.

    Parameters
    ----------
    strategy : str
        Checkpointing strategy
    grid_resolution : float
        Grid resolution in degrees
    batch_size : int
        Batch size to test
    device : torch.device
        Device to use
    num_iterations : int
        Number of iterations to average over

    Returns
    -------
    BenchmarkResult
        Benchmark results
    """
    tracker = MemoryTracker()
    tracker.reset()

    try:
        # Create grid
        lat_lons = create_grid(grid_resolution)
        num_grid_points = len(lat_lons)

        # Create model
        model = setup_model(lat_lons, strategy, device, use_efficient_batching=True)

        # Create dummy data
        features = torch.randn(batch_size, num_grid_points, 78, device=device)
        target = torch.randn(batch_size, num_grid_points, 78, device=device)

        # Warm up
        model.train()
        output = model(features)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        model.zero_grad()

        tracker.reset()

        # Benchmark
        forward_times = []
        backward_times = []

        for _ in range(num_iterations):
            # Forward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()

            output = model(features)
            loss = nn.functional.mse_loss(output, target)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            forward_time = (time.perf_counter() - start_time) * 1000  # ms
            forward_times.append(forward_time)

            # Backward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()

            loss.backward()

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            backward_time = (time.perf_counter() - start_time) * 1000  # ms
            backward_times.append(backward_time)

            model.zero_grad()

            # Update memory stats
            tracker.update()

        # Calculate averages
        avg_forward_time = sum(forward_times) / len(forward_times)
        avg_backward_time = sum(backward_times) / len(backward_times)
        avg_total_time = avg_forward_time + avg_backward_time

        # Get memory stats
        mem_stats = tracker.get_stats()

        return BenchmarkResult(
            strategy=strategy,
            grid_resolution=grid_resolution,
            batch_size=batch_size,
            num_grid_points=num_grid_points,
            peak_memory_mb=mem_stats["peak_memory_mb"],
            allocated_memory_mb=mem_stats["allocated_memory_mb"],
            forward_time_ms=avg_forward_time,
            backward_time_ms=avg_backward_time,
            total_time_ms=avg_total_time,
            success=True,
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return BenchmarkResult(
                strategy=strategy,
                grid_resolution=grid_resolution,
                batch_size=batch_size,
                num_grid_points=len(create_grid(grid_resolution)),
                peak_memory_mb=0,
                allocated_memory_mb=0,
                forward_time_ms=0,
                backward_time_ms=0,
                total_time_ms=0,
                success=False,
                error="OOM",
            )
        else:
            raise

    finally:
        # Cleanup
        del model
        tracker.reset()


def print_results_table(results: List[BenchmarkResult], grid_resolution: float):
    """Print formatted results table.

    Parameters
    ----------
    results : List[BenchmarkResult]
        Benchmark results to display
    grid_resolution : float
        Grid resolution for this table
    """
    print(f"\n{'=' * 120}")
    print(f"Grid Resolution: {grid_resolution}° ({results[0].num_grid_points} points)")
    print(f"{'=' * 120}")
    print(
        f"{'Strategy':<20} {'Batch':<8} {'Peak Mem (MB)':<15} {'Total Time (ms)':<18} "
        f"{'Forward (ms)':<15} {'Backward (ms)':<15} {'Status':<10}"
    )
    print("-" * 120)

    for result in results:
        status = "✓ OK" if result.success else f"✗ {result.error}"
        if result.success:
            print(
                f"{result.strategy:<20} {result.batch_size:<8} "
                f"{result.peak_memory_mb:<15.1f} {result.total_time_ms:<18.1f} "
                f"{result.forward_time_ms:<15.1f} {result.backward_time_ms:<15.1f} "
                f"{status:<10}"
            )
        else:
            print(
                f"{result.strategy:<20} {result.batch_size:<8} "
                f"{'N/A':<15} {'N/A':<18} {'N/A':<15} {'N/A':<15} {status:<10}"
            )


def print_comparison_summary(all_results: List[BenchmarkResult]):
    """Print memory savings comparison.

    Parameters
    ----------
    all_results : List[BenchmarkResult]
        All benchmark results
    """
    print(f"\n{'=' * 100}")
    print("MEMORY SAVINGS COMPARISON")
    print(f"{'=' * 100}")

    # Group by grid resolution and batch size
    grouped = {}
    for result in all_results:
        key = (result.grid_resolution, result.batch_size)
        if key not in grouped:
            grouped[key] = {}
        if result.success:
            grouped[key][result.strategy] = result

    for (resolution, batch_size), strategies in sorted(grouped.items()):
        if "none" in strategies and len(strategies) > 1:
            baseline = strategies["none"]
            print(f"\nGrid: {resolution}°, Batch Size: {batch_size}")
            print(f"Baseline (no checkpointing): {baseline.peak_memory_mb:.1f} MB")
            print("-" * 80)

            for strategy_name, result in strategies.items():
                if strategy_name != "none":
                    savings = baseline.peak_memory_mb - result.peak_memory_mb
                    savings_pct = (savings / baseline.peak_memory_mb) * 100
                    time_overhead = result.total_time_ms - baseline.total_time_ms
                    time_overhead_pct = (time_overhead / baseline.total_time_ms) * 100

                    print(
                        f"  {strategy_name:<20}: {result.peak_memory_mb:>8.1f} MB  "
                        f"(↓ {savings:>6.1f} MB, {savings_pct:>5.1f}%)  "
                        f"Time: {result.total_time_ms:>7.1f} ms  "
                        f"(+{time_overhead:>6.1f} ms, {time_overhead_pct:>+5.1f}%)"
                    )


def main():
    """Run comprehensive gradient checkpointing benchmark."""
    print("=" * 100)
    print("GRADIENT CHECKPOINTING BENCHMARK")
    print("=" * 100)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"Total Memory: {total_memory:.2f} GB")
    else:
        print("WARNING: CUDA not available, benchmarking on CPU")

    # Test configurations
    grid_resolutions = [5.0, 2.5]  # Start with coarser grids
    batch_sizes = [1, 2, 4]
    strategies = [
        "none",  # Baseline
        "fine_grained",  # Layer-level checkpointing
        "processor_only",  # Checkpoint only processor
        "balanced",  # Checkpoint encoder + processor + decoder
        "full",  # Checkpoint entire model
    ]

    # Run benchmarks
    all_results = []

    for resolution in grid_resolutions:
        print(f"\n{'=' * 100}")
        print(f"Testing Grid Resolution: {resolution}°")
        print(f"{'=' * 100}")

        resolution_results = []

        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")

            for strategy in strategies:
                print(f"  Testing {strategy}...", end=" ", flush=True)

                result = benchmark_strategy(
                    strategy=strategy,
                    grid_resolution=resolution,
                    batch_size=batch_size,
                    device=device,
                    num_iterations=3,
                )

                resolution_results.append(result)
                all_results.append(result)

                if result.success:
                    print(
                        f"✓ Peak: {result.peak_memory_mb:.1f} MB, "
                        f"Time: {result.total_time_ms:.1f} ms"
                    )
                else:
                    print(f"✗ {result.error}")

        # Print results table for this resolution
        print_results_table(resolution_results, resolution)

    # Print comparison summary
    print_comparison_summary(all_results)

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_checkpointing_{timestamp}.json"

    results_dict = {
        "timestamp": timestamp,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "results": [
            {
                "strategy": r.strategy,
                "grid_resolution": r.grid_resolution,
                "batch_size": r.batch_size,
                "num_grid_points": r.num_grid_points,
                "peak_memory_mb": r.peak_memory_mb,
                "allocated_memory_mb": r.allocated_memory_mb,
                "forward_time_ms": r.forward_time_ms,
                "backward_time_ms": r.backward_time_ms,
                "total_time_ms": r.total_time_ms,
                "success": r.success,
                "error": r.error,
            }
            for r in all_results
        ],
    }

    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n{'=' * 100}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
