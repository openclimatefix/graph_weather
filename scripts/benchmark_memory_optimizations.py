#!/usr/bin/env python3
"""Benchmark memory optimizations (efficient batching + gradient checkpointing).

This script benchmarks memory usage and performance for different optimization strategies:
- Efficient batching (avoids graph replication)
- Gradient checkpointing (trades compute for memory)
- Combined optimizations

Results are saved to JSON files.
"""

import gc
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from graph_weather.models import GraphCast, GraphCastConfig
from graph_weather.models.layers.decoder import Decoder
from graph_weather.models.layers.encoder import Encoder
from graph_weather.models.layers.processor import Processor


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    test_type: str  # "efficient_batching" or "gradient_checkpointing"
    grid_resolution: float
    batch_size: int
    num_grid_points: int
    efficient_batching: bool
    use_checkpointing: bool
    checkpoint_strategy: Optional[str]
    peak_memory_mb: float
    allocated_memory_mb: float
    forward_time_ms: Optional[float]
    backward_time_ms: Optional[float]
    total_time_ms: Optional[float]
    success: bool
    error: Optional[str] = None


def get_memory_stats() -> Dict[str, float]:
    """Get current memory usage statistics."""
    stats = {}
    if torch.cuda.is_available():
        stats["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        stats["allocated_memory_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
    else:
        stats["peak_memory_mb"] = 0
        stats["allocated_memory_mb"] = 0
    return stats


def reset_memory():
    """Reset memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def create_lat_lon_grid(resolution_deg: float) -> List[Tuple[float, float]]:
    """Create a lat/lon grid at specified resolution."""
    lat_lons = []
    lats = np.arange(-90, 90, resolution_deg)
    lons = np.arange(0, 360, resolution_deg)
    for lat in lats:
        for lon in lons:
            lat_lons.append((float(lat), float(lon)))
    return lat_lons


def benchmark_efficient_batching(
    resolution_deg: float,
    batch_size: int,
    device: str = "cuda",
    num_iterations: int = 3,
) -> BenchmarkResult:
    """Benchmark efficient batching optimization (inference mode)."""
    reset_memory()

    lat_lons = create_lat_lon_grid(resolution_deg)
    num_nodes = len(lat_lons)

    try:
        encoder = Encoder(
            lat_lons, resolution=2, input_dim=78, output_dim=256, efficient_batching=True
        ).to(device)
        processor = Processor(256, num_blocks=9).to(device)
        decoder = Decoder(
            lat_lons, resolution=2, input_dim=256, output_dim=78, efficient_batching=True
        ).to(device)

        encoder.eval()
        processor.eval()
        decoder.eval()

        features = torch.randn(batch_size, num_nodes, 78, device=device)

        # Warmup
        with torch.no_grad():
            x, edge_idx, edge_attr = encoder(features)
            x = processor(x, edge_idx, edge_attr, batch_size=batch_size, efficient_batching=True)
            _ = decoder(x, features, batch_size)

        reset_memory()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                x, edge_idx, edge_attr = encoder(features)
                x = processor(
                    x, edge_idx, edge_attr, batch_size=batch_size, efficient_batching=True
                )
                output = decoder(x, features, batch_size)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - start)

        mem_stats = get_memory_stats()

        del encoder, processor, decoder, features, x, edge_idx, edge_attr, output
        reset_memory()

        return BenchmarkResult(
            test_type="efficient_batching",
            grid_resolution=resolution_deg,
            batch_size=batch_size,
            num_grid_points=num_nodes,
            efficient_batching=True,
            use_checkpointing=False,
            checkpoint_strategy=None,
            forward_time_ms=None,
            backward_time_ms=None,
            total_time_ms=float(np.mean(times)) * 1000,
            success=True,
            **mem_stats,
        )

    except RuntimeError as e:
        reset_memory()
        return BenchmarkResult(
            test_type="efficient_batching",
            grid_resolution=resolution_deg,
            batch_size=batch_size,
            num_grid_points=num_nodes,
            efficient_batching=True,
            use_checkpointing=False,
            checkpoint_strategy=None,
            peak_memory_mb=0,
            allocated_memory_mb=0,
            forward_time_ms=None,
            backward_time_ms=None,
            total_time_ms=None,
            success=False,
            error=str(e),
        )


def benchmark_gradient_checkpointing(
    resolution_deg: float,
    batch_size: int,
    strategy: str,
    device: str = "cuda",
    num_iterations: int = 3,
) -> BenchmarkResult:
    """Benchmark gradient checkpointing optimization (training mode)."""
    reset_memory()

    lat_lons = create_lat_lon_grid(resolution_deg)
    num_nodes = len(lat_lons)

    try:
        # Create model
        use_fine_grained = strategy == "fine_grained"
        model = GraphCast(
            lat_lons=lat_lons,
            resolution=2,
            input_dim=78,
            output_dim=78,
            hidden_dim=256,
            num_processor_blocks=9,
            hidden_layers=2,
            mlp_norm_type="LayerNorm",
            use_checkpointing=use_fine_grained,
            efficient_batching=True,
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
            GraphCastConfig.no_checkpointing(model)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        model.train()

        features = torch.randn(batch_size, num_nodes, 78, device=device)
        target = torch.randn(batch_size, num_nodes, 78, device=device)

        # Warmup
        output = model(features)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        model.zero_grad()

        reset_memory()

        # Benchmark
        forward_times = []
        backward_times = []

        for _ in range(num_iterations):
            # Forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            output = model(features)
            loss = nn.functional.mse_loss(output, target)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_time = (time.perf_counter() - start_time) * 1000
            forward_times.append(forward_time)

            # Backward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            loss.backward()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_time = (time.perf_counter() - start_time) * 1000
            backward_times.append(backward_time)

            model.zero_grad()

        mem_stats = get_memory_stats()

        del model, features, target, output, loss
        reset_memory()

        return BenchmarkResult(
            test_type="gradient_checkpointing",
            grid_resolution=resolution_deg,
            batch_size=batch_size,
            num_grid_points=num_nodes,
            efficient_batching=True,
            use_checkpointing=(strategy != "none"),
            checkpoint_strategy=strategy,
            forward_time_ms=float(np.mean(forward_times)),
            backward_time_ms=float(np.mean(backward_times)),
            total_time_ms=float(np.mean(forward_times) + np.mean(backward_times)),
            success=True,
            **mem_stats,
        )

    except RuntimeError as e:
        reset_memory()
        return BenchmarkResult(
            test_type="gradient_checkpointing",
            grid_resolution=resolution_deg,
            batch_size=batch_size,
            num_grid_points=num_nodes,
            efficient_batching=True,
            use_checkpointing=(strategy != "none"),
            checkpoint_strategy=strategy,
            peak_memory_mb=0,
            allocated_memory_mb=0,
            forward_time_ms=None,
            backward_time_ms=None,
            total_time_ms=None,
            success=False,
            error=str(e),
        )


def run_efficient_batching_benchmarks() -> List[Dict]:
    """Run efficient batching benchmarks."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = [
        (5.0, [1, 2, 4, 8]),
        (2.5, [1, 2, 4]),
        (1.0, [1]),
    ]

    results = []
    for resolution_deg, batch_sizes in configs:
        for batch_size in batch_sizes:
            result = benchmark_efficient_batching(resolution_deg, batch_size, device)
            results.append(result)

            if not result.success and "out of memory" in result.error.lower():
                break

    return [vars(r) for r in results]


def run_gradient_checkpointing_benchmarks() -> List[Dict]:
    """Run gradient checkpointing benchmarks."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = [
        (5.0, [1, 2, 4, 8]),
        (2.5, [1, 2, 4]),
        (1.0, [1]),
    ]

    strategies = ["none", "fine_grained", "processor_only", "balanced", "full"]

    results = []
    for resolution_deg, batch_sizes in configs:
        skip_rest = False
        for batch_size in batch_sizes:
            if skip_rest:
                continue

            for strategy in strategies:
                result = benchmark_gradient_checkpointing(
                    resolution_deg, batch_size, strategy, device
                )
                results.append(result)

            # Check if all strategies failed
            batch_results = results[-len(strategies) :]
            if all(not r.success for r in batch_results):
                skip_rest = True

    return [vars(r) for r in results]


def main():
    """Run all benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark memory optimizations")
    parser.add_argument(
        "--mode",
        choices=["efficient_batching", "gradient_checkpointing", "all"],
        default="all",
        help="Which benchmarks to run",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode in ["efficient_batching", "all"]:
        results = run_efficient_batching_benchmarks()
        output_file = f"benchmark_efficient_batching_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    if args.mode in ["gradient_checkpointing", "all"]:
        results = run_gradient_checkpointing_benchmarks()
        output_file = f"benchmark_gradient_checkpointing_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
