"""Benchmark memory usage for efficient batching."""

import gc
import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch

from graph_weather.models.layers.decoder import Decoder
from graph_weather.models.layers.encoder import Encoder
from graph_weather.models.layers.processor import Processor


def get_memory_stats() -> Dict[str, float]:
    """Get current memory usage statistics."""
    try:
        import psutil

        process = psutil.Process()
        cpu_mb = process.memory_info().rss / 1024 / 1024
    except ImportError:
        cpu_mb = 0

    stats = {"cpu_memory_mb": cpu_mb}

    if torch.cuda.is_available():
        stats["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        stats["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        stats["gpu_memory_mb"] = 0
        stats["gpu_peak_mb"] = 0

    return stats


def create_lat_lon_grid(resolution_deg: float) -> List[Tuple[float, float]]:
    """Create a lat/lon grid at specified resolution."""
    lat_lons = []
    lats = np.arange(-90, 90, resolution_deg)
    lons = np.arange(0, 360, resolution_deg)
    for lat in lats:
        for lon in lons:
            lat_lons.append((float(lat), float(lon)))
    return lat_lons


def benchmark_config(
    resolution_deg: float,
    batch_size: int,
    device: str = "cuda",
    num_iterations: int = 3,
) -> Dict:
    """Benchmark a specific configuration."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    lat_lons = create_lat_lon_grid(resolution_deg)
    num_nodes = len(lat_lons)

    try:
        encoder = Encoder(
            lat_lons, resolution=2, input_dim=78, output_dim=256, efficient_batching=True
        ).to(device).eval()
        processor = Processor(256, num_blocks=9).to(device).eval()
        decoder = Decoder(
            lat_lons, resolution=2, input_dim=256, output_dim=78, efficient_batching=True
        ).to(device).eval()

        features = torch.randn(batch_size, num_nodes, 78, device=device)

        # Warmup
        with torch.no_grad():
            x, edge_idx, edge_attr = encoder(features)
            x = processor(x, edge_idx, edge_attr, batch_size=batch_size, efficient_batching=True)
            _ = decoder(x, features)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                x, edge_idx, edge_attr = encoder(features)
                x = processor(
                    x, edge_idx, edge_attr, batch_size=batch_size, efficient_batching=True
                )
                output = decoder(x, features)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - start)

        mem_stats = get_memory_stats()

        result = {
            "resolution_deg": resolution_deg,
            "batch_size": batch_size,
            "num_nodes": num_nodes,
            "device": device,
            "success": True,
            "avg_time_s": float(np.mean(times)),
            "std_time_s": float(np.std(times)),
            **mem_stats,
        }

        del encoder, processor, decoder, features, x, edge_idx, edge_attr, output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    except RuntimeError as e:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "resolution_deg": resolution_deg,
            "batch_size": batch_size,
            "num_nodes": num_nodes,
            "device": device,
            "success": False,
            "error": str(e),
        }


def run_benchmarks() -> List[Dict]:
    """Run benchmark suite."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = [
        (5.0, [1, 2, 4, 8]),
        (2.5, [1, 2, 4]),
        (1.0, [1]),
    ]

    results = []
    for resolution_deg, batch_sizes in configs:
        for batch_size in batch_sizes:
            result = benchmark_config(resolution_deg, batch_size, device)
            results.append(result)

            if not result["success"] and "out of memory" in result.get("error", "").lower():
                break

    return results


if __name__ == "__main__":
    results = run_benchmarks()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")
