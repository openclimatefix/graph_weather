"""
Benchmark script to measure memory usage and performance of graph encoding.

This script tests the current COO (Coordinate) format implementation and can be used
to compare against CSC/CSR implementations.

Run this BEFORE and AFTER implementing CSC/CSR changes to compare results.
"""

import gc
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from graph_weather.models.layers.decoder import Decoder
from graph_weather.models.layers.encoder import Encoder
from graph_weather.models.layers.processor import Processor


def get_memory_stats() -> Dict[str, float]:
    """Get current memory usage statistics."""
    stats = {}

    # CPU Memory
    import psutil

    process = psutil.Process()
    stats["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024

    # GPU Memory
    if torch.cuda.is_available():
        stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        stats["gpu_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        stats["gpu_memory_allocated_mb"] = 0
        stats["gpu_memory_reserved_mb"] = 0
        stats["gpu_max_memory_allocated_mb"] = 0

    return stats


def create_lat_lon_grid(resolution_deg: float) -> List[Tuple[float, float]]:
    """Create a lat/lon grid at specified resolution.

    Args:
        resolution_deg: Grid resolution in degrees (e.g., 1.0 for 1-degree grid)

    Returns:
        List of (lat, lon) tuples
    """
    lat_lons = []
    lats = np.arange(-90, 90, resolution_deg)
    lons = np.arange(0, 360, resolution_deg)

    for lat in lats:
        for lon in lons:
            lat_lons.append((float(lat), float(lon)))

    return lat_lons


def benchmark_full_model(
    resolution_deg: float, batch_size: int, device: str = "cpu", num_iterations: int = 3
) -> Dict:
    """Benchmark full model (encoder + processor + decoder) with detailed component breakdown.

    Args:
        resolution_deg: Grid resolution in degrees
        batch_size: Batch size to test
        device: 'cpu' or 'cuda'
        num_iterations: Number of forward passes to average

    Returns:
        Dictionary with comprehensive benchmark results
    """
    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Create grid
    lat_lons = create_lat_lon_grid(resolution_deg)
    num_nodes = len(lat_lons)

    # Get memory before model creation
    mem_before = get_memory_stats()

    try:
        # Create full model components
        encoder = Encoder(
            lat_lons, resolution=2, input_dim=78, output_dim=256, efficient_batching=True
        ).eval()
        processor = Processor(
            256,
            hidden_dim_processor_edge=256,
            hidden_dim_processor_node=256,
            hidden_layers_processor_edge=2,
            hidden_layers_processor_node=2,
            num_blocks=9,
        ).eval()
        decoder = Decoder(
            lat_lons, resolution=2, input_dim=256, output_dim=78, efficient_batching=True
        ).eval()

        if device == "cuda":
            encoder = encoder.cuda()
            processor = processor.cuda()
            decoder = decoder.cuda()

        # Get memory after model creation
        mem_after_model = get_memory_stats()

        # Create input features
        features = torch.randn((batch_size, num_nodes, 78))
        if device == "cuda":
            features = features.cuda()

        # Get memory after input creation
        mem_after_input = get_memory_stats()

        # Warmup
        with torch.no_grad():
            x, edge_idx_enc, edge_attr_enc = encoder(features)
            x_proc = processor(
                x, edge_idx_enc, edge_attr_enc, batch_size=batch_size, efficient_batching=True
            )
            _ = decoder(x_proc, features)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Detailed per-component benchmarking
        encoder_times = []
        processor_times = []
        decoder_times = []

        encoder_peak_mem = []
        processor_peak_mem = []
        decoder_peak_mem = []

        batched_edge_index_sizes = []
        batched_edge_attr_sizes = []

        with torch.no_grad():
            for _ in range(num_iterations):
                # ENCODER
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                start = time.time()
                x, edge_idx_enc, edge_attr_enc = encoder(features)
                if device == "cuda":
                    torch.cuda.synchronize()
                encoder_times.append(time.time() - start)

                if device == "cuda":
                    encoder_peak_mem.append(torch.cuda.max_memory_allocated() / 1024 / 1024)

                # Track batched edge_index size
                batched_edge_index_sizes.append(
                    (edge_idx_enc.element_size() * edge_idx_enc.nelement()) / 1024 / 1024
                )
                batched_edge_attr_sizes.append(
                    (edge_attr_enc.element_size() * edge_attr_enc.nelement()) / 1024 / 1024
                )

                # PROCESSOR
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                start = time.time()
                x_proc = processor(
                    x, edge_idx_enc, edge_attr_enc, batch_size=batch_size, efficient_batching=True
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                processor_times.append(time.time() - start)

                if device == "cuda":
                    processor_peak_mem.append(torch.cuda.max_memory_allocated() / 1024 / 1024)

                # DECODER
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                start = time.time()
                output = decoder(x_proc, features)
                if device == "cuda":
                    torch.cuda.synchronize()
                decoder_times.append(time.time() - start)

                if device == "cuda":
                    decoder_peak_mem.append(torch.cuda.max_memory_allocated() / 1024 / 1024)

        # Get memory after forward pass
        mem_after_forward = get_memory_stats()

        # Calculate statistics
        results = {
            "resolution_deg": resolution_deg,
            "batch_size": batch_size,
            "num_nodes": num_nodes,
            "num_edges_original": encoder.graph.edge_index.size(1),
            "num_edges_batched": edge_idx_enc.size(1),
            "num_h3_nodes": encoder.h3_nodes.size(0),
            "device": device,
            "success": True,
            # Overall Memory stats (MB)
            "model_cpu_memory_mb": mem_after_model["cpu_memory_mb"] - mem_before["cpu_memory_mb"],
            "input_cpu_memory_mb": mem_after_input["cpu_memory_mb"]
            - mem_after_model["cpu_memory_mb"],
            "forward_cpu_memory_mb": mem_after_forward["cpu_memory_mb"]
            - mem_after_input["cpu_memory_mb"],
            "total_cpu_memory_mb": mem_after_forward["cpu_memory_mb"] - mem_before["cpu_memory_mb"],
            "model_gpu_memory_mb": mem_after_model["gpu_memory_allocated_mb"]
            - mem_before["gpu_memory_allocated_mb"],
            "input_gpu_memory_mb": mem_after_input["gpu_memory_allocated_mb"]
            - mem_after_model["gpu_memory_allocated_mb"],
            "forward_gpu_memory_mb": mem_after_forward["gpu_memory_allocated_mb"]
            - mem_after_input["gpu_memory_allocated_mb"],
            "total_gpu_memory_mb": mem_after_forward["gpu_memory_allocated_mb"]
            - mem_before["gpu_memory_allocated_mb"],
            "peak_gpu_memory_mb": mem_after_forward["gpu_max_memory_allocated_mb"],
            # Per-component timing (seconds)
            "encoder_avg_time_s": np.mean(encoder_times),
            "encoder_std_time_s": np.std(encoder_times),
            "processor_avg_time_s": np.mean(processor_times),
            "processor_std_time_s": np.std(processor_times),
            "decoder_avg_time_s": np.mean(decoder_times),
            "decoder_std_time_s": np.std(decoder_times),
            "total_avg_time_s": np.mean(encoder_times)
            + np.mean(processor_times)
            + np.mean(decoder_times),
            # Per-component peak memory (MB)
            "encoder_peak_gpu_mb": np.mean(encoder_peak_mem) if encoder_peak_mem else 0,
            "processor_peak_gpu_mb": np.mean(processor_peak_mem) if processor_peak_mem else 0,
            "decoder_peak_gpu_mb": np.mean(decoder_peak_mem) if decoder_peak_mem else 0,
            # Graph structure info (original vs batched)
            "edge_index_memory_original_mb": (
                encoder.graph.edge_index.element_size() * encoder.graph.edge_index.nelement()
            )
            / 1024
            / 1024,
            "edge_attr_memory_original_mb": (
                encoder.graph.edge_attr.element_size() * encoder.graph.edge_attr.nelement()
            )
            / 1024
            / 1024,
            "edge_index_memory_batched_mb": np.mean(batched_edge_index_sizes),
            "edge_attr_memory_batched_mb": np.mean(batched_edge_attr_sizes),
            "batching_overhead_ratio": (
                np.mean(batched_edge_index_sizes)
                / (
                    (encoder.graph.edge_index.element_size() * encoder.graph.edge_index.nelement())
                    / 1024
                    / 1024
                )
                if encoder.graph.edge_index.nelement() > 0
                else 0
            ),
        }

        # Clean up
        del encoder, processor, decoder, features, x, edge_idx_enc, edge_attr_enc, x_proc, output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    except RuntimeError as e:
        # OOM or other error
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


def run_benchmark_suite():
    """Run comprehensive benchmark suite with detailed component breakdown."""

    print("=" * 80)
    print("DETAILED GRAPH BENCHMARK - FULL MODEL (ENCODER + PROCESSOR + DECODER)")
    print("=" * 80)
    print()

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    print()

    # Test configurations
    # Format: (resolution_deg, batch_sizes_to_test)
    test_configs = [
        (5.0, [1, 2, 4, 8]),  # 5-degree grid (~2.6K nodes)
        (2.5, [1, 2, 4]),  # 2.5-degree grid (~10K nodes)
        (1.0, [1, 2]),  # 1-degree grid (~65K nodes) - PROBLEMATIC
        # (0.5, [1]),                  # 0.5-degree grid (~260K nodes) - Skip for 4GB VRAM
    ]

    all_results = []

    for resolution_deg, batch_sizes in test_configs:
        print(f"\n{'=' * 80}")
        print(f"Testing {resolution_deg}-degree grid")
        print(f"{'=' * 80}\n")

        for batch_size in batch_sizes:
            print(f"Batch size: {batch_size}")

            result = benchmark_full_model(
                resolution_deg=resolution_deg,
                batch_size=batch_size,
                device=device,
                num_iterations=3,
            )

            all_results.append(result)

            if result["success"]:
                print("  ✓ Success")
                print(f"  Grid Nodes: {result['num_nodes']:,}")
                print(f"  H3 Nodes: {result['num_h3_nodes']:,}")
                print(f"  Edges (original): {result['num_edges_original']:,}")
                print(f"  Edges (batched): {result['num_edges_batched']:,}")
                print(f"  Batching overhead: {result['batching_overhead_ratio']:.2f}x")
                print()
                print("  MEMORY BREAKDOWN:")
                print(
                    f"    Edge index (original): {result['edge_index_memory_original_mb']:.2f} MB"
                )
                print(f"    Edge index (batched):  {result['edge_index_memory_batched_mb']:.2f} MB")
                print(f"    Edge attr (original):  {result['edge_attr_memory_original_mb']:.2f} MB")
                print(f"    Edge attr (batched):   {result['edge_attr_memory_batched_mb']:.2f} MB")
                if device == "cuda":
                    print(f"    Encoder peak GPU:      {result['encoder_peak_gpu_mb']:.2f} MB")
                    print(f"    Processor peak GPU:    {result['processor_peak_gpu_mb']:.2f} MB")
                    print(f"    Decoder peak GPU:      {result['decoder_peak_gpu_mb']:.2f} MB")
                    print(f"    Total GPU memory:      {result['total_gpu_memory_mb']:.2f} MB")
                    print(f"    Peak GPU memory:       {result['peak_gpu_memory_mb']:.2f} MB")
                print()
                print("  TIMING BREAKDOWN:")
                print(
                    f"    Encoder:    {result['encoder_avg_time_s']*1000:.2f} ± {result['encoder_std_time_s']*1000:.2f} ms"
                )
                print(
                    f"    Processor:  {result['processor_avg_time_s']*1000:.2f} ± {result['processor_std_time_s']*1000:.2f} ms"
                )
                print(
                    f"    Decoder:    {result['decoder_avg_time_s']*1000:.2f} ± {result['decoder_std_time_s']*1000:.2f} ms"
                )
                print(f"    Total:      {result['total_avg_time_s']*1000:.2f} ms")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
            print()

            # If we hit OOM, don't try larger batch sizes
            if not result["success"] and "out of memory" in result.get("error", "").lower():
                print("  Skipping larger batch sizes due to OOM\n")
                break

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}\n")

    # Create summary table
    print(
        f"{'Res':<6} {'Batch':<6} {'Nodes':<8} {'Edges':<8} {'Batch OH':<10} {'Enc(ms)':<10} {'Proc(ms)':<10} {'Dec(ms)':<10} {'Peak MB':<10} {'Status':<8}"
    )
    print(f"{'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for result in all_results:
        res = f"{result['resolution_deg']:.1f}°"
        batch = str(result["batch_size"])
        nodes = f"{result['num_nodes']:,}" if "num_nodes" in result else "N/A"
        edges = f"{result.get('num_edges_original', 0):,}" if result["success"] else "N/A"
        batch_oh = (
            f"{result.get('batching_overhead_ratio', 0):.2f}x" if result["success"] else "N/A"
        )
        enc_time = f"{result.get('encoder_avg_time_s', 0)*1000:.1f}" if result["success"] else "N/A"
        proc_time = (
            f"{result.get('processor_avg_time_s', 0)*1000:.1f}" if result["success"] else "N/A"
        )
        dec_time = f"{result.get('decoder_avg_time_s', 0)*1000:.1f}" if result["success"] else "N/A"
        peak_mem = (
            f"{result.get('peak_gpu_memory_mb', 0):.1f}"
            if result["success"] and device == "cuda"
            else "N/A"
        )
        status = "OK" if result["success"] else "FAIL"

        print(
            f"{res:<6} {batch:<6} {nodes:<8} {edges:<8} {batch_oh:<10} {enc_time:<10} {proc_time:<10} {dec_time:<10} {peak_mem:<10} {status:<8}"
        )

    print()

    # Key findings
    print("KEY FINDINGS:")
    print("-" * 80)

    # Find max successful batch size for 1-degree grid
    one_deg_results = [r for r in all_results if r["resolution_deg"] == 1.0 and r["success"]]
    if one_deg_results:
        max_batch = max(r["batch_size"] for r in one_deg_results)
        print(f"✓ 1-degree grid: Max batch size = {max_batch}")

        # Show batching overhead for 1-degree
        for r in one_deg_results:
            print(
                f"  Batch {r['batch_size']}: {r['batching_overhead_ratio']:.2f}x memory overhead from batching"
            )
    else:
        print("✗ 1-degree grid: No successful runs (even batch_size=1 failed!)")

    # Time breakdown
    successful = [r for r in all_results if r["success"]]
    if successful:
        print()
        print("AVERAGE TIME DISTRIBUTION:")
        avg_enc = np.mean([r["encoder_avg_time_s"] for r in successful]) * 1000
        avg_proc = np.mean([r["processor_avg_time_s"] for r in successful]) * 1000
        avg_dec = np.mean([r["decoder_avg_time_s"] for r in successful]) * 1000
        total = avg_enc + avg_proc + avg_dec

        print(f"  Encoder:   {avg_enc:.1f} ms ({avg_enc/total*100:.1f}%)")
        print(f"  Processor: {avg_proc:.1f} ms ({avg_proc/total*100:.1f}%)")
        print(f"  Decoder:   {avg_dec:.1f} ms ({avg_dec/total*100:.1f}%)")

    # Memory breakdown
    if successful and device == "cuda":
        print()
        print("AVERAGE PEAK MEMORY DISTRIBUTION:")
        avg_enc_mem = np.mean([r["encoder_peak_gpu_mb"] for r in successful])
        avg_proc_mem = np.mean([r["processor_peak_gpu_mb"] for r in successful])
        avg_dec_mem = np.mean([r["decoder_peak_gpu_mb"] for r in successful])

        print(f"  Encoder peak:   {avg_enc_mem:.1f} MB")
        print(f"  Processor peak: {avg_proc_mem:.1f} MB")
        print(f"  Decoder peak:   {avg_dec_mem:.1f} MB")

        # Batching overhead analysis
        print()
        print("BATCHING OVERHEAD ANALYSIS:")
        for r in successful:
            if r["edge_index_memory_original_mb"] > 0:
                overhead_mb = r["edge_index_memory_batched_mb"] - r["edge_index_memory_original_mb"]
                print(
                    f"  {r['resolution_deg']:.1f}° batch={r['batch_size']}: +{overhead_mb:.2f} MB ({r['batching_overhead_ratio']:.2f}x original)"
                )

    print()

    return all_results


if __name__ == "__main__":
    # Install psutil if not available
    try:
        import psutil
    except ImportError:
        print("Installing psutil for CPU memory monitoring...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])

    results = run_benchmark_suite()

    # Save results with timestamp
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_baseline_{timestamp}.json"

    with open(output_file, "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        results_serializable = []
        for r in results:
            r_copy = {}
            for k, v in r.items():
                if isinstance(v, (np.integer, np.floating)):
                    r_copy[k] = float(v)
                else:
                    r_copy[k] = v
            results_serializable.append(r_copy)
        json.dump(results_serializable, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to {output_file}")
    print(f"{'=' * 80}")
