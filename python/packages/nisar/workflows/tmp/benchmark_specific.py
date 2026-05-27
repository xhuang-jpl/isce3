#!/usr/bin/env python3
'''
Benchmark specific case: 1000x1000 array with 31x31 window
'''
import numpy as np
import time
import tracemalloc
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rubbersheet import apply_filter


def benchmark_specific_case():
    """Benchmark the specific case: 1000x1000 with 31x31 window."""
    print("=" * 80)
    print("BENCHMARK: 1000x1000 array with 31x31 window")
    print("=" * 80)
    print()

    # Create test array
    np.random.seed(42)
    test_array = np.random.randn(1000, 1000).astype(np.float64)
    test_array[::10, ::10] = np.nan  # Add ~1% NaN values

    array_size_mb = test_array.nbytes / 1024 / 1024

    print(f"Array shape: {test_array.shape}")
    print(f"Array dtype: {test_array.dtype}")
    print(f"Array size: {array_size_mb:.2f} MB")
    print(f"NaN count: {np.count_nonzero(np.isnan(test_array))} ({np.count_nonzero(np.isnan(test_array))/test_array.size*100:.2f}%)")
    print(f"Window size: 31x31")
    print()

    # Warm-up run
    print("Performing warm-up run...")
    _ = apply_filter(test_array, 31, filter_type='mean', axis='both')
    print("Warm-up complete")
    print()

    # Test both mean and median
    for filter_type in ['mean', 'median']:
        print(f"Testing {filter_type.upper()} filter:")
        print("-" * 80)

        # Runtime benchmark (multiple runs)
        num_runs = 5
        times = []

        for i in range(num_runs):
            start = time.time()
            result = apply_filter(test_array, 31, filter_type=filter_type, axis='both')
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.4f} seconds")

        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        print()
        print(f"  Mean time:   {mean_time:.4f} ± {std_time:.4f} seconds")
        print(f"  Min time:    {min_time:.4f} seconds")
        print(f"  Max time:    {max_time:.4f} seconds")
        print()

        # Memory benchmark
        print("  Memory benchmark:")
        tracemalloc.start()
        tracemalloc.reset_peak()

        _ = apply_filter(test_array, 31, filter_type=filter_type, axis='both')

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024
        overhead_mb = peak_mb - array_size_mb
        overhead_factor = peak_mb / array_size_mb

        print(f"    Current memory: {current_mb:.2f} MB")
        print(f"    Peak memory:    {peak_mb:.2f} MB")
        print(f"    Overhead:       {overhead_mb:.2f} MB ({overhead_factor:.2f}x base size)")
        print()

        # Check result
        nan_out = np.count_nonzero(np.isnan(result))
        print(f"  Result NaN count: {nan_out} ({nan_out/result.size*100:.2f}%)")
        print(f"  Result mean (ignoring NaN): {np.nanmean(result):.6f}")
        print(f"  Result std (ignoring NaN): {np.nanstd(result):.6f}")
        print()

    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_specific_case()
