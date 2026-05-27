#!/usr/bin/env python3
"""
Test the optimized approach without reshape
"""
import numpy as np
import tracemalloc
import time

tracemalloc.start()

def test_optimized_approach():
    """Test direct axis computation without reshape"""

    # Create test array (realistic size)
    nrows, ncols = 5000, 2500
    window_size_az, window_size_rg = 31, 31

    print("="*60)
    print("Testing Optimized Approach: axis=(2,3) without reshape")
    print("="*60)

    array = np.random.randn(nrows, ncols).astype(np.float64)
    array[np.random.rand(nrows, ncols) < 0.1] = np.nan

    print(f"Array shape: {array.shape}")
    print(f"Array size: {array.nbytes / 1024**2:.2f} MB")
    print(f"Window size: {window_size_az}×{window_size_rg}\n")

    # Pad
    half_az = window_size_az // 2
    half_rg = window_size_rg // 2
    padded = np.pad(array, ((half_az, half_az), (half_rg, half_rg)),
                    mode='constant', constant_values=np.nan)

    print(f"Padded shape: {padded.shape}")
    print(f"Padded size: {padded.nbytes / 1024**2:.2f} MB\n")

    # Create stride view
    shape = (nrows, ncols, window_size_az, window_size_rg)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])

    snapshot_before = tracemalloc.take_snapshot()
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    snapshot_after_stride = tracemalloc.take_snapshot()
    stats = snapshot_after_stride.compare_to(snapshot_before, 'lineno')
    total_diff = sum(stat.size_diff for stat in stats)

    print(f"Windows shape: {windows.shape}")
    print(f"Windows theoretical size (if materialized): {windows.nbytes / 1024**3:.2f} GB")
    print(f"Actual memory used by stride view: {total_diff / 1024:.2f} KB (just metadata)\n")

    # Apply filter directly
    print("Applying nanmean with axis=(2,3)...")
    start_time = time.time()

    snapshot_before_filter = tracemalloc.take_snapshot()
    result = np.nanmean(windows, axis=(2, 3))
    snapshot_after_filter = tracemalloc.take_snapshot()

    elapsed = time.time() - start_time

    stats = snapshot_after_filter.compare_to(snapshot_before_filter, 'lineno')
    total_diff = sum(stat.size_diff for stat in stats)

    print(f"Filtering completed in {elapsed:.2f} seconds")
    print(f"Memory used by nanmean: {total_diff / 1024**2:.2f} MB")
    print(f"Result shape: {result.shape}")
    print(f"Result size: {result.nbytes / 1024**2:.2f} MB")

    # Get peak memory
    current, peak = tracemalloc.get_traced_memory()
    print(f"\nPeak memory usage: {peak / 1024**2:.2f} MB")
    print(f"Current memory usage: {current / 1024**2:.2f} MB")

    print("\n" + "="*60)
    print("SUCCESS: Optimized approach works without memory error!")
    print("="*60)

    return result

if __name__ == "__main__":
    result = test_optimized_approach()
