#!/usr/bin/env python3
"""
Benchmark memory usage for sliding window filtering with stride tricks
"""
import numpy as np
import psutil
import os
import warnings


def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_current_approach(array, window_size_az, window_size_rg):
    """Benchmark the current implementation with reshape"""
    print("\n=== Current Approach (with reshape) ===")

    nrows, ncols = array.shape
    half_window_az = window_size_az // 2
    half_window_rg = window_size_rg // 2

    # Pad the array
    padded = np.pad(array,
                    ((half_window_az, half_window_az), (half_window_rg, half_window_rg)),
                    mode='constant', constant_values=np.nan)

    mem_after_pad = get_memory_usage_mb()
    print(f"Memory after padding: {mem_after_pad:.2f} MB")

    # Create 4D sliding window view
    shape = (nrows, ncols, window_size_az, window_size_rg)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    mem_after_stride = get_memory_usage_mb()
    print(f"Memory after stride tricks: {mem_after_stride:.2f} MB (view only)")
    print(f"  - windows.shape: {windows.shape}")
    print(f"  - windows.nbytes (if materialized): {windows.nbytes / 1024**3:.2f} GB")

    # Reshape to flatten the window dimensions
    print("\nReshaping windows...")
    windows_flat = windows.reshape(nrows, ncols, -1)

    mem_after_reshape = get_memory_usage_mb()
    print(f"Memory after reshape: {mem_after_reshape:.2f} MB")
    print(f"  - Memory increase from reshape: {mem_after_reshape - mem_after_stride:.2f} MB")
    print(f"  - Did reshape create a copy? {not np.shares_memory(windows, windows_flat)}")

    # Apply filter
    print("\nApplying nanmean filter...")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN slice encountered')
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        filtered = np.nanmean(windows_flat, axis=2)

    mem_after_filter = get_memory_usage_mb()
    print(f"Memory after filtering: {mem_after_filter:.2f} MB")
    print(f"  - Filtered shape: {filtered.shape}")

    return filtered, mem_after_filter - mem_after_pad


def benchmark_optimized_approach(array, window_size_az, window_size_rg):
    """Benchmark the optimized implementation without reshape"""
    print("\n=== Optimized Approach (without reshape) ===")

    nrows, ncols = array.shape
    half_window_az = window_size_az // 2
    half_window_rg = window_size_rg // 2

    # Pad the array
    padded = np.pad(array,
                    ((half_window_az, half_window_az), (half_window_rg, half_window_rg)),
                    mode='constant', constant_values=np.nan)

    mem_after_pad = get_memory_usage_mb()
    print(f"Memory after padding: {mem_after_pad:.2f} MB")

    # Create 4D sliding window view
    shape = (nrows, ncols, window_size_az, window_size_rg)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    mem_after_stride = get_memory_usage_mb()
    print(f"Memory after stride tricks: {mem_after_stride:.2f} MB (view only)")
    print(f"  - windows.shape: {windows.shape}")

    # Apply filter directly on 4D array (no reshape needed)
    print("\nApplying nanmean filter directly on 4D array...")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN slice encountered')
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        filtered = np.nanmean(windows, axis=(2, 3))

    mem_after_filter = get_memory_usage_mb()
    print(f"Memory after filtering: {mem_after_filter:.2f} MB")
    print(f"  - Filtered shape: {filtered.shape}")

    return filtered, mem_after_filter - mem_after_pad


def main():
    print("=" * 70)
    print("Sliding Window Filter Memory Benchmark")
    print("=" * 70)

    # Test with realistic dimensions
    test_cases = [
        (1000, 500, 11, 11, "Small: 1000×500, window 11×11"),
        (5000, 2500, 31, 31, "Medium: 5000×2500, window 31×31"),
        (10000, 5000, 31, 31, "Large: 10000×5000, window 31×31"),
    ]

    for nrows, ncols, window_az, window_rg, description in test_cases:
        print(f"\n{'='*70}")
        print(f"Test Case: {description}")
        print(f"{'='*70}")

        # Create test array with some NaN values
        np.random.seed(42)
        array = np.random.randn(nrows, ncols).astype(np.float64)
        array[np.random.rand(nrows, ncols) < 0.1] = np.nan  # 10% NaN values

        mem_start = get_memory_usage_mb()
        print(f"Initial memory: {mem_start:.2f} MB")
        print(f"Array size: {array.nbytes / 1024**2:.2f} MB")

        # Benchmark current approach
        try:
            result1, mem_used1 = benchmark_current_approach(array, window_az, window_rg)
            print(f"\n>>> Total memory overhead: {mem_used1:.2f} MB")
        except MemoryError:
            print("\n>>> MEMORY ERROR: Not enough memory!")
            result1 = None
            mem_used1 = float('inf')

        # Benchmark optimized approach
        try:
            result2, mem_used2 = benchmark_optimized_approach(array, window_az, window_rg)
            print(f"\n>>> Total memory overhead: {mem_used2:.2f} MB")
        except MemoryError:
            print("\n>>> MEMORY ERROR: Not enough memory!")
            result2 = None
            mem_used2 = float('inf')

        # Compare results
        if result1 is not None and result2 is not None:
            print(f"\n{'='*70}")
            print("COMPARISON")
            print(f"{'='*70}")
            print(f"Memory savings: {mem_used1 - mem_used2:.2f} MB")
            print(f"Memory reduction: {(1 - mem_used2/mem_used1)*100:.1f}%")

            # Verify results are identical
            max_diff = np.nanmax(np.abs(result1 - result2))
            print(f"Max difference between results: {max_diff:.2e}")
            print(f"Results are identical: {np.allclose(result1, result2, equal_nan=True)}")

        print()


if __name__ == "__main__":
    main()
