#!/usr/bin/env python3
"""
Simple memory benchmark for sliding window filtering
"""
import numpy as np
import tracemalloc

# Start memory tracking
tracemalloc.start()

def test_reshape_vs_direct():
    """Compare reshape vs direct axis computation"""

    # Create test array
    nrows, ncols = 5000, 2500
    window_size_az, window_size_rg = 31, 31

    array = np.random.randn(nrows, ncols).astype(np.float64)
    array[np.random.rand(nrows, ncols) < 0.1] = np.nan

    print(f"Array shape: {array.shape}")
    print(f"Array size: {array.nbytes / 1024**2:.2f} MB\n")

    # Pad
    half_az = window_size_az // 2
    half_rg = window_size_rg // 2
    padded = np.pad(array, ((half_az, half_az), (half_rg, half_rg)),
                    mode='constant', constant_values=np.nan)

    # Create stride view
    shape = (nrows, ncols, window_size_az, window_size_rg)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    print(f"Windows shape: {windows.shape}")
    print(f"Windows theoretical size: {windows.nbytes / 1024**3:.2f} GB\n")

    # Method 1: With reshape
    print("="*60)
    print("Method 1: With reshape")
    print("="*60)
    snapshot_before = tracemalloc.take_snapshot()

    windows_flat = windows.reshape(nrows, ncols, -1)
    print(f"After reshape - shape: {windows_flat.shape}")
    print(f"Shares memory with original: {np.shares_memory(windows, windows_flat)}")

    snapshot_after_reshape = tracemalloc.take_snapshot()
    stats = snapshot_after_reshape.compare_to(snapshot_before, 'lineno')
    total_diff = sum(stat.size_diff for stat in stats)
    print(f"Memory allocated by reshape: {total_diff / 1024**2:.2f} MB")

    result1 = np.nanmean(windows_flat, axis=2)

    snapshot_after_mean = tracemalloc.take_snapshot()
    stats = snapshot_after_mean.compare_to(snapshot_after_reshape, 'lineno')
    total_diff = sum(stat.size_diff for stat in stats)
    print(f"Memory allocated by nanmean: {total_diff / 1024**2:.2f} MB")
    print(f"Result shape: {result1.shape}\n")

    del windows_flat

    # Method 2: Direct axis
    print("="*60)
    print("Method 2: Direct axis=(2,3)")
    print("="*60)
    snapshot_before = tracemalloc.take_snapshot()

    result2 = np.nanmean(windows, axis=(2, 3))

    snapshot_after = tracemalloc.take_snapshot()
    stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    total_diff = sum(stat.size_diff for stat in stats)
    print(f"Memory allocated by nanmean: {total_diff / 1024**2:.2f} MB")
    print(f"Result shape: {result2.shape}\n")

    # Verify results match
    print("="*60)
    print("Verification")
    print("="*60)
    print(f"Results are identical: {np.allclose(result1, result2, equal_nan=True)}")
    print(f"Max difference: {np.nanmax(np.abs(result1 - result2)):.2e}")

if __name__ == "__main__":
    test_reshape_vs_direct()
