#!/usr/bin/env python3
"""
Compare both approaches on a smaller array
"""
import numpy as np
import time

def test_both_approaches():
    """Compare reshape vs axis=(2,3) on smaller array"""

    # Smaller test
    nrows, ncols = 1000, 500
    window_size_az, window_size_rg = 11, 11

    print("="*60)
    print(f"Array: {nrows}×{ncols}, Window: {window_size_az}×{window_size_rg}")
    print("="*60)

    array = np.random.randn(nrows, ncols).astype(np.float64)
    array[np.random.rand(nrows, ncols) < 0.1] = np.nan

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

    print(f"Windows theoretical size: {windows.nbytes / 1024**2:.2f} MB\n")

    # Method 1: With reshape
    print("Method 1: reshape + nanmean(axis=2)")
    print("-" * 60)
    start = time.time()
    try:
        windows_flat = windows.reshape(nrows, ncols, -1)
        print(f"  Reshape: {'COPY' if not np.shares_memory(windows, windows_flat) else 'VIEW'}")
        result1 = np.nanmean(windows_flat, axis=2)
        elapsed1 = time.time() - start
        print(f"  Time: {elapsed1:.3f} seconds")
        print(f"  Result shape: {result1.shape}\n")
    except MemoryError as e:
        print(f"  FAILED: {e}\n")
        result1 = None
        elapsed1 = None

    # Method 2: Direct axis
    print("Method 2: nanmean(axis=(2,3))")
    print("-" * 60)
    start = time.time()
    result2 = np.nanmean(windows, axis=(2, 3))
    elapsed2 = time.time() - start
    print(f"  Time: {elapsed2:.3f} seconds")
    print(f"  Result shape: {result2.shape}\n")

    # Compare
    if result1 is not None:
        print("="*60)
        print(f"Results identical: {np.allclose(result1, result2, equal_nan=True)}")
        print(f"Speedup: {elapsed1/elapsed2:.2f}x")

if __name__ == "__main__":
    test_both_approaches()
