#!/usr/bin/env python3
"""
Test the specific case where reshape FAILS but optimized approach succeeds
"""
import numpy as np
import time

def test_failure_case():
    print("="*70)
    print("CRITICAL TEST: Array size where reshape() FAILS")
    print("="*70)

    # This size typically causes reshape to fail
    nrows, ncols = 5000, 2500
    window_size = 31

    print(f"\nArray: {nrows}×{ncols}")
    print(f"Window: {window_size}×{window_size}")

    array = np.random.randn(nrows, ncols).astype(np.float64)
    array[np.random.rand(nrows, ncols) < 0.1] = np.nan

    print(f"Array size: {array.nbytes / 1024**2:.2f} MB")

    # Prepare windows
    half = window_size // 2
    padded = np.pad(array, ((half, half), (half, half)),
                    mode='constant', constant_values=np.nan)

    shape = (nrows, ncols, window_size, window_size)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    print(f"Windows shape: {windows.shape}")
    print(f"Theoretical memory if materialized: {windows.nbytes / 1024**3:.2f} GB")

    # Test 1: Reshape (ORIGINAL - SHOULD FAIL)
    print("\n" + "-"*70)
    print("Method 1: WITH reshape (ORIGINAL APPROACH)")
    print("-"*70)
    try:
        print("Attempting: windows.reshape(nrows, ncols, -1)...")
        time_start = time.time()
        windows_flat = windows.reshape(nrows, ncols, -1)
        time_elapsed = time.time() - time_start

        print(f"✓ Reshape succeeded (unexpected!)")
        print(f"  Time: {time_elapsed:.3f} seconds")
        print(f"  Memory allocated: {windows_flat.nbytes / 1024**3:.2f} GB")

        # Try to use it
        result1 = np.nanmean(windows_flat, axis=2)
        print(f"✓ Filter completed")
        method1_success = True
        del windows_flat

    except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
        print(f"✗ FAILED: MemoryError")
        print(f"  Error message: {str(e)}")
        print(f"  This is EXPECTED - reshape cannot allocate {windows.nbytes / 1024**3:.2f} GB")
        method1_success = False
        result1 = None

    # Test 2: Direct axis (OPTIMIZED - SHOULD SUCCEED)
    print("\n" + "-"*70)
    print("Method 2: WITHOUT reshape (OPTIMIZED APPROACH)")
    print("-"*70)
    try:
        print("Attempting: np.nanmean(windows, axis=(2,3))...")
        time_start = time.time()
        result2 = np.nanmean(windows, axis=(2, 3))
        time_elapsed = time.time() - time_start

        print(f"✓ SUCCESS!")
        print(f"  Time: {time_elapsed:.3f} seconds")
        print(f"  Result shape: {result2.shape}")
        print(f"  Result size: {result2.nbytes / 1024**2:.2f} MB")
        method2_success = True

    except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
        print(f"✗ FAILED: {e}")
        method2_success = False
        result2 = None

    # Summary
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)

    if not method1_success and method2_success:
        print("✓ OPTIMIZATION SUCCESS!")
        print(f"  Original (reshape): FAILED - MemoryError")
        print(f"  Optimized (no reshape): SUCCESS")
        print(f"  Optimization enables processing of {nrows}×{ncols} frames")
        print(f"  that would otherwise fail!")

        if result1 is not None and result2 is not None:
            identical = np.allclose(result1, result2, equal_nan=True)
            print(f"  Results identical: {identical}")

    elif method1_success and method2_success:
        print("Both methods succeeded (may depend on available RAM)")
        if result1 is not None:
            identical = np.allclose(result1, result2, equal_nan=True)
            print(f"Results identical: {identical}")
            if identical:
                print(f"Max difference: {np.nanmax(np.abs(result1 - result2)):.2e}")
    else:
        print("Unexpected: both methods failed")

    print("="*70)

if __name__ == "__main__":
    test_failure_case()
