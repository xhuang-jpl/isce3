#!/usr/bin/env python3
"""
Detailed correctness validation: Compare optimized implementation
against a naive (but correct) reference implementation.
"""
import sys
import os
import numpy as np
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.abspath('../'))

from rubbersheet import apply_filter


def reference_filter_naive(array, window_size, filter_type='mean'):
    """
    Reference implementation using explicit loops (slow but obviously correct).
    This serves as ground truth to validate the optimized version.
    """
    nrows, ncols = array.shape
    half_window = window_size // 2
    result = np.full_like(array, np.nan)

    for i in range(nrows):
        for j in range(ncols):
            # Extract window
            row_start = max(0, i - half_window)
            row_end = min(nrows, i + half_window + 1)
            col_start = max(0, j - half_window)
            col_end = min(ncols, j + half_window + 1)

            window = array[row_start:row_end, col_start:col_end]

            # Compute statistic
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'All-NaN slice encountered')
                warnings.filterwarnings('ignore', r'Mean of empty slice')
                if filter_type == 'mean':
                    result[i, j] = np.nanmean(window)
                elif filter_type == 'median':
                    result[i, j] = np.nanmedian(window)

    return result


def test_correctness_against_reference():
    """
    Compare optimized apply_filter against naive reference implementation.
    """
    print("="*70)
    print("CORRECTNESS VALIDATION: Optimized vs. Reference Implementation")
    print("="*70)

    # Test parameters
    np.random.seed(12345)
    test_cases = [
        (50, 30, 5, 0.0, "Small, no NaN"),
        (50, 30, 5, 0.1, "Small, 10% NaN"),
        (50, 30, 5, 0.3, "Small, 30% NaN"),
        (100, 80, 7, 0.1, "Medium, 10% NaN"),
        (100, 80, 11, 0.2, "Medium, 20% NaN"),
    ]

    all_passed = True

    for nrows, ncols, window_size, nan_fraction, description in test_cases:
        print(f"\nTest Case: {description}")
        print(f"  Array: {nrows}×{ncols}, Window: {window_size}×{window_size}, NaN: {nan_fraction*100:.0f}%")
        print("-" * 70)

        # Create test array
        array = np.random.randn(nrows, ncols).astype(np.float64)
        if nan_fraction > 0:
            array[np.random.rand(nrows, ncols) < nan_fraction] = np.nan

        # Test MEAN filter
        print("  Testing MEAN filter...")
        reference_mean = reference_filter_naive(array, window_size, 'mean')
        optimized_mean = apply_filter(array, window_size, filter_type='mean', axis='both')

        # Compare
        if not np.allclose(reference_mean, optimized_mean, equal_nan=True, rtol=1e-10, atol=1e-12):
            print("    ✗ FAILED: Results differ!")
            diff = np.abs(reference_mean - optimized_mean)
            valid_diff = diff[~np.isnan(diff)]
            if len(valid_diff) > 0:
                print(f"    Max difference: {np.max(valid_diff):.2e}")
                print(f"    Mean difference: {np.mean(valid_diff):.2e}")
            all_passed = False
        else:
            max_diff = np.nanmax(np.abs(reference_mean - optimized_mean))
            print(f"    ✓ PASSED: Max difference = {max_diff:.2e}")

        # Test MEDIAN filter
        print("  Testing MEDIAN filter...")
        reference_median = reference_filter_naive(array, window_size, 'median')
        optimized_median = apply_filter(array, window_size, filter_type='median', axis='both')

        # Compare
        if not np.allclose(reference_median, optimized_median, equal_nan=True, rtol=1e-10, atol=1e-12):
            print("    ✗ FAILED: Results differ!")
            diff = np.abs(reference_median - optimized_median)
            valid_diff = diff[~np.isnan(diff)]
            if len(valid_diff) > 0:
                print(f"    Max difference: {np.max(valid_diff):.2e}")
                print(f"    Mean difference: {np.mean(valid_diff):.2e}")
            all_passed = False
        else:
            max_diff = np.nanmax(np.abs(reference_median - optimized_median))
            print(f"    ✓ PASSED: Max difference = {max_diff:.2e}")

    print("\n" + "="*70)
    if all_passed:
        print("ALL CORRECTNESS TESTS PASSED ✓")
        print("The optimized implementation is numerically identical to reference.")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*70)

    return 0 if all_passed else 1


def test_edge_boundary_conditions():
    """
    Test that boundary conditions are handled correctly.
    """
    print("\n" + "="*70)
    print("BOUNDARY CONDITION TESTS")
    print("="*70)

    # Test 1: Single pixel
    print("\nTest 1: Single pixel (1×1)")
    array = np.array([[5.0]])
    result = apply_filter(array, 3, filter_type='mean', axis='both')
    assert result.shape == (1, 1), "Shape mismatch"
    assert result[0, 0] == 5.0, "Value mismatch"
    print("✓ Single pixel handled correctly")

    # Test 2: Single row
    print("\nTest 2: Single row (1×10)")
    array = np.arange(10, dtype=np.float64).reshape(1, 10)
    result = apply_filter(array, 3, filter_type='mean', axis='both')
    assert result.shape == (1, 10), "Shape mismatch"
    print(f"✓ Single row handled correctly")

    # Test 3: Single column
    print("\nTest 3: Single column (10×1)")
    array = np.arange(10, dtype=np.float64).reshape(10, 1)
    result = apply_filter(array, 3, filter_type='mean', axis='both')
    assert result.shape == (10, 1), "Shape mismatch"
    print(f"✓ Single column handled correctly")

    # Test 4: Window larger than array
    print("\nTest 4: Window (11×11) larger than array (5×5)")
    array = np.random.randn(5, 5).astype(np.float64)
    result = apply_filter(array, 11, filter_type='mean', axis='both')
    assert result.shape == (5, 5), "Shape mismatch"
    # Each pixel should see the entire array
    expected = np.full((5, 5), np.nanmean(array))
    assert np.allclose(result, expected), "Values incorrect"
    print(f"✓ Large window handled correctly")

    print("\n✓ All boundary condition tests passed!")


if __name__ == "__main__":
    print("\nNumPy version:", np.__version__)

    exit_code = test_correctness_against_reference()
    test_edge_boundary_conditions()

    sys.exit(exit_code)
