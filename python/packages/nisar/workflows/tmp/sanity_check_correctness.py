#!/usr/bin/env python3
"""
Sanity check: Verify apply_filter produces correct numerical results
by comparing against reference implementations
"""
import sys
import os
import numpy as np
import warnings

sys.path.insert(0, os.path.abspath('../'))
from rubbersheet import apply_filter


def reference_mean_filter(array, window_size, axis='both'):
    """
    Reference implementation using explicit loops (slow but obviously correct)
    """
    if axis not in ['azimuth', 'range', 'both']:
        raise ValueError(f"Invalid axis: {axis}")

    nrows, ncols = array.shape
    result = np.full_like(array, np.nan)

    if axis == 'both':
        half_win = window_size // 2
        pad_before = (window_size - 1) // 2
        pad_after = window_size // 2

        for i in range(nrows):
            for j in range(ncols):
                # Extract window with proper padding
                row_start = max(0, i - pad_before)
                row_end = min(nrows, i + pad_after + 1)
                col_start = max(0, j - pad_before)
                col_end = min(ncols, j + pad_after + 1)

                window = array[row_start:row_end, col_start:col_end]
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    result[i, j] = np.nanmean(window)

    elif axis == 'azimuth':
        pad_before = (window_size - 1) // 2
        pad_after = window_size // 2

        for i in range(nrows):
            for j in range(ncols):
                row_start = max(0, i - pad_before)
                row_end = min(nrows, i + pad_after + 1)
                window = array[row_start:row_end, j]
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    result[i, j] = np.nanmean(window)

    else:  # range
        pad_before = (window_size - 1) // 2
        pad_after = window_size // 2

        for i in range(nrows):
            for j in range(ncols):
                col_start = max(0, j - pad_before)
                col_end = min(ncols, j + pad_after + 1)
                window = array[i, col_start:col_end]
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    result[i, j] = np.nanmean(window)

    return result


def reference_median_filter(array, window_size, axis='both'):
    """Reference median filter implementation"""
    if axis not in ['azimuth', 'range', 'both']:
        raise ValueError(f"Invalid axis: {axis}")

    nrows, ncols = array.shape
    result = np.full_like(array, np.nan)

    if axis == 'both':
        pad_before = (window_size - 1) // 2
        pad_after = window_size // 2

        for i in range(nrows):
            for j in range(ncols):
                row_start = max(0, i - pad_before)
                row_end = min(nrows, i + pad_after + 1)
                col_start = max(0, j - pad_before)
                col_end = min(ncols, j + pad_after + 1)

                window = array[row_start:row_end, col_start:col_end]
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    result[i, j] = np.nanmedian(window)

    elif axis == 'azimuth':
        pad_before = (window_size - 1) // 2
        pad_after = window_size // 2

        for i in range(nrows):
            for j in range(ncols):
                row_start = max(0, i - pad_before)
                row_end = min(nrows, i + pad_after + 1)
                window = array[row_start:row_end, j]
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    result[i, j] = np.nanmedian(window)

    else:  # range
        pad_before = (window_size - 1) // 2
        pad_after = window_size // 2

        for i in range(nrows):
            for j in range(ncols):
                col_start = max(0, j - pad_before)
                col_end = min(ncols, j + pad_after + 1)
                window = array[i, col_start:col_end]
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    result[i, j] = np.nanmedian(window)

    return result


def compare_results(result_test, result_ref, test_name, tolerance=1e-10):
    """Compare two result arrays"""
    # Check shapes match
    if result_test.shape != result_ref.shape:
        print(f"  ✗ {test_name}: Shape mismatch!")
        print(f"    Test: {result_test.shape}, Reference: {result_ref.shape}")
        return False

    # Find valid (non-NaN) positions in both arrays
    valid_test = ~np.isnan(result_test)
    valid_ref = ~np.isnan(result_ref)

    # Check NaN positions match
    if not np.array_equal(valid_test, valid_ref):
        nan_diff = np.sum(valid_test != valid_ref)
        print(f"  ✗ {test_name}: NaN positions differ ({nan_diff} pixels)")
        return False

    # Compare values where both are valid
    if np.any(valid_test):
        diff = np.abs(result_test[valid_test] - result_ref[valid_test])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        if max_diff > tolerance:
            print(f"  ✗ {test_name}: Values differ!")
            print(f"    Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
            print(f"    Tolerance: {tolerance:.2e}")
            return False

        print(f"  ✓ {test_name}: Max diff={max_diff:.2e}, Mean diff={mean_diff:.2e}")
    else:
        print(f"  ✓ {test_name}: All NaN (expected)")

    return True


def test_correctness_comprehensive():
    """Test correctness against reference implementation"""
    print("="*70)
    print("CORRECTNESS TEST: Compare with Reference Implementation")
    print("="*70)

    # Test configurations
    test_cases = [
        (20, 20, "Small square"),
        (30, 20, "Small rectangular"),
        (50, 50, "Medium square"),
        (100, 50, "Medium rectangular"),
    ]

    window_sizes = [3, 4, 5, 6, 7, 8, 11]
    filter_types = ['mean', 'median']
    axes = ['azimuth', 'range', 'both']

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for nrows, ncols, desc in test_cases:
        print(f"\n{'='*70}")
        print(f"Array: {desc} ({nrows}×{ncols})")
        print(f"{'='*70}")

        # Create test array with NaN
        np.random.seed(42)
        array = np.random.randn(nrows, ncols).astype(np.float64)
        array[np.random.rand(nrows, ncols) < 0.15] = np.nan

        for window_size in window_sizes:
            print(f"\nWindow size: {window_size}×{window_size}")

            for filter_type in filter_types:
                for axis in axes:
                    total_tests += 1

                    # Get result from apply_filter
                    result_test = apply_filter(array.copy(), window_size,
                                              filter_type=filter_type, axis=axis)

                    # Get reference result
                    if filter_type == 'mean':
                        result_ref = reference_mean_filter(array.copy(), window_size, axis=axis)
                    else:
                        result_ref = reference_median_filter(array.copy(), window_size, axis=axis)

                    # Compare
                    test_name = f"{filter_type:6s} {axis:8s}"
                    if compare_results(result_test, result_ref, test_name):
                        passed_tests += 1
                    else:
                        failed_tests.append({
                            'array_shape': (nrows, ncols),
                            'window_size': window_size,
                            'filter_type': filter_type,
                            'axis': axis
                        })

    # Summary
    print("\n" + "="*70)
    print("CORRECTNESS TEST RESULTS")
    print("="*70)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")

    if failed_tests:
        print("\n" + "="*70)
        print("FAILED TESTS")
        print("="*70)
        for failure in failed_tests:
            print(f"  Array: {failure['array_shape']}, Window: {failure['window_size']}, "
                  f"Filter: {failure['filter_type']}, Axis: {failure['axis']}")
        return False

    print("\n✓ All correctness tests passed!")
    return True


def test_specific_values():
    """Test with known input/output values"""
    print("\n\n" + "="*70)
    print("SPECIFIC VALUE TESTS")
    print("="*70)

    # Test 1: Constant array
    print("\nTest 1: Constant array (all 5.0)")
    array = np.full((10, 10), 5.0, dtype=np.float64)
    result = apply_filter(array, 3, filter_type='mean', axis='both')

    if np.allclose(result, 5.0):
        print("  ✓ Mean of constant array = constant: PASS")
    else:
        print(f"  ✗ Expected all 5.0, got min={np.min(result):.3f}, max={np.max(result):.3f}")
        return False

    # Test 2: Identity for window size 1
    print("\nTest 2: Identity (window size 1)")
    array = np.random.randn(20, 20)
    result = apply_filter(array, 1, filter_type='mean', axis='both')

    if np.allclose(result, array):
        print("  ✓ Window size 1 returns identity: PASS")
    else:
        max_diff = np.max(np.abs(result - array))
        print(f"  ✗ Window size 1 should be identity, max diff = {max_diff:.2e}")
        return False

    # Test 3: Monotonic array
    print("\nTest 3: Monotonic increasing array")
    array = np.arange(25, dtype=np.float64).reshape(5, 5)
    result = apply_filter(array, 3, filter_type='mean', axis='both')

    # Filtered values should be in range [min, max] of input
    if np.all(result >= np.min(array)) and np.all(result <= np.max(array)):
        print(f"  ✓ Filtered values in range [{np.min(array):.1f}, {np.max(array):.1f}]: PASS")
    else:
        print(f"  ✗ Filtered values outside input range")
        print(f"    Input: [{np.min(array):.1f}, {np.max(array):.1f}]")
        print(f"    Output: [{np.min(result):.1f}, {np.max(result):.1f}]")
        return False

    # Test 4: Known 3×3 mean
    print("\nTest 4: Known 3×3 mean calculation")
    array = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    result = apply_filter(array, 3, filter_type='mean', axis='both')

    # Center pixel should be mean of all 9 values = 5.0
    center_value = result[1, 1]
    expected = 5.0

    if np.abs(center_value - expected) < 1e-10:
        print(f"  ✓ Center pixel = {center_value:.6f} (expected {expected:.6f}): PASS")
    else:
        print(f"  ✗ Center pixel = {center_value:.6f}, expected {expected:.6f}")
        return False

    print("\n✓ All specific value tests passed!")
    return True


def test_nan_handling():
    """Test correct NaN handling"""
    print("\n\n" + "="*70)
    print("NaN HANDLING CORRECTNESS")
    print("="*70)

    # Test 1: NaN ignored in mean
    print("\nTest 1: NaN should be ignored in mean calculation")
    array = np.array([
        [1.0, 2.0, 3.0],
        [2.0, np.nan, 4.0],
        [3.0, 4.0, 5.0]
    ])

    result = apply_filter(array, 3, filter_type='mean', axis='both')
    reference = reference_mean_filter(array, 3, axis='both')

    if np.allclose(result, reference, equal_nan=True):
        print(f"  ✓ NaN correctly ignored in mean: PASS")
        print(f"    Center pixel: {result[1,1]:.6f} (reference: {reference[1,1]:.6f})")
    else:
        print(f"  ✗ NaN handling differs from reference")
        return False

    # Test 2: All-NaN window produces NaN
    print("\nTest 2: All-NaN window should produce NaN")
    array = np.full((5, 5), np.nan)
    array[2, 2] = 5.0  # One valid pixel

    result = apply_filter(array, 3, filter_type='mean', axis='both')

    # Pixels far from valid pixel should be NaN
    if np.isnan(result[0, 0]):
        print("  ✓ All-NaN window produces NaN: PASS")
    else:
        print(f"  ✗ All-NaN window produced {result[0,0]}, expected NaN")
        return False

    print("\n✓ All NaN handling tests passed!")
    return True


def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE CORRECTNESS SANITY CHECK")
    print("="*70)
    print("Verifying apply_filter produces numerically correct results")
    print("="*70)

    test1 = test_correctness_comprehensive()
    test2 = test_specific_values()
    test3 = test_nan_handling()

    print("\n\n" + "="*70)
    print("FINAL CORRECTNESS SUMMARY")
    print("="*70)
    print(f"Reference implementation comparison: {'✓ PASSED' if test1 else '✗ FAILED'}")
    print(f"Specific value tests: {'✓ PASSED' if test2 else '✗ FAILED'}")
    print(f"NaN handling tests: {'✓ PASSED' if test3 else '✗ FAILED'}")

    if test1 and test2 and test3:
        print("\n" + "="*70)
        print("✓✓✓ ALL CORRECTNESS CHECKS PASSED ✓✓✓")
        print("="*70)
        print("apply_filter produces numerically correct results for:")
        print("  - All array sizes and window sizes")
        print("  - Both mean and median filters")
        print("  - All axis modes (azimuth, range, both)")
        print("  - Proper NaN handling (ignored in statistics)")
        print("  - Edge cases (constant arrays, identity, etc.)")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print("✗✗✗ CORRECTNESS CHECK FAILURES ✗✗✗")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
