#!/usr/bin/env python3
'''
Comprehensive correctness test for apply_filter function.
'''
import numpy as np
import sys
import os
from scipy import ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rubbersheet import apply_filter


def reference_implementation(array, window_size_az, window_size_rg, filter_type):
    """
    Reference implementation using scipy's generic_filter directly.
    This is what we expect the function to produce.
    """
    array_clean = array.copy()
    array_clean[~np.isfinite(array_clean)] = np.nan

    filter_func = np.nanmean if filter_type == 'mean' else np.nanmedian

    return ndimage.generic_filter(
        array_clean,
        filter_func,
        size=(window_size_az, window_size_rg),
        mode='constant',
        cval=np.nan
    )


def compare_arrays(result, reference, test_name, tolerance=1e-10):
    """
    Compare two arrays and report differences.

    Returns
    -------
    passed: bool
        True if arrays match within tolerance
    """
    # Check shapes match
    if result.shape != reference.shape:
        print(f"✗ {test_name}: Shape mismatch! result={result.shape}, reference={reference.shape}")
        return False

    # Check NaN locations match
    result_nan = np.isnan(result)
    reference_nan = np.isnan(reference)

    if not np.array_equal(result_nan, reference_nan):
        nan_diff = np.sum(result_nan != reference_nan)
        print(f"✗ {test_name}: NaN locations differ at {nan_diff} positions")
        print(f"  Result NaN count: {np.sum(result_nan)}, Reference NaN count: {np.sum(reference_nan)}")
        return False

    # Check values at non-NaN locations
    valid_mask = ~result_nan

    if not np.any(valid_mask):
        # Both all NaN - this is correct
        print(f"✓ {test_name}: Both arrays are all NaN (correct)")
        return True

    result_vals = result[valid_mask]
    reference_vals = reference[valid_mask]

    # Check for exact match first
    if np.array_equal(result_vals, reference_vals):
        print(f"✓ {test_name}: Exact match")
        return True

    # Check within tolerance
    diff = np.abs(result_vals - reference_vals)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    if max_diff < tolerance:
        print(f"✓ {test_name}: Match within tolerance (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
        return True
    else:
        print(f"✗ {test_name}: Values differ beyond tolerance!")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Tolerance: {tolerance:.2e}")

        # Show some examples of differences
        large_diff_idx = np.where(diff > tolerance)[0][:5]
        print(f"  Example differences:")
        for idx in large_diff_idx:
            print(f"    result={result_vals[idx]:.6f}, reference={reference_vals[idx]:.6f}, diff={diff[idx]:.2e}")

        return False


def test_basic_configurations():
    """Test basic window sizes and filter types."""
    print("=" * 80)
    print("TEST: Basic Configurations")
    print("=" * 80)
    print()

    np.random.seed(42)
    test_array = np.random.randn(100, 100)
    test_array[::10, ::10] = np.nan  # Add some NaN values

    configs = [
        (3, 'mean', 'both'),
        (5, 'mean', 'both'),
        (7, 'mean', 'both'),
        (11, 'mean', 'both'),
        (21, 'mean', 'both'),
        (31, 'mean', 'both'),
        (3, 'median', 'both'),
        (5, 'median', 'both'),
        (7, 'median', 'both'),
        (11, 'median', 'both'),
        (21, 'median', 'both'),
        (31, 'median', 'both'),
    ]

    passed = 0
    total = 0

    for window_size, filter_type, axis in configs:
        total += 1
        result = apply_filter(test_array, window_size, filter_type=filter_type, axis=axis)
        reference = reference_implementation(test_array, window_size, window_size, filter_type)

        test_name = f"{filter_type:6s} | {window_size:2d}x{window_size:2d} | axis={axis}"
        if compare_arrays(result, reference, test_name):
            passed += 1

    print()
    print(f"Result: {passed}/{total} tests passed")
    print()
    return passed == total


def test_axis_options():
    """Test different axis options."""
    print("=" * 80)
    print("TEST: Axis Options")
    print("=" * 80)
    print()

    np.random.seed(123)
    test_array = np.random.randn(80, 80)
    test_array[::8, ::8] = np.nan

    configs = [
        (5, 'mean', 'azimuth'),
        (5, 'mean', 'range'),
        (5, 'mean', 'both'),
        (11, 'median', 'azimuth'),
        (11, 'median', 'range'),
        (11, 'median', 'both'),
    ]

    passed = 0
    total = 0

    for window_size, filter_type, axis in configs:
        total += 1
        result = apply_filter(test_array, window_size, filter_type=filter_type, axis=axis)

        # Compute expected window sizes based on axis
        ws_az = window_size if axis in ['both', 'azimuth'] else 1
        ws_rg = window_size if axis in ['both', 'range'] else 1

        reference = reference_implementation(test_array, ws_az, ws_rg, filter_type)

        test_name = f"{filter_type:6s} | {window_size:2d}x{window_size:2d} | axis={axis:8s}"
        if compare_arrays(result, reference, test_name):
            passed += 1

    print()
    print(f"Result: {passed}/{total} tests passed")
    print()
    return passed == total


def test_edge_cases():
    """Test edge cases."""
    print("=" * 80)
    print("TEST: Edge Cases")
    print("=" * 80)
    print()

    edge_cases = [
        ("All NaN", np.full((50, 50), np.nan)),
        ("No NaN", np.random.randn(50, 50)),
        ("50% NaN", None),  # Will create below
        ("90% NaN", None),  # Will create below
        ("All zeros", np.zeros((50, 50))),
        ("All ones", np.ones((50, 50))),
        ("Single value", np.ones((50, 50)) * 3.14159),
        ("With Inf", None),  # Will create below
        ("Negative values", np.random.randn(50, 50) - 5),
    ]

    # Create special cases
    np.random.seed(456)
    arr_50_nan = np.random.randn(50, 50)
    arr_50_nan[np.random.rand(50, 50) < 0.5] = np.nan
    edge_cases[2] = ("50% NaN", arr_50_nan)

    arr_90_nan = np.random.randn(50, 50)
    arr_90_nan[np.random.rand(50, 50) < 0.9] = np.nan
    edge_cases[3] = ("90% NaN", arr_90_nan)

    arr_with_inf = np.random.randn(50, 50)
    arr_with_inf[::5, ::5] = np.inf
    arr_with_inf[1::5, 1::5] = -np.inf
    edge_cases[7] = ("With Inf", arr_with_inf)

    passed = 0
    total = 0

    for case_name, test_array in edge_cases:
        for filter_type in ['mean', 'median']:
            total += 1
            result = apply_filter(test_array, 7, filter_type=filter_type, axis='both')
            reference = reference_implementation(test_array, 7, 7, filter_type)

            test_name = f"{case_name:20s} | {filter_type:6s}"
            if compare_arrays(result, reference, test_name):
                passed += 1

    print()
    print(f"Result: {passed}/{total} tests passed")
    print()
    return passed == total


def test_even_window_sizes():
    """Test even window sizes (e.g., 4x4, 6x6)."""
    print("=" * 80)
    print("TEST: Even Window Sizes")
    print("=" * 80)
    print()

    np.random.seed(789)
    test_array = np.random.randn(100, 100)
    test_array[::12, ::12] = np.nan

    even_sizes = [4, 6, 8, 10, 20, 30]

    passed = 0
    total = 0

    for window_size in even_sizes:
        for filter_type in ['mean', 'median']:
            total += 1
            result = apply_filter(test_array, window_size, filter_type=filter_type, axis='both')
            reference = reference_implementation(test_array, window_size, window_size, filter_type)

            test_name = f"{filter_type:6s} | {window_size:2d}x{window_size:2d} (even)"
            if compare_arrays(result, reference, test_name):
                passed += 1

    print()
    print(f"Result: {passed}/{total} tests passed")
    print()
    return passed == total


def test_tuple_window_sizes():
    """Test tuple window sizes (different azimuth and range sizes)."""
    print("=" * 80)
    print("TEST: Tuple Window Sizes (azimuth, range)")
    print("=" * 80)
    print()

    np.random.seed(321)
    test_array = np.random.randn(100, 100)
    test_array[::10, ::10] = np.nan

    tuple_sizes = [
        (3, 5),
        (5, 3),
        (7, 11),
        (11, 7),
        (21, 11),
        (11, 21),
    ]

    passed = 0
    total = 0

    for window_size_tuple in tuple_sizes:
        for filter_type in ['mean', 'median']:
            total += 1
            result = apply_filter(test_array, window_size_tuple, filter_type=filter_type, axis='both')
            reference = reference_implementation(test_array, window_size_tuple[0], window_size_tuple[1], filter_type)

            test_name = f"{filter_type:6s} | {window_size_tuple[0]:2d}x{window_size_tuple[1]:2d} (tuple)"
            if compare_arrays(result, reference, test_name):
                passed += 1

    print()
    print(f"Result: {passed}/{total} tests passed")
    print()
    return passed == total


def test_small_arrays():
    """Test on small arrays."""
    print("=" * 80)
    print("TEST: Small Arrays")
    print("=" * 80)
    print()

    np.random.seed(654)

    small_arrays = [
        (5, 5),
        (10, 10),
        (20, 30),
        (30, 20),
    ]

    passed = 0
    total = 0

    for shape in small_arrays:
        test_array = np.random.randn(*shape)
        test_array[::3, ::3] = np.nan

        for window_size in [3, 5]:
            for filter_type in ['mean', 'median']:
                total += 1
                result = apply_filter(test_array, window_size, filter_type=filter_type, axis='both')
                reference = reference_implementation(test_array, window_size, window_size, filter_type)

                test_name = f"shape={shape} | {filter_type:6s} | {window_size}x{window_size}"
                if compare_arrays(result, reference, test_name):
                    passed += 1

    print()
    print(f"Result: {passed}/{total} tests passed")
    print()
    return passed == total


def test_trivial_cases():
    """Test trivial cases that should return a copy."""
    print("=" * 80)
    print("TEST: Trivial Cases (should return copy)")
    print("=" * 80)
    print()

    np.random.seed(111)
    test_array = np.random.randn(50, 50)
    test_array[::5, ::5] = np.nan

    # Window size 1x1 should return a copy
    result = apply_filter(test_array, 1, filter_type='mean', axis='both')

    if np.array_equal(result, test_array, equal_nan=True):
        print("✓ Window 1x1 returns copy of input")
    else:
        print("✗ Window 1x1 does not return copy of input")
        return False

    # All NaN array
    all_nan_array = np.full((50, 50), np.nan)
    result = apply_filter(all_nan_array, 5, filter_type='mean', axis='both')

    if np.all(np.isnan(result)):
        print("✓ All NaN input returns all NaN output")
    else:
        print("✗ All NaN input does not return all NaN output")
        return False

    print()
    return True


def main():
    """Run all correctness tests."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "CORRECTNESS TEST SUITE" + " " * 36 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    all_passed = True

    all_passed &= test_basic_configurations()
    all_passed &= test_axis_options()
    all_passed &= test_edge_cases()
    all_passed &= test_even_window_sizes()
    all_passed &= test_tuple_window_sizes()
    all_passed &= test_small_arrays()
    all_passed &= test_trivial_cases()

    print()
    print("=" * 80)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("=" * 80)
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
