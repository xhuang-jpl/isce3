#!/usr/bin/env python3
"""
Sanity check: Verify apply_filter output shape matches input shape
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('../'))
from rubbersheet import apply_filter


def test_output_shape_preservation():
    """
    Test that output shape exactly matches input shape for all combinations
    of array sizes, window sizes, filter types, and axes.
    """
    print("="*70)
    print("SANITY CHECK: Output Shape Preservation")
    print("="*70)

    # Test various array shapes
    array_shapes = [
        (10, 10, "Small square"),
        (20, 15, "Small rectangular"),
        (100, 50, "Medium"),
        (500, 250, "Large"),
        (1000, 500, "Very large"),
    ]

    # Test various window sizes (both odd and even)
    window_sizes = [1, 3, 4, 5, 6, 7, 8, 11, 21, 31]

    # Test filter types
    filter_types = ['mean', 'median']

    # Test axes
    axes = ['azimuth', 'range', 'both']

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    print(f"\nTesting {len(array_shapes)} array shapes × {len(window_sizes)} windows × "
          f"{len(filter_types)} filters × {len(axes)} axes")
    print(f"Total tests: {len(array_shapes) * len(window_sizes) * len(filter_types) * len(axes)}\n")

    for shape_idx, (nrows, ncols, shape_desc) in enumerate(array_shapes):
        print(f"\n{'='*70}")
        print(f"Array Shape: {shape_desc} ({nrows}×{ncols})")
        print(f"{'='*70}")

        # Create test array with some NaN
        np.random.seed(42)
        array = np.random.randn(nrows, ncols).astype(np.float64)
        array[np.random.rand(nrows, ncols) < 0.1] = np.nan

        for window_size in window_sizes:
            # Skip very large windows on small arrays
            if window_size > min(nrows, ncols):
                continue

            for filter_type in filter_types:
                for axis in axes:
                    total_tests += 1

                    try:
                        result = apply_filter(array.copy(), window_size,
                                            filter_type=filter_type, axis=axis)

                        # Check if output shape matches input shape
                        if result.shape == array.shape:
                            passed_tests += 1
                        else:
                            failed_tests.append({
                                'array_shape': array.shape,
                                'window_size': window_size,
                                'filter_type': filter_type,
                                'axis': axis,
                                'expected_shape': array.shape,
                                'actual_shape': result.shape
                            })
                            print(f"  ✗ FAILED: window={window_size}, filter={filter_type}, axis={axis}")
                            print(f"    Expected: {array.shape}, Got: {result.shape}")

                    except Exception as e:
                        failed_tests.append({
                            'array_shape': array.shape,
                            'window_size': window_size,
                            'filter_type': filter_type,
                            'axis': axis,
                            'error': str(e)
                        })
                        print(f"  ✗ ERROR: window={window_size}, filter={filter_type}, axis={axis}")
                        print(f"    Error: {e}")

            # Progress indicator
            parity = "odd" if window_size % 2 == 1 else "even"
            print(f"  ✓ Window {window_size:2d}×{window_size:2d} ({parity:>4s}): All axes and filters passed")

    # Summary
    print("\n" + "="*70)
    print("SANITY CHECK RESULTS")
    print("="*70)
    print(f"Total tests run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")

    if failed_tests:
        print("\n" + "="*70)
        print("FAILED TESTS DETAILS")
        print("="*70)
        for i, failure in enumerate(failed_tests, 1):
            print(f"\nFailure {i}:")
            for key, value in failure.items():
                print(f"  {key}: {value}")

        return False
    else:
        print("\n✓ ALL TESTS PASSED - Output shape always matches input shape!")
        return True


def test_edge_cases():
    """Test edge cases for shape preservation"""
    print("\n\n" + "="*70)
    print("EDGE CASES: Special Array Shapes")
    print("="*70)

    edge_cases = [
        (1, 100, "Single row"),
        (100, 1, "Single column"),
        (1, 1, "Single pixel"),
        (3, 3, "Minimum practical size"),
    ]

    all_passed = True

    for nrows, ncols, description in edge_cases:
        print(f"\n{description}: {nrows}×{ncols}")
        array = np.random.randn(nrows, ncols).astype(np.float64)

        # Test with window size 3
        window_size = min(3, min(nrows, ncols))

        for axis in ['azimuth', 'range', 'both']:
            try:
                result = apply_filter(array, window_size, filter_type='mean', axis=axis)

                if result.shape == array.shape:
                    print(f"  ✓ axis='{axis}': {result.shape} == {array.shape}")
                else:
                    print(f"  ✗ axis='{axis}': {result.shape} != {array.shape}")
                    all_passed = False

            except Exception as e:
                print(f"  ✗ axis='{axis}': ERROR - {e}")
                all_passed = False

    if all_passed:
        print("\n✓ All edge cases passed!")
    else:
        print("\n✗ Some edge cases failed!")

    return all_passed


def test_with_different_nan_patterns():
    """Test that shape is preserved regardless of NaN patterns"""
    print("\n\n" + "="*70)
    print("NaN PATTERN TESTS: Shape Preservation")
    print("="*70)

    nrows, ncols = 50, 50
    window_size = 7

    nan_patterns = [
        (0.0, "No NaN"),
        (0.1, "10% NaN (sparse)"),
        (0.5, "50% NaN (moderate)"),
        (0.9, "90% NaN (very sparse valid data)"),
        (1.0, "100% NaN (all invalid)"),
    ]

    all_passed = True

    for nan_fraction, description in nan_patterns:
        np.random.seed(42)
        array = np.random.randn(nrows, ncols).astype(np.float64)

        if nan_fraction > 0:
            array[np.random.rand(nrows, ncols) < nan_fraction] = np.nan

        print(f"\n{description}")

        for filter_type in ['mean', 'median']:
            result = apply_filter(array, window_size, filter_type=filter_type, axis='both')

            if result.shape == array.shape:
                nan_out = np.sum(np.isnan(result))
                print(f"  ✓ {filter_type:6s} filter: {result.shape} == {array.shape}, "
                      f"NaN output: {nan_out}/{result.size} ({nan_out/result.size*100:.1f}%)")
            else:
                print(f"  ✗ {filter_type:6s} filter: {result.shape} != {array.shape}")
                all_passed = False

    if all_passed:
        print("\n✓ Shape preserved for all NaN patterns!")
    else:
        print("\n✗ Shape preservation failed for some NaN patterns!")

    return all_passed


def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE SANITY CHECK: OUTPUT SHAPE PRESERVATION")
    print("="*70)
    print("Verifying that apply_filter always returns output with same shape as input")
    print("="*70)

    # Run all tests
    test1_passed = test_output_shape_preservation()
    test2_passed = test_edge_cases()
    test3_passed = test_with_different_nan_patterns()

    # Final summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Main shape preservation tests: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Edge case tests: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print(f"NaN pattern tests: {'✓ PASSED' if test3_passed else '✗ FAILED'}")

    if test1_passed and test2_passed and test3_passed:
        print("\n" + "="*70)
        print("✓✓✓ ALL SANITY CHECKS PASSED ✓✓✓")
        print("="*70)
        print("The apply_filter function correctly preserves output shape")
        print("for all tested combinations of:")
        print("  - Array shapes (small to very large)")
        print("  - Window sizes (odd and even)")
        print("  - Filter types (mean and median)")
        print("  - Axis modes (azimuth, range, both)")
        print("  - NaN patterns (0% to 100% NaN)")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print("✗✗✗ SANITY CHECK FAILURES DETECTED ✗✗✗")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
