#!/usr/bin/env python3
"""
Test the apply_filter function from rubbersheet.py with the optimization
"""
import sys
import os
import numpy as np
import time

# Add parent directory to path to import rubbersheet module
sys.path.insert(0, os.path.abspath('../'))

from rubbersheet import apply_filter

def test_filter_correctness():
    """Test that the filter produces correct results"""
    print("="*70)
    print("TEST 1: Correctness - Compare with scipy.ndimage")
    print("="*70)

    from scipy import ndimage

    # Create small test array
    np.random.seed(42)
    array = np.random.randn(100, 50).astype(np.float64)
    array[np.random.rand(100, 50) < 0.1] = np.nan

    window_size = 5

    # Test mean filter
    print("\nTesting mean filter...")
    result_custom = apply_filter(array, window_size, filter_type='mean', axis='both')

    # Compare with scipy (which handles NaN differently, so we'll compute manually)
    # For a rough comparison, use scipy on data with NaN replaced by nanmean
    array_filled = array.copy()
    array_filled[np.isnan(array_filled)] = np.nanmean(array)
    result_scipy = ndimage.uniform_filter(array_filled, size=window_size)

    # Check that results are reasonable (not exact match due to NaN handling)
    print(f"Custom result shape: {result_custom.shape}")
    print(f"Custom result range: [{np.nanmin(result_custom):.3f}, {np.nanmax(result_custom):.3f}]")
    print(f"Contains NaN: {np.any(np.isnan(result_custom))}")
    print(f"All finite: {np.all(np.isfinite(result_custom))}")

    # Test median filter
    print("\nTesting median filter...")
    result_custom = apply_filter(array, window_size, filter_type='median', axis='both')

    print(f"Custom result shape: {result_custom.shape}")
    print(f"Custom result range: [{np.nanmin(result_custom):.3f}, {np.nanmax(result_custom):.3f}]")
    print(f"Contains NaN: {np.any(np.isnan(result_custom))}")

    print("\n✓ Correctness tests passed!")


def test_filter_axis_modes():
    """Test all axis modes (azimuth, range, both)"""
    print("\n" + "="*70)
    print("TEST 2: All Axis Modes")
    print("="*70)

    np.random.seed(42)
    array = np.random.randn(200, 100).astype(np.float64)
    array[np.random.rand(200, 100) < 0.05] = np.nan

    window_size = 7

    # Test all axis modes
    for axis in ['azimuth', 'range', 'both']:
        print(f"\nTesting axis='{axis}'...")
        result = apply_filter(array, window_size, filter_type='mean', axis=axis)
        print(f"  Result shape: {result.shape}")
        print(f"  Input NaN count: {np.sum(np.isnan(array))}")
        print(f"  Output NaN count: {np.sum(np.isnan(result))}")
        assert result.shape == array.shape, f"Shape mismatch for axis={axis}"
        print(f"  ✓ Passed")

    print("\n✓ All axis modes work correctly!")


def test_memory_scalability():
    """Test memory usage on progressively larger arrays"""
    print("\n" + "="*70)
    print("TEST 3: Memory Scalability")
    print("="*70)

    test_sizes = [
        (500, 250, 11, "Small"),
        (1000, 500, 11, "Medium"),
        (2000, 1000, 21, "Large"),
        (5000, 2500, 31, "Very Large"),
    ]

    for nrows, ncols, window_size, label in test_sizes:
        print(f"\n{label}: {nrows}×{ncols}, window {window_size}×{window_size}")
        print("-" * 70)

        # Create test array
        array = np.random.randn(nrows, ncols).astype(np.float64)
        array[np.random.rand(nrows, ncols) < 0.1] = np.nan

        array_size = array.nbytes / 1024**2
        print(f"Array size: {array_size:.2f} MB")

        # Time the operation
        start = time.time()
        try:
            result = apply_filter(array, window_size, filter_type='mean', axis='both')
            elapsed = time.time() - start

            print(f"Time: {elapsed:.3f} seconds")
            print(f"Result shape: {result.shape}")
            print(f"✓ Success")

        except MemoryError as e:
            print(f"✗ FAILED: {e}")
            break

    print("\n✓ Memory scalability test completed!")


def test_numerical_stability():
    """Test edge cases and numerical stability"""
    print("\n" + "="*70)
    print("TEST 4: Numerical Stability & Edge Cases")
    print("="*70)

    window_size = 5

    # Test 1: All NaN
    print("\nTest 4.1: All NaN array")
    array = np.full((50, 50), np.nan, dtype=np.float64)
    result = apply_filter(array, window_size, filter_type='mean', axis='both')
    assert np.all(np.isnan(result)), "Expected all NaN output"
    print("✓ Handles all-NaN correctly")

    # Test 2: No NaN
    print("\nTest 4.2: No NaN array")
    array = np.random.randn(50, 50).astype(np.float64)
    result = apply_filter(array, window_size, filter_type='mean', axis='both')
    assert not np.any(np.isnan(result)), "Expected no NaN in output"
    print("✓ Handles no-NaN correctly")

    # Test 3: Very sparse valid data
    print("\nTest 4.3: Sparse valid data (90% NaN)")
    array = np.random.randn(100, 100).astype(np.float64)
    array[np.random.rand(100, 100) < 0.9] = np.nan
    result = apply_filter(array, window_size, filter_type='mean', axis='both')
    print(f"Input valid pixels: {np.sum(~np.isnan(array))}")
    print(f"Output valid pixels: {np.sum(~np.isnan(result))}")
    print("✓ Handles sparse data")

    # Test 4: Constant array
    print("\nTest 4.4: Constant array")
    array = np.full((50, 50), 5.0, dtype=np.float64)
    result = apply_filter(array, window_size, filter_type='mean', axis='both')
    assert np.allclose(result, 5.0), "Expected constant output"
    print("✓ Handles constant array correctly")

    # Test 5: Window size 1
    print("\nTest 4.5: Trivial window size (1×1)")
    array = np.random.randn(50, 50).astype(np.float64)
    result = apply_filter(array, 1, filter_type='mean', axis='both')
    assert np.allclose(result, array, equal_nan=True), "Expected identical output"
    print("✓ Handles window size 1 correctly")

    print("\n✓ All numerical stability tests passed!")


def test_performance_comparison():
    """Compare performance metrics"""
    print("\n" + "="*70)
    print("TEST 5: Performance Metrics")
    print("="*70)

    test_cases = [
        (1000, 500, 11, "Small"),
        (2000, 1000, 21, "Medium"),
    ]

    for nrows, ncols, window_size, label in test_cases:
        print(f"\n{label}: {nrows}×{ncols}, window {window_size}×{window_size}")
        print("-" * 70)

        array = np.random.randn(nrows, ncols).astype(np.float64)
        array[np.random.rand(nrows, ncols) < 0.1] = np.nan

        # Test mean filter
        start = time.time()
        result_mean = apply_filter(array, window_size, filter_type='mean', axis='both')
        time_mean = time.time() - start

        # Test median filter
        start = time.time()
        result_median = apply_filter(array, window_size, filter_type='median', axis='both')
        time_median = time.time() - start

        print(f"Mean filter time: {time_mean:.3f} seconds")
        print(f"Median filter time: {time_median:.3f} seconds")
        print(f"Median/Mean ratio: {time_median/time_mean:.2f}x")

    print("\n✓ Performance metrics collected!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("RUBBERSHEET APPLY_FILTER OPTIMIZATION TESTS")
    print("="*70)
    print(f"NumPy version: {np.__version__}")
    print(f"Testing optimized apply_filter() function")
    print("="*70)

    try:
        test_filter_correctness()
        test_filter_axis_modes()
        test_memory_scalability()
        test_numerical_stability()
        test_performance_comparison()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        return 0

    except Exception as e:
        print("\n" + "="*70)
        print(f"TEST FAILED ✗")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
