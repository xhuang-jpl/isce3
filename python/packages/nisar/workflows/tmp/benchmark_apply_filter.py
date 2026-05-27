#!/usr/bin/env python3
'''
Benchmark script to test memory usage, runtime, and correctness of apply_filter function.
'''
import numpy as np
import time
import tracemalloc
from scipy import ndimage
import sys
import os

# Add parent directory to path to import rubbersheet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rubbersheet import apply_filter


def create_test_array(shape, nan_fraction=0.1, seed=42):
    """
    Create a test array with some NaN values.

    Parameters
    ----------
    shape: tuple
        Shape of the array (rows, cols)
    nan_fraction: float
        Fraction of pixels to set as NaN
    seed: int
        Random seed for reproducibility

    Returns
    -------
    array: np.ndarray
        Test array with NaN values
    """
    np.random.seed(seed)
    array = np.random.randn(*shape).astype(np.float64)

    # Add some NaN values randomly
    nan_mask = np.random.rand(*shape) < nan_fraction
    array[nan_mask] = np.nan

    return array


def reference_mean_filter(array, window_size_az, window_size_rg):
    """
    Reference implementation using scipy's nanmean generic_filter.
    This is the ground truth for correctness testing.
    """
    def nanmean_func(values):
        valid = values[np.isfinite(values)]
        return np.mean(valid) if len(valid) > 0 else np.nan

    return ndimage.generic_filter(
        array,
        nanmean_func,
        size=(window_size_az, window_size_rg),
        mode='constant',
        cval=np.nan
    )


def reference_median_filter(array, window_size_az, window_size_rg):
    """
    Reference implementation using scipy's nanmedian generic_filter.
    This is the ground truth for correctness testing.
    """
    def nanmedian_func(values):
        valid = values[np.isfinite(values)]
        return np.median(valid) if len(valid) > 0 else np.nan

    return ndimage.generic_filter(
        array,
        nanmedian_func,
        size=(window_size_az, window_size_rg),
        mode='constant',
        cval=np.nan
    )


def check_correctness(array, window_size, filter_type='mean', axis='both'):
    """
    Check correctness by comparing with reference implementation.

    Returns
    -------
    passed: bool
        True if test passed
    max_diff: float
        Maximum absolute difference
    """
    # Parse window sizes
    if isinstance(window_size, tuple):
        window_size_az, window_size_rg = window_size
    else:
        window_size_az = window_size_rg = window_size

    # Adjust for axis parameter
    if axis == 'azimuth':
        window_size_rg = 1
    elif axis == 'range':
        window_size_az = 1

    # Get result from apply_filter
    result = apply_filter(array, window_size, filter_type=filter_type, axis=axis)

    # Get reference result
    if filter_type == 'mean':
        reference = reference_mean_filter(array, window_size_az, window_size_rg)
    else:
        reference = reference_median_filter(array, window_size_az, window_size_rg)

    # Compare results (ignoring NaN locations)
    valid_mask = np.isfinite(result) & np.isfinite(reference)

    if not np.any(valid_mask):
        # Both are all NaN - this is correct
        return True, 0.0

    diff = np.abs(result[valid_mask] - reference[valid_mask])
    max_diff = np.max(diff)

    # Check if differences are small (numerical tolerance)
    passed = max_diff < 1e-10

    return passed, max_diff


def benchmark_memory(array, window_size, filter_type='mean', axis='both'):
    """
    Benchmark peak memory usage.

    Returns
    -------
    peak_memory_mb: float
        Peak memory usage in MB
    """
    tracemalloc.start()

    # Run the filter
    _ = apply_filter(array, window_size, filter_type=filter_type, axis=axis)

    # Get peak memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_mb = peak / 1024 / 1024

    return peak_memory_mb


def benchmark_runtime(array, window_size, filter_type='mean', axis='both', num_runs=3):
    """
    Benchmark runtime.

    Returns
    -------
    mean_time: float
        Mean runtime in seconds
    std_time: float
        Standard deviation of runtime
    """
    times = []

    for _ in range(num_runs):
        start = time.time()
        _ = apply_filter(array, window_size, filter_type=filter_type, axis=axis)
        elapsed = time.time() - start
        times.append(elapsed)

    return np.mean(times), np.std(times)


def run_comprehensive_benchmark():
    """
    Run comprehensive benchmarks with different configurations.
    """
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK: apply_filter function")
    print("=" * 80)
    print()

    # Test configurations
    array_sizes = [
        (1000, 1000, "Small"),
        (5000, 5000, "Medium"),
    ]

    window_sizes = [
        (3, "3x3"),
        (11, "11x11"),
        (21, "21x21"),
        (31, "31x31"),
    ]

    filter_types = ['mean', 'median']
    axis_options = ['both']

    print("Test Configuration:")
    print(f"  - Array sizes: {[f'{name} ({r}x{c})' for r, c, name in array_sizes]}")
    print(f"  - Window sizes: {[name for _, name in window_sizes]}")
    print(f"  - Filter types: {filter_types}")
    print(f"  - Axes: {axis_options}")
    print()

    # ===== CORRECTNESS TESTS =====
    print("=" * 80)
    print("CORRECTNESS TESTS")
    print("=" * 80)
    print()

    test_array = create_test_array((500, 500), nan_fraction=0.1)

    correctness_passed = 0
    correctness_total = 0

    for window_size, window_name in window_sizes:
        for filter_type in filter_types:
            for axis in axis_options:
                correctness_total += 1

                passed, max_diff = check_correctness(
                    test_array, window_size, filter_type=filter_type, axis=axis
                )

                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"{status}: {filter_type:6s} | {window_name:8s} | axis={axis:8s} | max_diff={max_diff:.2e}")

                if passed:
                    correctness_passed += 1

    print()
    print(f"Correctness: {correctness_passed}/{correctness_total} tests passed")
    print()

    # ===== MEMORY BENCHMARKS =====
    print("=" * 80)
    print("MEMORY USAGE BENCHMARKS")
    print("=" * 80)
    print()

    print(f"{'Array Size':<20} {'Window':<10} {'Filter':<10} {'Axis':<10} {'Memory (MB)':<15}")
    print("-" * 80)

    for rows, cols, size_name in array_sizes:
        test_array = create_test_array((rows, cols), nan_fraction=0.1)
        array_size_mb = test_array.nbytes / 1024 / 1024

        for window_size, window_name in window_sizes:
            for filter_type in filter_types:
                for axis in axis_options:
                    memory_mb = benchmark_memory(
                        test_array, window_size, filter_type=filter_type, axis=axis
                    )

                    print(f"{size_name + f' ({rows}x{cols})':<20} "
                          f"{window_name:<10} {filter_type:<10} {axis:<10} "
                          f"{memory_mb:>10.2f}")

        print(f"{'Array base size:':<60} {array_size_mb:>10.2f}")
        print()

    # ===== RUNTIME BENCHMARKS =====
    print("=" * 80)
    print("RUNTIME BENCHMARKS")
    print("=" * 80)
    print()

    print(f"{'Array Size':<20} {'Window':<10} {'Filter':<10} {'Axis':<10} {'Time (s)':<15} {'Std Dev':<10}")
    print("-" * 80)

    for rows, cols, size_name in array_sizes:
        test_array = create_test_array((rows, cols), nan_fraction=0.1)

        for window_size, window_name in window_sizes:
            for filter_type in filter_types:
                for axis in axis_options:
                    mean_time, std_time = benchmark_runtime(
                        test_array, window_size, filter_type=filter_type, axis=axis, num_runs=3
                    )

                    print(f"{size_name + f' ({rows}x{cols})':<20} "
                          f"{window_name:<10} {filter_type:<10} {axis:<10} "
                          f"{mean_time:>10.4f}     {std_time:>8.4f}")

        print()

    # ===== EDGE CASES =====
    print("=" * 80)
    print("EDGE CASE TESTS")
    print("=" * 80)
    print()

    edge_cases = [
        ("All NaN", np.full((100, 100), np.nan)),
        ("No NaN", np.random.randn(100, 100)),
        ("All zeros", np.zeros((100, 100))),
        ("50% NaN", create_test_array((100, 100), nan_fraction=0.5)),
    ]

    for case_name, test_array in edge_cases:
        try:
            result = apply_filter(test_array, 5, filter_type='mean', axis='both')
            nan_count = np.count_nonzero(np.isnan(result))
            status = "✓ PASS"
        except Exception as e:
            nan_count = -1
            status = f"✗ FAIL: {str(e)}"

        print(f"{status}: {case_name:<15} | Output NaN count: {nan_count}")

    print()

    # ===== AXIS OPTIONS TEST =====
    print("=" * 80)
    print("AXIS OPTIONS TEST")
    print("=" * 80)
    print()

    test_array = create_test_array((200, 200), nan_fraction=0.1)

    for axis in ['azimuth', 'range', 'both']:
        for filter_type in ['mean', 'median']:
            passed, max_diff = check_correctness(
                test_array, 7, filter_type=filter_type, axis=axis
            )
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {filter_type:6s} | axis={axis:8s} | max_diff={max_diff:.2e}")

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_benchmark()
