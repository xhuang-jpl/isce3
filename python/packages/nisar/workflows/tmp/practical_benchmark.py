#!/usr/bin/env python3
'''
Practical benchmark for apply_filter with realistic scenarios.
'''
import numpy as np
import time
import tracemalloc
from scipy import ndimage
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rubbersheet import apply_filter


def create_test_array(shape, nan_fraction=0.1, seed=42):
    """Create test array with NaN values."""
    np.random.seed(seed)
    array = np.random.randn(*shape).astype(np.float64)
    nan_mask = np.random.rand(*shape) < nan_fraction
    array[nan_mask] = np.nan
    return array


def benchmark_case(array, window_size, filter_type, axis='both', num_runs=3):
    """
    Benchmark a single case for memory and runtime.

    Returns
    -------
    mean_time, std_time, peak_memory_mb
    """
    times = []

    # Runtime benchmark
    for _ in range(num_runs):
        start = time.time()
        result = apply_filter(array, window_size, filter_type=filter_type, axis=axis)
        elapsed = time.time() - start
        times.append(elapsed)

    # Memory benchmark
    tracemalloc.start()
    _ = apply_filter(array, window_size, filter_type=filter_type, axis=axis)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_mb = peak / 1024 / 1024

    return np.mean(times), np.std(times), peak_memory_mb


def main():
    print("=" * 80)
    print("PRACTICAL BENCHMARK: apply_filter")
    print("=" * 80)
    print()

    # Realistic test scenarios
    scenarios = [
        ((1000, 1000), "Small RSLC patch"),
        ((2000, 2000), "Medium RSLC patch"),
        ((4000, 4000), "Large RSLC patch"),
    ]

    window_sizes = [5, 11, 21, 31]
    filter_types = ['mean', 'median']

    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print()

    header = f"{'Scenario':<25} {'Filter':<8} {'Window':<8} {'Time (s)':<12} {'Std':<10} {'Memory (MB)':<12}"
    print(header)
    print("-" * 80)

    for shape, scenario_name in scenarios:
        array = create_test_array(shape, nan_fraction=0.1)
        array_size_mb = array.nbytes / 1024 / 1024

        for filter_type in filter_types:
            for window_size in window_sizes:
                mean_time, std_time, memory_mb = benchmark_case(
                    array, window_size, filter_type, axis='both', num_runs=3
                )

                print(f"{scenario_name:<25} {filter_type:<8} {window_size:<8} "
                      f"{mean_time:>10.4f}  {std_time:>8.4f}  {memory_mb:>10.2f}")

        print(f"{'  Array base size:':<63} {array_size_mb:>10.2f}")
        print()

    # Memory efficiency analysis
    print("=" * 80)
    print("MEMORY EFFICIENCY ANALYSIS")
    print("=" * 80)
    print()
    print("Comparing memory overhead vs array size:")
    print()

    test_array = create_test_array((2000, 2000), nan_fraction=0.1)
    base_size = test_array.nbytes / 1024 / 1024

    print(f"Base array size: {base_size:.2f} MB")
    print()
    print(f"{'Window Size':<15} {'Mean Memory (MB)':<20} {'Median Memory (MB)':<20} {'Overhead Factor':<15}")
    print("-" * 80)

    for ws in [5, 11, 21, 31]:
        _, _, mem_mean = benchmark_case(test_array, ws, 'mean', num_runs=1)
        _, _, mem_median = benchmark_case(test_array, ws, 'median', num_runs=1)
        overhead = max(mem_mean, mem_median) / base_size

        window_str = f"{ws}x{ws}"
        print(f"{window_str:<15} {mem_mean:>18.2f}  {mem_median:>18.2f}  {overhead:>13.2f}x")

    print()

    # Performance scaling analysis
    print("=" * 80)
    print("PERFORMANCE SCALING ANALYSIS")
    print("=" * 80)
    print()
    print("How runtime scales with window size (2000x2000 array):")
    print()

    print(f"{'Window Size':<15} {'Mean Time (s)':<18} {'Median Time (s)':<18} {'Ratio':<10}")
    print("-" * 80)

    times_mean = []
    times_median = []

    for ws in [5, 11, 21, 31]:
        t_mean, _, _ = benchmark_case(test_array, ws, 'mean', num_runs=3)
        t_median, _, _ = benchmark_case(test_array, ws, 'median', num_runs=3)
        times_mean.append(t_mean)
        times_median.append(t_median)

        ratio = t_median / t_mean if t_mean > 0 else 0

        window_str = f"{ws}x{ws}"
        print(f"{window_str:<15} {t_mean:>16.4f}  {t_median:>16.4f}  {ratio:>8.2f}x")

    print()
    print(f"Window size scaling factor (31x31 vs 5x5):")
    print(f"  Mean filter:   {times_mean[-1]/times_mean[0]:>6.2f}x slower")
    print(f"  Median filter: {times_median[-1]/times_median[0]:>6.2f}x slower")
    print()

    # Edge cases
    print("=" * 80)
    print("EDGE CASE VALIDATION")
    print("=" * 80)
    print()

    edge_cases = [
        ("All NaN", np.full((500, 500), np.nan)),
        ("No NaN", np.random.randn(500, 500)),
        ("50% NaN", create_test_array((500, 500), nan_fraction=0.5)),
        ("All zeros", np.zeros((500, 500))),
        ("Single value", np.ones((500, 500)) * 3.14),
    ]

    for case_name, test_array in edge_cases:
        try:
            result = apply_filter(test_array, 11, filter_type='mean', axis='both')
            nan_in = np.count_nonzero(np.isnan(test_array))
            nan_out = np.count_nonzero(np.isnan(result))
            mean_val = np.nanmean(result) if np.any(np.isfinite(result)) else np.nan
            status = "✓"
        except Exception as e:
            nan_in = nan_out = -1
            mean_val = np.nan
            status = f"✗ {str(e)}"

        print(f"{status} {case_name:<20} | NaN in: {nan_in:>6} | NaN out: {nan_out:>6} | mean: {mean_val:>10.4f}")

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
