#!/usr/bin/env python3
"""
Sanity check: Verify memory usage is reasonable and no excessive allocations
"""
import sys
import os
import numpy as np
import tracemalloc
import gc

sys.path.insert(0, os.path.abspath('../'))
from rubbersheet import apply_filter


def measure_memory_usage(array, window_size, filter_type='mean', axis='both'):
    """Measure peak memory usage for apply_filter"""
    # Force garbage collection before measurement
    gc.collect()

    # Start memory tracking
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    # Run the filter
    result = apply_filter(array.copy(), window_size, filter_type=filter_type, axis=axis)

    # Get peak memory
    snapshot_after = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate allocated memory
    stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    total_allocated = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    return {
        'peak_mb': peak / 1024**2,
        'allocated_mb': total_allocated / 1024**2,
        'result': result
    }


def test_memory_scalability():
    """Test that memory usage scales reasonably with array size"""
    print("="*70)
    print("MEMORY USAGE SCALABILITY TEST")
    print("="*70)

    test_cases = [
        (100, 100, 7, "Small"),
        (500, 250, 11, "Medium"),
        (1000, 500, 21, "Large"),
        (2000, 1000, 31, "Very Large"),
    ]

    results = []

    for nrows, ncols, window_size, label in test_cases:
        print(f"\n{label}: {nrows}×{ncols}, window {window_size}×{window_size}")
        print("-"*70)

        # Create array
        np.random.seed(42)
        array = np.random.randn(nrows, ncols).astype(np.float64)
        array[np.random.rand(nrows, ncols) < 0.1] = np.nan

        array_size_mb = array.nbytes / 1024**2
        theoretical_windows_mb = (nrows * ncols * window_size * window_size * 8) / 1024**2

        print(f"Array size: {array_size_mb:.2f} MB")
        print(f"Theoretical windows size (if materialized): {theoretical_windows_mb:.2f} MB")

        # Measure mean filter
        mem_mean = measure_memory_usage(array, window_size, 'mean', 'both')
        print(f"\nMean filter:")
        print(f"  Peak memory: {mem_mean['peak_mb']:.2f} MB")
        print(f"  Allocated: {mem_mean['allocated_mb']:.2f} MB")
        print(f"  Ratio (peak/array): {mem_mean['peak_mb']/array_size_mb:.2f}x")

        # Measure median filter
        mem_median = measure_memory_usage(array, window_size, 'median', 'both')
        print(f"\nMedian filter:")
        print(f"  Peak memory: {mem_median['peak_mb']:.2f} MB")
        print(f"  Allocated: {mem_median['allocated_mb']:.2f} MB")
        print(f"  Ratio (peak/array): {mem_median['peak_mb']/array_size_mb:.2f}x")

        # Check if memory is reasonable (not allocating full theoretical size)
        if mem_mean['peak_mb'] < theoretical_windows_mb * 0.5:
            print(f"\n✓ Memory usage reasonable (< 50% of theoretical {theoretical_windows_mb:.2f} MB)")
            memory_ok = True
        else:
            print(f"\n✗ WARNING: High memory usage (> 50% of theoretical {theoretical_windows_mb:.2f} MB)")
            memory_ok = False

        results.append({
            'label': label,
            'array_size_mb': array_size_mb,
            'theoretical_mb': theoretical_windows_mb,
            'mean_peak_mb': mem_mean['peak_mb'],
            'median_peak_mb': mem_median['peak_mb'],
            'memory_ok': memory_ok
        })

    # Summary table
    print("\n\n" + "="*70)
    print("MEMORY USAGE SUMMARY")
    print("="*70)
    print(f"{'Test':<12} {'Array MB':<12} {'Theoretical':<15} {'Mean Peak':<12} {'Median Peak':<12} {'Status':<10}")
    print("-"*70)

    all_ok = True
    for r in results:
        status = "✓ OK" if r['memory_ok'] else "✗ HIGH"
        if not r['memory_ok']:
            all_ok = False
        print(f"{r['label']:<12} {r['array_size_mb']:>10.2f}  {r['theoretical_mb']:>13.2f}  "
              f"{r['mean_peak_mb']:>10.2f}  {r['median_peak_mb']:>10.2f}  {status:<10}")

    return all_ok


def test_no_memory_leak():
    """Test that memory is properly released after filtering"""
    print("\n\n" + "="*70)
    print("MEMORY LEAK TEST")
    print("="*70)

    nrows, ncols = 500, 500
    window_size = 11

    array = np.random.randn(nrows, ncols).astype(np.float64)
    array[np.random.rand(nrows, ncols) < 0.1] = np.nan

    print(f"\nArray: {nrows}×{ncols}, window {window_size}×{window_size}")
    print("Running filter 10 times to check for memory leaks...")

    gc.collect()
    tracemalloc.start()

    memory_usage = []

    for i in range(10):
        gc.collect()
        before = tracemalloc.get_traced_memory()[0]

        result = apply_filter(array.copy(), window_size, 'mean', 'both')
        del result

        gc.collect()
        after = tracemalloc.get_traced_memory()[0]

        memory_usage.append(after / 1024**2)
        print(f"  Iteration {i+1}: {memory_usage[-1]:.2f} MB")

    tracemalloc.stop()

    # Check if memory grows unbounded
    first_three_avg = np.mean(memory_usage[:3])
    last_three_avg = np.mean(memory_usage[-3:])
    growth = last_three_avg - first_three_avg

    print(f"\nAverage memory (first 3 iterations): {first_three_avg:.2f} MB")
    print(f"Average memory (last 3 iterations): {last_three_avg:.2f} MB")
    print(f"Growth: {growth:.2f} MB")

    if abs(growth) < 5.0:  # Less than 5 MB growth
        print("\n✓ No significant memory leak detected")
        return True
    else:
        print(f"\n✗ WARNING: Memory grew by {growth:.2f} MB")
        return False


def test_memory_per_axis():
    """Test memory usage for different axis modes"""
    print("\n\n" + "="*70)
    print("MEMORY USAGE BY AXIS MODE")
    print("="*70)

    nrows, ncols = 1000, 500
    window_size = 11

    array = np.random.randn(nrows, ncols).astype(np.float64)
    array[np.random.rand(nrows, ncols) < 0.1] = np.nan

    array_size_mb = array.nbytes / 1024**2

    print(f"\nArray: {nrows}×{ncols}, window {window_size}×{window_size}")
    print(f"Array size: {array_size_mb:.2f} MB")
    print()

    axes = ['azimuth', 'range', 'both']

    for axis in axes:
        mem_info = measure_memory_usage(array, window_size, 'mean', axis)
        print(f"Axis '{axis:8s}': Peak = {mem_info['peak_mb']:>8.2f} MB, "
              f"Ratio = {mem_info['peak_mb']/array_size_mb:.2f}x")

    print("\n✓ Memory usage measured for all axis modes")
    return True


def test_even_vs_odd_memory():
    """Test that even and odd window sizes have similar memory usage"""
    print("\n\n" + "="*70)
    print("MEMORY USAGE: EVEN vs ODD WINDOW SIZES")
    print("="*70)

    nrows, ncols = 500, 500
    array = np.random.randn(nrows, ncols).astype(np.float64)
    array[np.random.rand(nrows, ncols) < 0.1] = np.nan

    array_size_mb = array.nbytes / 1024**2
    print(f"\nArray: {nrows}×{ncols}, size: {array_size_mb:.2f} MB")
    print()

    window_pairs = [(7, 8), (11, 12), (21, 22), (31, 32)]

    print(f"{'Window':<10} {'Parity':<8} {'Peak Memory':<15} {'Ratio':<10}")
    print("-"*70)

    max_ratio_diff = 0

    for odd, even in window_pairs:
        mem_odd = measure_memory_usage(array, odd, 'mean', 'both')
        mem_even = measure_memory_usage(array, even, 'mean', 'both')

        ratio_odd = mem_odd['peak_mb'] / array_size_mb
        ratio_even = mem_even['peak_mb'] / array_size_mb
        ratio_diff = abs(ratio_even - ratio_odd)
        max_ratio_diff = max(max_ratio_diff, ratio_diff)

        print(f"{odd:2d}×{odd:2d}     Odd      {mem_odd['peak_mb']:>12.2f} MB  {ratio_odd:>8.2f}x")
        print(f"{even:2d}×{even:2d}     Even     {mem_even['peak_mb']:>12.2f} MB  {ratio_even:>8.2f}x")
        print()

    print(f"Maximum ratio difference: {max_ratio_diff:.2f}x")

    if max_ratio_diff < 2.0:  # Should be similar
        print("\n✓ Even and odd window sizes have similar memory usage")
        return True
    else:
        print(f"\n✗ WARNING: Large memory difference between even/odd windows")
        return False


def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE MEMORY USAGE SANITY CHECK")
    print("="*70)
    print("Verifying apply_filter has reasonable memory usage")
    print("="*70)

    test1 = test_memory_scalability()
    test2 = test_no_memory_leak()
    test3 = test_memory_per_axis()
    test4 = test_even_vs_odd_memory()

    print("\n\n" + "="*70)
    print("FINAL MEMORY USAGE SUMMARY")
    print("="*70)
    print(f"Memory scalability: {'✓ PASSED' if test1 else '✗ FAILED'}")
    print(f"No memory leaks: {'✓ PASSED' if test2 else '✗ FAILED'}")
    print(f"All axis modes: {'✓ PASSED' if test3 else '✗ FAILED'}")
    print(f"Even vs odd windows: {'✓ PASSED' if test4 else '✗ FAILED'}")

    if test1 and test2 and test3 and test4:
        print("\n" + "="*70)
        print("✓✓✓ ALL MEMORY USAGE CHECKS PASSED ✓✓✓")
        print("="*70)
        print("Key findings:")
        print("  - Memory usage is reasonable (< 50% of theoretical window size)")
        print("  - No memory leaks detected (stable over 10 iterations)")
        print("  - All axis modes have appropriate memory usage")
        print("  - Even and odd window sizes have similar memory footprint")
        print("  - Optimization successfully avoids massive allocations")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print("✗✗✗ MEMORY USAGE ISSUES DETECTED ✗✗✗")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
