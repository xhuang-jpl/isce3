#!/usr/bin/env python3
"""
Comprehensive memory performance benchmark for the sliding window filter optimization
"""
import sys
import os
import numpy as np
import tracemalloc
import psutil
import gc
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath('../'))
from rubbersheet import apply_filter


def get_process_memory_mb():
    """Get current process RSS memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


def benchmark_original_reshape(array, window_size_az, window_size_rg):
    """
    Benchmark the ORIGINAL implementation with reshape (memory intensive)
    This will likely fail or use massive memory on large arrays
    """
    print("\n" + "="*70)
    print("ORIGINAL APPROACH: With reshape (memory intensive)")
    print("="*70)

    nrows, ncols = array.shape
    half_window_az = window_size_az // 2
    half_window_rg = window_size_rg // 2

    # Track memory
    mem_start = get_process_memory_mb()
    tracemalloc.start()
    snapshot_start = tracemalloc.take_snapshot()

    print(f"Initial memory: {mem_start:.2f} MB")

    try:
        # Pad the array
        padded = np.pad(array,
                        ((half_window_az, half_window_az), (half_window_rg, half_window_rg)),
                        mode='constant', constant_values=np.nan)

        mem_after_pad = get_process_memory_mb()
        print(f"After padding: {mem_after_pad:.2f} MB (+{mem_after_pad - mem_start:.2f} MB)")

        # Create stride view
        shape = (nrows, ncols, window_size_az, window_size_rg)
        strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

        mem_after_stride = get_process_memory_mb()
        print(f"After stride tricks: {mem_after_stride:.2f} MB (+{mem_after_stride - mem_after_pad:.2f} MB)")
        print(f"  Windows shape: {windows.shape}")
        print(f"  Theoretical size if materialized: {windows.nbytes / 1024**3:.2f} GB")

        # ORIGINAL: Reshape (this creates a copy!)
        print("\nAttempting reshape (THIS IS THE PROBLEM)...")
        time_reshape_start = time.time()

        windows_flat = windows.reshape(nrows, ncols, -1)

        time_reshape = time.time() - time_reshape_start
        mem_after_reshape = get_process_memory_mb()

        print(f"After reshape: {mem_after_reshape:.2f} MB (+{mem_after_reshape - mem_after_stride:.2f} MB)")
        print(f"  Reshape time: {time_reshape:.3f} seconds")
        print(f"  Created copy: {not np.shares_memory(windows, windows_flat)}")

        # Apply filter
        print("\nApplying nanmean...")
        time_filter_start = time.time()
        result = np.nanmean(windows_flat, axis=2)
        time_filter = time.time() - time_filter_start

        mem_final = get_process_memory_mb()
        print(f"After filtering: {mem_final:.2f} MB")
        print(f"  Filter time: {time_filter:.3f} seconds")

        # Get peak memory
        snapshot_end = tracemalloc.take_snapshot()
        stats = snapshot_end.compare_to(snapshot_start, 'lineno')
        total_allocated = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\n>>> MEMORY SUMMARY (Original) <<<")
        print(f"  Total RSS increase: {mem_final - mem_start:.2f} MB")
        print(f"  Total allocated: {total_allocated / 1024**2:.2f} MB")
        print(f"  Peak traced memory: {peak / 1024**2:.2f} MB")
        print(f"  Total time: {time_reshape + time_filter:.3f} seconds")

        return {
            'success': True,
            'mem_increase': mem_final - mem_start,
            'mem_peak': peak / 1024**2,
            'time_total': time_reshape + time_filter,
            'result': result
        }

    except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
        tracemalloc.stop()
        print(f"\n✗ MEMORY ERROR: {e}")
        print("Original approach FAILED due to insufficient memory!")
        return {
            'success': False,
            'error': str(e)
        }


def benchmark_optimized_no_reshape(array, window_size_az, window_size_rg):
    """
    Benchmark the OPTIMIZED implementation without reshape (memory efficient)
    """
    print("\n" + "="*70)
    print("OPTIMIZED APPROACH: Without reshape (memory efficient)")
    print("="*70)

    nrows, ncols = array.shape
    half_window_az = window_size_az // 2
    half_window_rg = window_size_rg // 2

    # Track memory
    mem_start = get_process_memory_mb()
    tracemalloc.start()
    snapshot_start = tracemalloc.take_snapshot()

    print(f"Initial memory: {mem_start:.2f} MB")

    # Pad the array
    padded = np.pad(array,
                    ((half_window_az, half_window_az), (half_window_rg, half_window_rg)),
                    mode='constant', constant_values=np.nan)

    mem_after_pad = get_process_memory_mb()
    print(f"After padding: {mem_after_pad:.2f} MB (+{mem_after_pad - mem_start:.2f} MB)")

    # Create stride view
    shape = (nrows, ncols, window_size_az, window_size_rg)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    mem_after_stride = get_process_memory_mb()
    print(f"After stride tricks: {mem_after_stride:.2f} MB (+{mem_after_stride - mem_after_pad:.2f} MB)")
    print(f"  Windows shape: {windows.shape}")
    print(f"  Theoretical size if materialized: {windows.nbytes / 1024**3:.2f} GB")

    # OPTIMIZED: Direct axis computation (no reshape!)
    print("\nApplying nanmean with axis=(2,3) - NO RESHAPE...")
    time_filter_start = time.time()

    result = np.nanmean(windows, axis=(2, 3))

    time_filter = time.time() - time_filter_start
    mem_final = get_process_memory_mb()

    print(f"After filtering: {mem_final:.2f} MB")
    print(f"  Filter time: {time_filter:.3f} seconds")

    # Get peak memory
    snapshot_end = tracemalloc.take_snapshot()
    stats = snapshot_end.compare_to(snapshot_start, 'lineno')
    total_allocated = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n>>> MEMORY SUMMARY (Optimized) <<<")
    print(f"  Total RSS increase: {mem_final - mem_start:.2f} MB")
    print(f"  Total allocated: {total_allocated / 1024**2:.2f} MB")
    print(f"  Peak traced memory: {peak / 1024**2:.2f} MB")
    print(f"  Total time: {time_filter:.3f} seconds")

    return {
        'success': True,
        'mem_increase': mem_final - mem_start,
        'mem_peak': peak / 1024**2,
        'time_total': time_filter,
        'result': result
    }


def benchmark_with_apply_filter(array, window_size):
    """
    Benchmark using the actual apply_filter function from rubbersheet
    """
    print("\n" + "="*70)
    print("RUBBERSHEET apply_filter() - PRODUCTION CODE")
    print("="*70)

    mem_start = get_process_memory_mb()
    tracemalloc.start()

    print(f"Initial memory: {mem_start:.2f} MB")
    print(f"Calling apply_filter(array, {window_size}, 'mean', 'both')...")

    time_start = time.time()
    result = apply_filter(array, window_size, filter_type='mean', axis='both')
    time_elapsed = time.time() - time_start

    mem_final = get_process_memory_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"After filtering: {mem_final:.2f} MB")
    print(f"  Time: {time_elapsed:.3f} seconds")
    print(f"  Result shape: {result.shape}")

    print(f"\n>>> MEMORY SUMMARY (Production) <<<")
    print(f"  Total RSS increase: {mem_final - mem_start:.2f} MB")
    print(f"  Peak traced memory: {peak / 1024**2:.2f} MB")
    print(f"  Total time: {time_elapsed:.3f} seconds")

    return {
        'success': True,
        'mem_increase': mem_final - mem_start,
        'mem_peak': peak / 1024**2,
        'time_total': time_elapsed,
        'result': result
    }


def run_memory_benchmark():
    """Run comprehensive memory benchmarks"""
    print("="*70)
    print("MEMORY PERFORMANCE BENCHMARK")
    print("="*70)
    print(f"NumPy version: {np.__version__}")
    print(f"Python version: {sys.version}")

    # Test configurations
    test_cases = [
        (1000, 500, 11, "Small: 1000×500, window 11×11"),
        (2000, 1000, 21, "Medium: 2000×1000, window 21×21"),
        (3000, 1500, 31, "Large: 3000×1500, window 31×31"),
    ]

    results_summary = []

    for nrows, ncols, window_size, description in test_cases:
        print("\n" + "="*70)
        print(f"TEST CASE: {description}")
        print("="*70)

        # Create test array
        np.random.seed(42)
        array = np.random.randn(nrows, ncols).astype(np.float64)
        array[np.random.rand(nrows, ncols) < 0.1] = np.nan

        array_size_mb = array.nbytes / 1024**2
        print(f"Array size: {array_size_mb:.2f} MB")
        print(f"Window size: {window_size}×{window_size}")

        theoretical_windows_size = (nrows * ncols * window_size * window_size * 8) / 1024**3
        print(f"Theoretical windows size: {theoretical_windows_size:.2f} GB")

        # Force garbage collection before each test
        gc.collect()
        time.sleep(1)

        # Test 1: Original approach (with reshape)
        result_original = benchmark_original_reshape(array.copy(), window_size, window_size)

        gc.collect()
        time.sleep(1)

        # Test 2: Optimized approach (no reshape)
        result_optimized = benchmark_optimized_no_reshape(array.copy(), window_size, window_size)

        gc.collect()
        time.sleep(1)

        # Test 3: Production apply_filter
        result_production = benchmark_with_apply_filter(array.copy(), window_size)

        # Compare results
        if result_original['success'] and result_optimized['success']:
            print("\n" + "="*70)
            print("COMPARISON")
            print("="*70)

            mem_savings = result_original['mem_increase'] - result_optimized['mem_increase']
            mem_savings_pct = (mem_savings / result_original['mem_increase']) * 100
            time_speedup = result_original['time_total'] / result_optimized['time_total']

            print(f"Memory savings: {mem_savings:.2f} MB ({mem_savings_pct:.1f}%)")
            print(f"Time speedup: {time_speedup:.2f}x")

            # Verify correctness
            max_diff = np.nanmax(np.abs(result_original['result'] - result_optimized['result']))
            print(f"Max difference: {max_diff:.2e}")
            print(f"Results identical: {np.allclose(result_original['result'], result_optimized['result'], equal_nan=True)}")

            results_summary.append({
                'description': description,
                'array_size_mb': array_size_mb,
                'original_mem': result_original['mem_increase'],
                'optimized_mem': result_optimized['mem_increase'],
                'production_mem': result_production['mem_increase'],
                'mem_savings_mb': mem_savings,
                'mem_savings_pct': mem_savings_pct,
                'speedup': time_speedup,
                'original_time': result_original['time_total'],
                'optimized_time': result_optimized['time_total'],
                'production_time': result_production['time_total'],
            })
        elif not result_original['success']:
            print("\n" + "="*70)
            print("RESULT: Original approach FAILED")
            print("="*70)
            print(f"Original: FAILED (MemoryError)")
            print(f"Optimized: SUCCESS ({result_optimized['mem_increase']:.2f} MB, {result_optimized['time_total']:.3f}s)")
            print(f"Production: SUCCESS ({result_production['mem_increase']:.2f} MB, {result_production['time_total']:.3f}s)")

            results_summary.append({
                'description': description,
                'array_size_mb': array_size_mb,
                'original_mem': 'FAILED',
                'optimized_mem': result_optimized['mem_increase'],
                'production_mem': result_production['mem_increase'],
                'mem_savings_mb': 'N/A',
                'mem_savings_pct': 'N/A',
                'speedup': 'N/A',
                'optimized_time': result_optimized['time_total'],
                'production_time': result_production['time_total'],
            })

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Test Case':<30} {'Array Size':<12} {'Original Mem':<15} {'Optimized Mem':<15} {'Savings':<15} {'Speedup':<10}")
    print("-"*70)
    for r in results_summary:
        original_str = f"{r['original_mem']:.1f} MB" if r['original_mem'] != 'FAILED' else 'FAILED'
        optimized_str = f"{r['optimized_mem']:.1f} MB"
        savings_str = f"{r['mem_savings_mb']:.1f} MB" if r['mem_savings_mb'] != 'N/A' else 'N/A'
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] != 'N/A' else 'N/A'

        print(f"{r['description']:<30} {r['array_size_mb']:>10.1f} MB {original_str:>14} {optimized_str:>14} {savings_str:>14} {speedup_str:>9}")

    print("\n" + "="*70)
    print("✓ BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_memory_benchmark()
