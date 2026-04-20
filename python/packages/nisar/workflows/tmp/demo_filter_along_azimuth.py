'''
Demo script for filter_along_azimuth function
Shows basic usage and performance
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from rubbersheet import filter_along_azimuth


def demo_basic_usage():
    '''Demonstrate basic usage'''
    print("=" * 60)
    print("Basic Usage Demo")
    print("=" * 60)

    # Create sample data with some NaN and Inf values
    array = np.array([
        [1.0, 5.0, 9.0],
        [2.0, np.nan, 10.0],
        [3.0, 7.0, np.inf],
        [4.0, 8.0, 12.0],
        [5.0, 9.0, 13.0]
    ])

    print("\nOriginal array:")
    print(array)

    # Apply mean filter
    result_mean = filter_along_azimuth(array, window_size=3, filter_type='mean')
    print("\nAfter mean filter (window_size=3):")
    print(result_mean)

    # Apply median filter
    result_median = filter_along_azimuth(array, window_size=3, filter_type='median')
    print("\nAfter median filter (window_size=3):")
    print(result_median)


def demo_performance():
    '''Demonstrate performance on larger arrays'''
    print("\n" + "=" * 60)
    print("Performance Demo")
    print("=" * 60)

    sizes = [(100, 100), (500, 500), (1000, 1000)]
    window_size = 5

    for nrows, ncols in sizes:
        # Create random array with some NaN values
        array = np.random.randn(nrows, ncols)
        # Add ~10% NaN values randomly
        nan_mask = np.random.rand(nrows, ncols) < 0.1
        array[nan_mask] = np.nan

        # Time mean filter
        start = time.time()
        _ = filter_along_azimuth(array, window_size=window_size, filter_type='mean')
        elapsed_mean = time.time() - start

        # Time median filter
        start = time.time()
        _ = filter_along_azimuth(array, window_size=window_size, filter_type='median')
        elapsed_median = time.time() - start

        print(f"\nArray size: {nrows} x {ncols}, window_size={window_size}")
        print(f"  Mean filter:   {elapsed_mean:.4f} seconds")
        print(f"  Median filter: {elapsed_median:.4f} seconds")


def demo_nan_handling():
    '''Demonstrate NaN and Inf handling'''
    print("\n" + "=" * 60)
    print("NaN/Inf Handling Demo")
    print("=" * 60)

    array = np.array([
        [1.0, 1.0],
        [np.nan, 2.0],
        [3.0, np.inf],
        [4.0, 4.0],
        [5.0, 5.0]
    ])

    print("\nOriginal array (note NaN and Inf):")
    print(array)

    result = filter_along_azimuth(array, window_size=3, filter_type='mean')
    print("\nFiltered array (NaN and Inf are ignored):")
    print(result)
    print("\nNotice:")
    print("- Row 1, Col 0: mean([1, 3]) = 2.0 (NaN ignored)")
    print("- Row 2, Col 1: mean([2, 4]) = 3.0 (Inf ignored)")


if __name__ == '__main__':
    demo_basic_usage()
    demo_nan_handling()
    demo_performance()

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
