#!/usr/bin/env python3
"""
Demo script showing apply_filter function with different axes
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rubbersheet import apply_filter


def print_array(title, array):
    """Pretty print array with title"""
    print(f"\n{title}")
    print("=" * 50)
    print(array)


def demo_azimuth_filtering():
    """Demo filtering along azimuth axis"""
    print("\n" + "+" * 70)
    print("DEMO 1: Filtering along AZIMUTH axis (vertical)")
    print("+" * 70)

    # Create test array with pattern
    array = np.array([
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0],
        [4.0, 40.0, 400.0],
        [5.0, 50.0, 500.0]
    ])

    print_array("Original array:", array)

    result_mean = apply_filter(array, window_size=3, filter_type='mean', axis='azimuth')
    print_array("After MEAN filter (window=3, axis='azimuth'):", result_mean)

    result_median = apply_filter(array, window_size=3, filter_type='median', axis='azimuth')
    print_array("After MEDIAN filter (window=3, axis='azimuth'):", result_median)


def demo_range_filtering():
    """Demo filtering along range axis"""
    print("\n" + "+" * 70)
    print("DEMO 2: Filtering along RANGE axis (horizontal)")
    print("+" * 70)

    # Create test array
    array = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [10.0, 20.0, 30.0, 40.0, 50.0],
        [100.0, 200.0, 300.0, 400.0, 500.0]
    ])

    print_array("Original array:", array)

    result_mean = apply_filter(array, window_size=3, filter_type='mean', axis='range')
    print_array("After MEAN filter (window=3, axis='range'):", result_mean)

    result_median = apply_filter(array, window_size=3, filter_type='median', axis='range')
    print_array("After MEDIAN filter (window=3, axis='range'):", result_median)


def demo_both_axes_filtering():
    """Demo filtering along both axes"""
    print("\n" + "+" * 70)
    print("DEMO 3: Filtering along BOTH axes (2D)")
    print("+" * 70)

    # Create test array with smooth gradient
    array = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0],
        [3.0, 4.0, 5.0, 6.0, 7.0],
        [4.0, 5.0, 6.0, 7.0, 8.0],
        [5.0, 6.0, 7.0, 8.0, 9.0]
    ])

    print_array("Original array:", array)

    result_mean = apply_filter(array, window_size=3, filter_type='mean', axis='both')
    print_array("After MEAN filter (window=3, axis='both'):", result_mean)


def demo_both_axes_different_sizes():
    """Demo filtering with different window sizes"""
    print("\n" + "+" * 70)
    print("DEMO 4: Filtering with DIFFERENT window sizes")
    print("+" * 70)

    array = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0],
        [3.0, 4.0, 5.0, 6.0, 7.0],
        [4.0, 5.0, 6.0, 7.0, 8.0],
        [5.0, 6.0, 7.0, 8.0, 9.0]
    ])

    print_array("Original array:", array)

    result = apply_filter(array, window_size=(5, 3), filter_type='mean', axis='both')
    print_array("After MEAN filter (window=(5,3), axis='both'):", result)
    print("\nNote: Window is 5 in azimuth (vertical), 3 in range (horizontal)")


def demo_outlier_suppression():
    """Demo median filter for outlier suppression"""
    print("\n" + "+" * 70)
    print("DEMO 5: Outlier suppression with MEDIAN filter")
    print("+" * 70)

    # Create array with outliers
    array = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 100.0, 1.0, 1.0],  # outlier
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 200.0, 1.0, 1.0, 1.0],  # another outlier
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ])

    print_array("Original array (with outliers at [1,2]=100 and [3,1]=200):", array)

    result_mean = apply_filter(array, window_size=3, filter_type='mean', axis='both')
    print_array("After MEAN filter (window=3, axis='both'):", result_mean)
    print("Note: Mean is affected by outliers")

    result_median = apply_filter(array, window_size=3, filter_type='median', axis='both')
    print_array("After MEDIAN filter (window=3, axis='both'):", result_median)
    print("Note: Median effectively suppresses outliers")


def demo_nan_handling():
    """Demo NaN handling"""
    print("\n" + "+" * 70)
    print("DEMO 6: NaN handling")
    print("+" * 70)

    array = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ])

    print_array("Original array (with NaN at [1,1]):", array)

    result = apply_filter(array, window_size=3, filter_type='mean', axis='azimuth')
    print_array("After MEAN filter (window=3, axis='azimuth'):", result)
    print("Note: NaN values are ignored in the computation")


def main():
    """Run all demos"""
    print("\n" + "#" * 70)
    print("# apply_filter() Function Demonstration")
    print("#" * 70)
    print("\nThis demo shows the apply_filter function with different:")
    print("  - axes: 'azimuth', 'range', 'both'")
    print("  - filter types: 'mean', 'median'")
    print("  - window sizes: single value or tuple (azimuth, range)")

    demo_azimuth_filtering()
    demo_range_filtering()
    demo_both_axes_filtering()
    demo_both_axes_different_sizes()
    demo_outlier_suppression()
    demo_nan_handling()

    print("\n" + "#" * 70)
    print("# Demo complete!")
    print("#" * 70)


if __name__ == '__main__':
    # Set numpy print options for better display
    np.set_printoptions(precision=2, suppress=True, linewidth=100)
    main()
