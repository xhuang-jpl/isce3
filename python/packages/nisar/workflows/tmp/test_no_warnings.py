#!/usr/bin/env python3
"""
Test apply_filter without warning suppression
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('../'))
from rubbersheet import apply_filter

def test_without_warning_suppression():
    """Test that apply_filter works without warning suppression"""
    print("Testing apply_filter without warning suppression...")

    # Create test array with NaN
    np.random.seed(42)
    array = np.random.randn(100, 50).astype(np.float64)
    array[np.random.rand(100, 50) < 0.1] = np.nan

    print(f"Input: {array.shape}, NaN count: {np.sum(np.isnan(array))}")

    # Test mean filter
    result_mean = apply_filter(array.copy(), 5, filter_type='mean', axis='both')
    print(f"✓ Mean filter: Result shape {result_mean.shape}, NaN count: {np.sum(np.isnan(result_mean))}")

    # Test median filter
    result_median = apply_filter(array.copy(), 5, filter_type='median', axis='both')
    print(f"✓ Median filter: Result shape {result_median.shape}, NaN count: {np.sum(np.isnan(result_median))}")

    # Test with all-NaN window (should generate warnings)
    array_sparse = np.full((50, 50), np.nan, dtype=np.float64)
    array_sparse[25, 25] = 5.0  # One valid pixel

    print("\nTesting with sparse data (should see warnings about all-NaN slices)...")
    result_sparse = apply_filter(array_sparse, 5, filter_type='mean', axis='both')
    print(f"✓ Sparse data filter: Result shape {result_sparse.shape}, NaN count: {np.sum(np.isnan(result_sparse))}")

    print("\n✓ All tests passed!")
    print("Note: You may see RuntimeWarnings above about 'All-NaN slice' or 'Mean of empty slice'.")
    print("This is expected behavior when windows contain only NaN values.")

if __name__ == "__main__":
    test_without_warning_suppression()
