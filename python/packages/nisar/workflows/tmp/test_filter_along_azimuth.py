'''
Unit tests for apply_filter function
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from rubbersheet import apply_filter


class TestApplyFilter(unittest.TestCase):
    '''Test suite for apply_filter function'''

    def test_mean_filter_basic(self):
        '''Test basic mean filtering along azimuth'''
        # Create simple test array
        array = np.array([
            [1.0, 5.0],
            [2.0, 6.0],
            [3.0, 7.0],
            [4.0, 8.0],
            [5.0, 9.0]
        ])

        result = apply_filter(array, window_size=3, filter_type='mean')

        # Check first column
        # Row 0: mean([1, 2]) = 1.5 (edge, only includes row 0, 1)
        # Row 1: mean([1, 2, 3]) = 2.0
        # Row 2: mean([2, 3, 4]) = 3.0
        # Row 3: mean([3, 4, 5]) = 4.0
        # Row 4: mean([4, 5]) = 4.5 (edge, only includes row 3, 4)

        expected_col0 = np.array([1.5, 2.0, 3.0, 4.0, 4.5])
        np.testing.assert_array_almost_equal(result[:, 0], expected_col0)

        # Check second column
        expected_col1 = np.array([5.5, 6.0, 7.0, 8.0, 8.5])
        np.testing.assert_array_almost_equal(result[:, 1], expected_col1)

    def test_median_filter_basic(self):
        '''Test basic median filtering along azimuth'''
        array = np.array([
            [1.0, 5.0],
            [2.0, 6.0],
            [3.0, 7.0],
            [4.0, 8.0],
            [5.0, 9.0]
        ])

        result = apply_filter(array, window_size=3, filter_type='median')

        # Check first column
        # Row 0: median([1, 2]) = 1.5
        # Row 1: median([1, 2, 3]) = 2.0
        # Row 2: median([2, 3, 4]) = 3.0
        # Row 3: median([3, 4, 5]) = 4.0
        # Row 4: median([4, 5]) = 4.5

        expected_col0 = np.array([1.5, 2.0, 3.0, 4.0, 4.5])
        np.testing.assert_array_almost_equal(result[:, 0], expected_col0)

    def test_nan_handling_mean(self):
        '''Test that NaN values are ignored in mean calculation'''
        array = np.array([
            [1.0, 5.0],
            [np.nan, 6.0],
            [3.0, np.nan],
            [4.0, 8.0],
            [5.0, 9.0]
        ])

        result = apply_filter(array, window_size=3, filter_type='mean')

        # Row 1, col 0: mean([1, 3]) = 2.0 (ignoring NaN)
        self.assertAlmostEqual(result[1, 0], 2.0)

        # Row 2, col 1: mean([6, 8]) = 7.0 (ignoring NaN)
        self.assertAlmostEqual(result[2, 1], 7.0)

    def test_inf_handling_mean(self):
        '''Test that Inf values are ignored in mean calculation'''
        array = np.array([
            [1.0, 5.0],
            [np.inf, 6.0],
            [3.0, -np.inf],
            [4.0, 8.0],
            [5.0, 9.0]
        ])

        result = apply_filter(array, window_size=3, filter_type='mean')

        # Row 1, col 0: mean([1, 3]) = 2.0 (ignoring Inf)
        self.assertAlmostEqual(result[1, 0], 2.0)

        # Row 2, col 1: mean([6, 8]) = 7.0 (ignoring -Inf)
        self.assertAlmostEqual(result[2, 1], 7.0)

    def test_nan_handling_median(self):
        '''Test that NaN values are ignored in median calculation'''
        array = np.array([
            [1.0, 5.0],
            [np.nan, 6.0],
            [3.0, np.nan],
            [4.0, 8.0],
            [5.0, 9.0]
        ])

        result = apply_filter(array, window_size=3, filter_type='median')

        # Row 1, col 0: median([1, 3]) = 2.0 (ignoring NaN)
        self.assertAlmostEqual(result[1, 0], 2.0)

        # Row 2, col 1: median([6, 8]) = 7.0 (ignoring NaN)
        self.assertAlmostEqual(result[2, 1], 7.0)

    def test_window_size_1(self):
        '''Test trivial case with window_size=1'''
        array = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])

        result = apply_filter(array, window_size=1, filter_type='mean')

        # Should return a copy of the input
        np.testing.assert_array_equal(result, array)

    def test_all_nan_array(self):
        '''Test with array containing all NaN values'''
        array = np.full((5, 3), np.nan)

        result = apply_filter(array, window_size=3, filter_type='mean')

        # Should return array of NaNs
        self.assertTrue(np.all(np.isnan(result)))

    def test_column_with_all_nan(self):
        '''Test with one column having all NaN values'''
        array = np.array([
            [1.0, np.nan],
            [2.0, np.nan],
            [3.0, np.nan]
        ])

        result = apply_filter(array, window_size=3, filter_type='mean')

        # First column should be filtered normally
        self.assertFalse(np.any(np.isnan(result[:, 0])))

        # Second column should remain all NaN
        self.assertTrue(np.all(np.isnan(result[:, 1])))

    def test_large_window_size(self):
        '''Test with window_size larger than array height'''
        array = np.array([
            [1.0, 5.0],
            [2.0, 6.0],
            [3.0, 7.0]
        ])

        result = apply_filter(array, window_size=5, filter_type='mean')

        # All rows should average all values in the column
        expected_col0 = np.array([2.0, 2.0, 2.0])
        expected_col1 = np.array([6.0, 6.0, 6.0])

        np.testing.assert_array_almost_equal(result[:, 0], expected_col0)
        np.testing.assert_array_almost_equal(result[:, 1], expected_col1)

    def test_filter_only_along_azimuth(self):
        '''Test that filtering only occurs along azimuth, not range'''
        # Create array with distinct patterns in azimuth vs range
        array = np.array([
            [1.0, 100.0, 1.0],
            [2.0, 200.0, 2.0],
            [3.0, 300.0, 3.0]
        ])

        result = apply_filter(array, window_size=3, filter_type='mean')

        # Each column should be filtered independently
        # Middle column should not affect side columns
        self.assertLess(result[1, 0], 50)  # Should be ~2, not affected by 200
        self.assertLess(result[1, 2], 50)  # Should be ~2, not affected by 200

    def test_invalid_window_size_even(self):
        '''Test that even window_size raises ValueError'''
        array = np.array([[1.0, 2.0], [3.0, 4.0]])

        with self.assertRaises(ValueError) as context:
            apply_filter(array, window_size=4, filter_type='mean')

        self.assertIn('must be odd', str(context.exception))

    def test_invalid_window_size_zero(self):
        '''Test that window_size=0 raises ValueError'''
        array = np.array([[1.0, 2.0], [3.0, 4.0]])

        with self.assertRaises(ValueError) as context:
            apply_filter(array, window_size=0, filter_type='mean')

        self.assertIn('must be >= 1', str(context.exception))

    def test_invalid_window_size_negative(self):
        '''Test that negative window_size raises ValueError'''
        array = np.array([[1.0, 2.0], [3.0, 4.0]])

        with self.assertRaises(ValueError) as context:
            apply_filter(array, window_size=-3, filter_type='mean')

        self.assertIn('must be >= 1', str(context.exception))

    def test_invalid_filter_type(self):
        '''Test that invalid filter_type raises ValueError'''
        array = np.array([[1.0, 2.0], [3.0, 4.0]])

        with self.assertRaises(ValueError) as context:
            apply_filter(array, window_size=3, filter_type='gaussian')

        self.assertIn("must be 'mean' or 'median'", str(context.exception))

    def test_mixed_nan_inf(self):
        '''Test with mix of NaN, Inf, and valid values'''
        array = np.array([
            [1.0, 5.0],
            [np.nan, 6.0],
            [np.inf, 7.0],
            [3.0, -np.inf],
            [4.0, 8.0]
        ])

        result = apply_filter(array, window_size=3, filter_type='mean')

        # Check that function completes without error
        self.assertEqual(result.shape, array.shape)

        # Row 3, col 0: should ignore Inf and compute mean of valid values
        self.assertTrue(np.isfinite(result[3, 0]))

    def test_single_row_array(self):
        '''Test with single-row array'''
        array = np.array([[1.0, 2.0, 3.0]])

        result = apply_filter(array, window_size=1, filter_type='mean')

        np.testing.assert_array_equal(result, array)

    def test_single_column_array(self):
        '''Test with single-column array'''
        array = np.array([
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0]
        ])

        result = apply_filter(array, window_size=3, filter_type='mean')

        expected = np.array([[1.5], [2.0], [3.0], [4.0], [4.5]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_median_with_even_number_valid_values(self):
        '''Test median calculation with even number of valid values'''
        array = np.array([
            [1.0],
            [np.nan],
            [2.0],
            [np.nan],
            [3.0]
        ])

        result = apply_filter(array, window_size=5, filter_type='median')

        # Row 2 should have median of [1, 2, 3] = 2.0
        self.assertAlmostEqual(result[2, 0], 2.0)

    def test_preserves_dtype(self):
        '''Test that output preserves float type'''
        array = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], dtype=np.float32)

        result = apply_filter(array, window_size=3, filter_type='mean')

        # Should be floating point
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_larger_window_size(self):
        '''Test with larger window size (7)'''
        array = np.arange(20).reshape(10, 2).astype(float)

        result = apply_filter(array, window_size=7, filter_type='mean')

        # Verify shape is preserved
        self.assertEqual(result.shape, array.shape)

        # Verify no NaNs in result (all values are valid)
        self.assertFalse(np.any(np.isnan(result)))

    def test_filter_range_axis(self):
        '''Test filtering along range axis only'''
        array = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0]
        ])

        result = apply_filter(array, window_size=3, filter_type='mean', axis='range')

        # Check middle column of first row - should be average of cols 0, 1, 2
        expected_middle = np.mean(array[0, 0:3])
        self.assertAlmostEqual(result[0, 1], expected_middle)

        # Check that filtering is only along range (each row independent)
        # Row 0, col 1 should not be affected by row 1 values
        self.assertLess(result[0, 1], 5.0)  # Should be ~2, not affected by row 1

    def test_filter_both_axes_single_window(self):
        '''Test filtering along both axes with single window size'''
        array = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0]
        ])

        result = apply_filter(array, window_size=3, filter_type='mean', axis='both')

        # Check center pixel - should be average of 3x3 window
        expected_center = np.mean(array[1:4, 1:4])
        self.assertAlmostEqual(result[2, 2], expected_center)

        # Verify shape is preserved
        self.assertEqual(result.shape, array.shape)

    def test_filter_both_axes_tuple_window(self):
        '''Test filtering with different window sizes for azimuth and range'''
        array = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0]
        ])

        # Apply filter with window size (5, 3) - 5 in azimuth, 3 in range
        result = apply_filter(array, window_size=(5, 3), filter_type='mean', axis='both')

        # Check center pixel - should be average of 5x3 window (entire column, 3 in range)
        expected_center = np.mean(array[:, 1:4])
        self.assertAlmostEqual(result[2, 2], expected_center)

    def test_median_filter_both_axes(self):
        '''Test median filtering in both directions'''
        # Create data with outlier
        array = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 100.0, 1.0, 1.0],  # outlier
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ])

        result = apply_filter(array, window_size=3, filter_type='median', axis='both')

        # Median should suppress the outlier
        self.assertLess(result[2, 2], 50.0)

    def test_invalid_axis(self):
        '''Test that invalid axis raises ValueError'''
        array = np.array([[1.0, 2.0], [3.0, 4.0]])

        with self.assertRaises(ValueError) as context:
            apply_filter(array, window_size=3, axis='diagonal')

        self.assertIn("axis must be", str(context.exception))

    def test_backward_compatibility_default_axis(self):
        '''Test that default axis='azimuth' maintains backward compatibility'''
        array = np.array([
            [1.0, 5.0],
            [2.0, 6.0],
            [3.0, 7.0],
            [4.0, 8.0],
            [5.0, 9.0]
        ])

        # Old behavior: default should filter along azimuth
        result_default = apply_filter(array, window_size=3, filter_type='mean')
        result_explicit = apply_filter(array, window_size=3, filter_type='mean', axis='azimuth')

        np.testing.assert_array_equal(result_default, result_explicit)

    def test_range_axis_with_nan(self):
        '''Test range axis filtering with NaN values'''
        array = np.array([
            [1.0, np.nan, 3.0, 4.0, 5.0],
            [6.0, 7.0, np.nan, 9.0, 10.0]
        ])

        result = apply_filter(array, window_size=3, filter_type='mean', axis='range')

        # Check that NaN is handled (ignored in computation)
        self.assertTrue(np.isfinite(result[0, 1]))
        self.assertTrue(np.isfinite(result[1, 2]))

    def test_both_axes_trivial_case(self):
        '''Test both axes with window_size=1'''
        array = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = apply_filter(array, window_size=1, filter_type='mean', axis='both')

        # Should return a copy
        np.testing.assert_array_equal(result, array)


if __name__ == '__main__':
    unittest.main()
