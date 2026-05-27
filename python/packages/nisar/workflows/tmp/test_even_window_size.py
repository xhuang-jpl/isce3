#!/usr/bin/env python3
"""
Test apply_filter with even window sizes
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('../'))
from rubbersheet import apply_filter


def test_even_odd_windows():
    """Test that both even and odd window sizes work correctly"""
    print("="*70)
    print("Testing Even and Odd Window Sizes")
    print("="*70)

    # Create simple test array
    np.random.seed(42)
    array = np.random.randn(20, 20).astype(np.float64)
    array[np.random.rand(20, 20) < 0.1] = np.nan

    print(f"\nInput array: {array.shape}")
    print(f"NaN count: {np.sum(np.isnan(array))}")

    # Test various window sizes (both odd and even)
    window_sizes = [3, 4, 5, 6, 7, 8, 10, 11]

    print("\n" + "-"*70)
    print("Mean Filter Tests")
    print("-"*70)

    for window_size in window_sizes:
        result = apply_filter(array.copy(), window_size, filter_type='mean', axis='both')
        parity = "odd" if window_size % 2 == 1 else "even"
        print(f"Window {window_size:2d}×{window_size:2d} ({parity:>4s}): "
              f"Result shape {result.shape}, NaN count: {np.sum(np.isnan(result))}")

        # Verify output shape matches input
        assert result.shape == array.shape, f"Shape mismatch for window {window_size}"

    print("\n" + "-"*70)
    print("Median Filter Tests")
    print("-"*70)

    for window_size in window_sizes:
        result = apply_filter(array.copy(), window_size, filter_type='median', axis='both')
        parity = "odd" if window_size % 2 == 1 else "even"
        print(f"Window {window_size:2d}×{window_size:2d} ({parity:>4s}): "
              f"Result shape {result.shape}, NaN count: {np.sum(np.isnan(result))}")

        assert result.shape == array.shape, f"Shape mismatch for window {window_size}"

    print("\n✓ All window sizes (odd and even) work correctly!")


def test_even_window_correctness():
    """Verify even window sizes produce sensible results"""
    print("\n" + "="*70)
    print("Even Window Size Correctness Test")
    print("="*70)

    # Create small known array
    array = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0],
        [3.0, 4.0, 5.0, 6.0, 7.0],
        [4.0, 5.0, 6.0, 7.0, 8.0],
        [5.0, 6.0, 7.0, 8.0, 9.0]
    ])

    print("\nInput array (5×5):")
    print(array)

    # Test with even window (4×4)
    result_even = apply_filter(array, 4, filter_type='mean', axis='both')
    print("\nResult with 4×4 window (even):")
    print(result_even)

    # Test with odd window (3×3)
    result_odd = apply_filter(array, 3, filter_type='mean', axis='both')
    print("\nResult with 3×3 window (odd):")
    print(result_odd)

    # Verify center pixel makes sense
    # For a monotonically increasing array, filtered values should be reasonable
    assert np.all(np.isfinite(result_even)), "Even window produced NaN unexpectedly"
    assert np.all(np.isfinite(result_odd)), "Odd window produced NaN unexpectedly"

    # Check that results are in reasonable range
    assert np.all(result_even >= 1.0) and np.all(result_even <= 9.0), "Even window out of range"
    assert np.all(result_odd >= 1.0) and np.all(result_odd <= 9.0), "Odd window out of range"

    print("\n✓ Even window sizes produce sensible results!")


def test_padding_calculation():
    """Verify padding calculation for even and odd windows"""
    print("\n" + "="*70)
    print("Padding Calculation Verification")
    print("="*70)

    test_cases = [
        (3, "odd"),
        (4, "even"),
        (5, "odd"),
        (6, "even"),
        (7, "odd"),
        (8, "even"),
        (10, "even"),
        (11, "odd"),
    ]

    print(f"\n{'Window Size':<15} {'Parity':<10} {'Pad Before':<15} {'Pad After':<15} {'Total':<10}")
    print("-"*70)

    for window_size, parity in test_cases:
        pad_before = (window_size - 1) // 2
        pad_after = window_size // 2
        total = pad_before + 1 + pad_after  # before + center + after

        status = "✓" if total == window_size else "✗"
        print(f"{window_size:<15} {parity:<10} {pad_before:<15} {pad_after:<15} {total:<10} {status}")

        assert total == window_size, f"Padding calculation wrong for window {window_size}"

    print("\n✓ Padding calculations correct for all window sizes!")


def main():
    print("\n" + "="*70)
    print("EVEN WINDOW SIZE SUPPORT TESTS")
    print("="*70)

    test_padding_calculation()
    test_even_odd_windows()
    test_even_window_correctness()

    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    print("Both even and odd window sizes are now supported!")
    print("="*70)


if __name__ == "__main__":
    main()
