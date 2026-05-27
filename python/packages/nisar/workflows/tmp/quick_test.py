#!/usr/bin/env python3
'''Quick test for apply_filter correctness'''
import numpy as np
import sys
import os
from scipy import ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rubbersheet import apply_filter

# Create simple test array
np.random.seed(42)
test_array = np.random.randn(100, 100)
test_array[::10, ::10] = np.nan  # Add some NaN values

print("Testing apply_filter function...")
print(f"Test array shape: {test_array.shape}")
print(f"NaN count: {np.count_nonzero(np.isnan(test_array))}")
print()

# Test configurations
configs = [
    (3, 'mean', 'both'),
    (5, 'mean', 'both'),
    (11, 'median', 'both'),
    (5, 'mean', 'azimuth'),
    (5, 'mean', 'range'),
]

for window_size, filter_type, axis in configs:
    # Our implementation
    result = apply_filter(test_array, window_size, filter_type=filter_type, axis=axis)

    # Reference implementation
    ws_az = window_size if axis in ['both', 'azimuth'] else 1
    ws_rg = window_size if axis in ['both', 'range'] else 1

    if filter_type == 'mean':
        def ref_func(values):
            valid = values[np.isfinite(values)]
            return np.mean(valid) if len(valid) > 0 else np.nan
    else:
        def ref_func(values):
            valid = values[np.isfinite(values)]
            return np.median(valid) if len(valid) > 0 else np.nan

    reference = ndimage.generic_filter(
        test_array,
        ref_func,
        size=(ws_az, ws_rg),
        mode='constant',
        cval=np.nan
    )

    # Compare
    valid_mask = np.isfinite(result) & np.isfinite(reference)
    if np.any(valid_mask):
        max_diff = np.max(np.abs(result[valid_mask] - reference[valid_mask]))
        passed = max_diff < 1e-10
    else:
        max_diff = 0.0
        passed = True

    status = "✓" if passed else "✗"
    print(f"{status} {filter_type:6s} | {window_size}x{window_size} | axis={axis:8s} | max_diff={max_diff:.2e}")

print("\nAll tests completed!")
