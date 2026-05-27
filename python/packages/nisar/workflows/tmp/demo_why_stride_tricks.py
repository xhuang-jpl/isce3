#!/usr/bin/env python3
"""
Demonstrate WHY we need stride tricks for spatial filtering
"""
import numpy as np

print("="*70)
print("Why Stride Tricks Are Necessary for Spatial Filtering")
print("="*70)

# Create simple test array
array = np.arange(25, dtype=float).reshape(5, 5)
print("\nOriginal Array (5×5):")
print(array)

print("\n" + "="*70)
print("WRONG: Using np.nanmean directly")
print("="*70)
result_wrong = np.nanmean(array)
print(f"\nResult: {result_wrong}")
print("Shape: scalar (just one number!)")
print("❌ This is WRONG - we lost all spatial information!")

print("\n" + "="*70)
print("CORRECT: Using stride tricks + np.nanmean")
print("="*70)

window_size = 3
half_window = window_size // 2

# Pad array
padded = np.pad(array, ((half_window, half_window), (half_window, half_window)),
                mode='constant', constant_values=np.nan)
print(f"\nPadded array (7×7):")
print(padded)

# Create sliding windows with stride tricks
nrows, ncols = array.shape
shape = (nrows, ncols, window_size, window_size)
strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

print(f"\nWindows shape: {windows.shape}")
print("  Meaning: For each of the 5×5 output pixels, we have a 3×3 neighborhood")

# Show a few example windows
print("\n--- Example: Window at position (0, 0) ---")
print("This is the 3×3 neighborhood around pixel [0,0]:")
print(windows[0, 0, :, :])
print(f"Mean of this window: {np.nanmean(windows[0, 0, :, :]):.2f}")

print("\n--- Example: Window at position (2, 2) (center) ---")
print("This is the 3×3 neighborhood around pixel [2,2]:")
print(windows[2, 2, :, :])
print(f"Mean of this window: {np.nanmean(windows[2, 2, :, :]):.2f}")

# Compute filtered result
result_correct = np.nanmean(windows, axis=(2, 3))

print("\n" + "="*70)
print("Final Filtered Result (5×5):")
print("="*70)
print(result_correct)
print(f"\nShape: {result_correct.shape}")
print("✓ This is CORRECT - spatial structure preserved!")

print("\n" + "="*70)
print("Key Insight")
print("="*70)
print("""
Without stride tricks:
  - We'd only get ONE number (global mean/median)
  - Loses all spatial information

With stride tricks:
  - Each output pixel gets its own local neighborhood
  - Output has same shape as input
  - This is what makes it a SPATIAL FILTER
""")
