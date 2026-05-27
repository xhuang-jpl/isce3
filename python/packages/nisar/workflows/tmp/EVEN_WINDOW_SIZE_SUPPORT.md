# Even Window Size Support

## Summary

The `apply_filter()` function now supports **both even and odd window sizes**.

Previously, the function enforced odd window sizes only (3, 5, 7, 9, 11, etc.).  
Now, it accepts any window size ≥ 1, including even sizes (4, 6, 8, 10, etc.).

---

## Changes Made

### File: `python/packages/nisar/workflows/rubbersheet.py`

#### 1. Removed Odd-Only Validation (Lines 980-984)

**Before:**
```python
# Validate window sizes
if window_size_az < 1:
    raise ValueError(f"window_size_azimuth must be >= 1, got {window_size_az}")
if window_size_rg < 1:
    raise ValueError(f"window_size_range must be >= 1, got {window_size_rg}")
if window_size_az % 2 == 0:
    raise ValueError(f"window_size_azimuth must be odd, got {window_size_az}")
if window_size_rg % 2 == 0:
    raise ValueError(f"window_size_range must be odd, got {window_size_rg}")
```

**After:**
```python
# Validate window sizes
if window_size_az < 1:
    raise ValueError(f"window_size_azimuth must be >= 1, got {window_size_az}")
if window_size_rg < 1:
    raise ValueError(f"window_size_range must be >= 1, got {window_size_rg}")
```

#### 2. Updated Padding Calculation (Lines 1000-1010)

**Before (symmetric padding - only works for odd):**
```python
half_window_az = window_size_az // 2
half_window_rg = window_size_rg // 2

padded = np.pad(array_clean,
                ((half_window_az, half_window_az), (half_window_rg, half_window_rg)),
                mode='constant', constant_values=np.nan)
```

**After (asymmetric padding - works for both):**
```python
# Calculate padding for both odd and even window sizes
# For odd windows (e.g., 5): pad_before=2, pad_after=2 → 2+1+2=5 ✓
# For even windows (e.g., 4): pad_before=1, pad_after=2 → 1+1+2=4 ✓
pad_before_az = (window_size_az - 1) // 2
pad_after_az = window_size_az // 2
pad_before_rg = (window_size_rg - 1) // 2
pad_after_rg = window_size_rg // 2

# Pad the array to handle edges (asymmetric for even window sizes)
padded = np.pad(array_clean,
                ((pad_before_az, pad_after_az), (pad_before_rg, pad_after_rg)),
                mode='constant', constant_values=np.nan)
```

---

## Padding Calculation Logic

### Why Asymmetric Padding for Even Windows?

For a window to contain exactly `N` pixels, with the "center" at the current pixel:

```
Total pixels in window = pad_before + 1 (current) + pad_after
```

For **odd windows** (e.g., 5×5):
- Symmetric padding works: `pad_before = pad_after = 2`
- Total: `2 + 1 + 2 = 5` ✓

For **even windows** (e.g., 4×4):
- Need asymmetric: `pad_before = 1, pad_after = 2`
- Total: `1 + 1 + 2 = 4` ✓

### Formula

```python
pad_before = (window_size - 1) // 2
pad_after = window_size // 2
```

### Verification Table

| Window Size | Parity | pad_before | pad_after | Total | Valid? |
|-------------|--------|------------|-----------|-------|--------|
| 3 | odd | 1 | 1 | 3 | ✓ |
| 4 | even | 1 | 2 | 4 | ✓ |
| 5 | odd | 2 | 2 | 5 | ✓ |
| 6 | even | 2 | 3 | 6 | ✓ |
| 7 | odd | 3 | 3 | 7 | ✓ |
| 8 | even | 3 | 4 | 8 | ✓ |
| 10 | even | 4 | 5 | 10 | ✓ |
| 11 | odd | 5 | 5 | 11 | ✓ |

---

## Testing Results

### Test 1: Both Even and Odd Windows Work

Tested window sizes: 3, 4, 5, 6, 7, 8, 10, 11

**Mean Filter:**
```
Window  3× 3 ( odd): Result shape (20, 20), NaN count: 0 ✓
Window  4× 4 (even): Result shape (20, 20), NaN count: 0 ✓
Window  5× 5 ( odd): Result shape (20, 20), NaN count: 0 ✓
Window  6× 6 (even): Result shape (20, 20), NaN count: 0 ✓
Window  7× 7 ( odd): Result shape (20, 20), NaN count: 0 ✓
Window  8× 8 (even): Result shape (20, 20), NaN count: 0 ✓
Window 10×10 (even): Result shape (20, 20), NaN count: 0 ✓
Window 11×11 ( odd): Result shape (20, 20), NaN count: 0 ✓
```

**Median Filter:** All tests pass similarly.

### Test 2: Even Window Produces Sensible Results

Input (5×5 monotonic array):
```
[[1. 2. 3. 4. 5.]
 [2. 3. 4. 5. 6.]
 [3. 4. 5. 6. 7.]
 [4. 5. 6. 7. 8.]
 [5. 6. 7. 8. 9.]]
```

Result with 4×4 window:
```
[[3.  3.5 4.5 5.  5.5]
 [3.5 4.  5.  5.5 6. ]
 [4.5 5.  6.  6.5 7. ]
 [5.  5.5 6.5 7.  7.5]
 [5.5 6.  7.  7.5 8. ]]
```

✓ All values finite, in expected range [1, 9]

---

## Impact

### Before
- ✓ Odd window sizes: 3, 5, 7, 9, 11, ...
- ❌ Even window sizes: ValueError raised

### After
- ✓ **All window sizes ≥ 1** supported
- ✓ Odd: 3, 5, 7, 9, 11, ...
- ✓ **Even: 4, 6, 8, 10, 12, ...**
- ✓ Backward compatible (odd sizes still work identically)

### Use Cases Enabled

1. **Even kernel sizes**: Some filtering applications prefer even windows (e.g., 4×4, 8×8)
2. **Power-of-2 sizes**: Efficient for certain hardware (4, 8, 16, 32)
3. **Flexibility**: Users can choose any window size based on their needs

---

## Backward Compatibility

✅ **Fully backward compatible**

- Existing code using odd window sizes (e.g., 31×31) continues to work identically
- Same numerical results (padding for odd windows unchanged)
- No API changes

---

## Configuration Schema

The schema already supports even window sizes:

**File:** `share/nisar/schemas/insar.yaml`
```yaml
azimuth_offset_filter_options:
    kernel_size: int(min=3, required=False)
```

- `min=3`: Sensible minimum (1×1 trivial, 2×2 arguably too small)
- No odd-only restriction in schema
- Now implementation matches schema capability

---

## Conclusion

The `apply_filter()` function now fully supports both even and odd window sizes through proper asymmetric padding. This enhancement:

1. ✅ Increases flexibility for users
2. ✅ Enables power-of-2 window sizes
3. ✅ Maintains backward compatibility
4. ✅ Produces correct results (validated by tests)
5. ✅ Matches schema specification

**Status**: ✓ Production ready
