# Fix for `_round` Function Multi-Label Support

## Issue Description

The `_round` function in `ultralytics/data/utils.py` (lines 684-685) had a bug where it used `int(c[0])` to extract class information. This approach only works for single-label format but fails for multi-label format.

### The Problem

- **Single-label format**: `c = [0]` (shape: `(1,)`) → `int(c[0])` correctly returns class index `0`
- **Multi-label format**: `c = [1, 0, 1, 0]` (shape: `(C,)`) → `int(c[0])` incorrectly returns `1` (the VALUE at index 0) instead of `[0, 2]` (the class indices)

This caused all multi-label information to be lost when generating dataset statistics for HUB, as only the first element's value was extracted instead of the indices of all active classes.

## Solution

Modified the `_round` function to detect the format and handle each appropriately:

```python
# OLD (buggy) code:
return [[int(c[0]), *(round(float(x), 4) for x in points)] for c, points in zipped]

# NEW (fixed) code:
return [
    [int(c[0]) if len(c) == 1 else np.nonzero(c)[0].tolist(), *(round(float(x), 4) for x in points)]
    for c, points in zipped
]
```

### How It Works

1. **Single-label format** (`len(c) == 1`): Uses `int(c[0])` to extract the class ID
2. **Multi-label format** (`len(c) > 1`): Uses `np.nonzero(c)[0].tolist()` to extract all active class indices

## Testing

Created comprehensive tests to verify the fix:

### Test 1: Single-Label Format (Backward Compatibility)
```python
Input: c = [0]  # Class 0
Output: 0  # ✓ Correct
```

### Test 2: Multi-Label Format
```python
Input: c = [1, 0, 1, 0]  # Classes 0 and 2
OLD buggy output: 1  # ✗ Wrong (returns value, not indices)
NEW fixed output: [0, 2]  # ✓ Correct
```

### Test 3: COCO-Style (80 classes)
```python
Input: c = [1, 0, ..., 0, 1, 0]  # 80 elements, classes 0, 15, 62 active
Output: [0, 15, 62]  # ✓ Correct
```

## Impact

- **File Modified**: `ultralytics/data/utils.py` (lines 673-691)
- **Function**: `HUBDatasetStats.get_json()._round()`
- **Backward Compatibility**: ✓ Fully maintained for single-label datasets
- **Multi-Label Support**: ✓ Now correctly extracts all active class indices

## Verification

All tests pass:
- ✓ Single-label format continues to work correctly
- ✓ Multi-label format now correctly extracts all class indices
- ✓ No linting errors
- ✓ Backward compatibility maintained

## Related Documentation

See `MULTI_LABEL_IMPLEMENTATION.md` for the complete multi-label support implementation across the entire YOLO pipeline.
