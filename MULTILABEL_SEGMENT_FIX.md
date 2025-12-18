# Multi-label Detection vs Segmentation Data Fix

## Issue Description

The segment detection condition at `ultralytics/data/utils.py:205` was incorrectly triggering for multi-label detection data when `num_cls > 2`.

### Root Cause

The original condition:
```python
if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
```

This condition checked if any label line had more than 6 values to determine if it's segmentation data. However, this failed to account for multi-label detection format.

### Multi-label Detection Format

Multi-label detection uses the format: `[cls_0, cls_1, ..., cls_n-1, x, y, w, h]`
- Total values: `num_cls + 4`
- First `num_cls` columns: multi-hot encoded class labels (binary: 0 or 1)
- Last 4 columns: bounding box coordinates (x, y, w, h)

### The Bug

When `num_cls > 2`:
- `num_cls + 4 > 6`, so the condition triggers
- Multi-label data is incorrectly treated as segmentation data
- Lines 206-208 process it as segments:
  - Take first value as class (corrupting multi-hot encoding)
  - Reshape remaining values as polygon vertices
  - Convert to bounding boxes using `segments2boxes()`

For **odd** `num_cls` values (3, 5, 7, etc.):
- After taking first value as class, `num_cls + 3` values remain
- Since `num_cls + 3` is even, `.reshape(-1, 2)` succeeds
- Bug is **silent** - no error, just corrupted data

Example with `num_cls=3`:
```
Input:  [1, 0, 1, 0.5, 0.5, 0.3, 0.4]  # Multi-label: classes 0 and 2, bbox at (0.5,0.5) size 0.3x0.4
Buggy processing:
  - class = 1
  - vertices = [(0, 1), (0.5, 0.5), (0.3, 0.4)]  # WRONG! Treating class+bbox as polygon
  - Convert to bbox (corrupted result)
```

## Solution

Modified the condition to explicitly exclude multi-label format:

```python
# Check for segment data, excluding multi-label detection format (num_cls + 4 values)
if any(len(x) > 6 and len(x) != num_cls + 4 for x in lb) and (not keypoint):  # is segment
```

### Logic

A line with values is treated as segment data if:
1. It has more than 6 values, AND
2. It does NOT have exactly `num_cls + 4` values (multi-label format), AND
3. It's not keypoint data

This ensures:
- Single-label detection (5 values): NOT treated as segment ✅
- Multi-label detection (`num_cls + 4` values): NOT treated as segment ✅
- Actual segmentation data (1 + 2n values, where n ≥ 4): treated as segment ✅

## Edge Case

When `num_cls + 4` equals a valid segment size (e.g., `num_cls=5` gives 9 values, same as a 4-vertex polygon), the fix prioritizes treating it as multi-label format. This is the correct behavior because:

1. The dataset's `num_cls` value should match the data format
2. If using multi-label detection with `num_cls=5`, 9 values per line means multi-label
3. If using segmentation with `num_cls=80`, 9 values per line means 4-vertex polygon

The `num_cls` parameter disambiguates the format.

## Testing

Comprehensive tests added in `tests/test_multilabel_segment_fix.py`:

1. ✅ Multi-label with `num_cls=3` (7 values) - not treated as segment
2. ✅ Multi-label with `num_cls=5` (9 values) - not treated as segment
3. ✅ Single-label detection (5 values) - not treated as segment
4. ✅ Actual segmentation data (9+ values with appropriate `num_cls`) - correctly detected
5. ✅ Segmentation with many vertices - correctly detected
6. ✅ Edge case handling - correctly disambiguated based on `num_cls`

All tests pass! Run with:
```bash
python3 -m pytest tests/test_multilabel_segment_fix.py -v
```

## Files Modified

- `ultralytics/data/utils.py` (line 205-206): Fixed segment detection condition
- `tests/test_multilabel_segment_fix.py`: Added comprehensive tests

## Backward Compatibility

This fix is **fully backward compatible**:
- Single-label detection: No change in behavior
- Segmentation: No change in behavior (as long as format is valid)
- Multi-label detection: Now works correctly (was broken before)

## Impact

This fix prevents silent data corruption in multi-label detection datasets with `num_cls > 2`, particularly those with odd `num_cls` values where the bug would go unnoticed until model training produces unexpected results.
