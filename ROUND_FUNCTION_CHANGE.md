# Visual Comparison of the Fix

## Location
**File**: `ultralytics/data/utils.py`  
**Lines**: 684-691 (in `HUBDatasetStats.get_json()._round()`)

## Before (Buggy Code)

```python
def _round(labels):
    """Update labels to integer class and 4 decimal place floats."""
    if self.task == "detect":
        coordinates = labels["bboxes"]
    elif self.task in {"segment", "obb"}:
        coordinates = [x.flatten() for x in labels["segments"]]
    elif self.task == "pose":
        n, nk, nd = labels["keypoints"].shape
        coordinates = np.concatenate((labels["bboxes"], labels["keypoints"].reshape(n, nk * nd)), 1)
    else:
        raise ValueError(f"Undefined dataset task={self.task}.")
    zipped = zip(labels["cls"], coordinates)
    return [[int(c[0]), *(round(float(x), 4) for x in points)] for c, points in zipped]
    #       ^^^^^^^^^ BUG: Only takes first element value, not class indices
```

## After (Fixed Code)

```python
def _round(labels):
    """Update labels to integer class and 4 decimal place floats."""
    if self.task == "detect":
        coordinates = labels["bboxes"]
    elif self.task in {"segment", "obb"}:
        coordinates = [x.flatten() for x in labels["segments"]]
    elif self.task == "pose":
        n, nk, nd = labels["keypoints"].shape
        coordinates = np.concatenate((labels["bboxes"], labels["keypoints"].reshape(n, nk * nd)), 1)
    else:
        raise ValueError(f"Undefined dataset task={self.task}.")
    zipped = zip(labels["cls"], coordinates)
    # Handle both single-label and multi-label formats
    # Single-label: c is [class_id], use int(c[0])
    # Multi-label: c is [1,0,1,0,...], extract indices using np.nonzero
    return [
        [int(c[0]) if len(c) == 1 else np.nonzero(c)[0].tolist(), *(round(float(x), 4) for x in points)]
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ FIX: Handles both formats
        for c, points in zipped
    ]
```

## Key Change

```python
# OLD (line 685):
return [[int(c[0]), ...] for c, points in zipped]

# NEW (lines 688-691):
return [
    [int(c[0]) if len(c) == 1 else np.nonzero(c)[0].tolist(), ...]
    for c, points in zipped
]
```

## Example Behavior

### Single-Label Format (Backward Compatible)
```python
c = np.array([5])  # Class 5

# OLD: int(c[0]) = 5 ✓
# NEW: int(c[0]) if len(c) == 1 = 5 ✓
```

### Multi-Label Format (Now Fixed)
```python
c = np.array([1, 0, 1, 0])  # Classes 0 and 2 are active

# OLD: int(c[0]) = 1 ✗ WRONG (returns value at index 0, not class indices)
# NEW: np.nonzero(c)[0].tolist() = [0, 2] ✓ CORRECT (returns active class indices)
```

### Multi-Label with 80 Classes (COCO)
```python
c = np.zeros(80)
c[[0, 15, 62]] = 1  # Classes 0, 15, 62 are active

# OLD: int(c[0]) = 1 ✗ WRONG (loses all but one class)
# NEW: np.nonzero(c)[0].tolist() = [0, 15, 62] ✓ CORRECT (preserves all classes)
```

## Impact

✓ **Backward Compatible**: Single-label datasets continue to work exactly as before  
✓ **Multi-Label Fixed**: Multi-label datasets now correctly export all class indices  
✓ **No Breaking Changes**: The fix only affects the output format for multi-label data
