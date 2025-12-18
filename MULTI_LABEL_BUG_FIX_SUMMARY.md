# Multi-Label Bug Fix Summary

## Issue Description

The code in `ultralytics/data/augment.py` had two locations that incorrectly handled multi-label format data:

**Problem**: When `label["cls"]` has multi-label shape `(N, C)` where C > 1:
- `squeeze(-1)` does nothing (can't squeeze a dimension with size > 1)
- `.tolist()` produces a list of lists
- Iterating gives `cls` as a list rather than a scalar
- Calling `int(cls)` raises `TypeError: int() argument must be a string, a bytes-like object or a real number, not 'list'`

## Affected Code Locations

1. **`BaseMixTransform._update_label_text()` method** (lines 450-453)
   - Function: Updates label text and class IDs for mixed labels in image augmentation
   - Used by: Mosaic, MixUp, and other mix transforms

2. **`RandomLoadText.__call__()` method** (lines 2349-2350)
   - Function: Randomly samples positive and negative texts and updates class indices
   - Used by: Text-based detection models

## Solution

Added conditional logic to detect and handle both single-label and multi-label formats:

### Fix #1: BaseMixTransform._update_label_text() (lines 451-461)

**Before:**
```python
for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
    text = label["texts"][int(cls)]
    label["cls"][i] = text2id[tuple(text)]
```

**After:**
```python
cls_array = label["cls"]
# Handle both single-label (N, 1) and multi-label (N, C) formats
if hasattr(cls_array, "ndim") and cls_array.ndim > 1 and cls_array.shape[1] > 1:
    # Multi-label format: convert multi-hot to class indices using argmax
    if isinstance(cls_array, torch.Tensor):
        cls_list = cls_array.argmax(dim=1).tolist()
    else:
        cls_list = cls_array.argmax(axis=1).tolist()
else:
    # Single-label format: squeeze and convert to list
    cls_list = cls_array.squeeze(-1).tolist()

for i, cls in enumerate(cls_list):
    text = label["texts"][int(cls)]
    label["cls"][i] = text2id[tuple(text)]
```

### Fix #2: RandomLoadText.__call__() (lines 2362-2368)

**Before:**
```python
for i, label in enumerate(cls.squeeze(-1).tolist()):
    if label not in label2ids:
        continue
    valid_idx[i] = True
    new_cls.append([label2ids[label]])
```

**After:**
```python
# Handle both single-label (N, 1) and multi-label (N, C) formats
if cls.ndim > 1 and cls.shape[1] > 1:
    # Multi-label format: convert multi-hot to class indices using argmax
    cls_list = cls.argmax(axis=1).tolist()
else:
    # Single-label format: squeeze and convert to list
    cls_list = cls.squeeze(-1).tolist()

for i, label in enumerate(cls_list):
    if label not in label2ids:
        continue
    valid_idx[i] = True
    new_cls.append([label2ids[label]])
```

## Implementation Details

### Multi-Label Handling Strategy

For multi-label format `(N, C)`, the fix uses `argmax` to convert the multi-hot vector to a single class index:
- **Single-label format**: `[[1], [0], [2]]` → `[1, 0, 2]` (squeeze works)
- **Multi-label format**: `[[1,0,0], [0,1,0], [0,0,1]]` → `[0, 1, 2]` (argmax used)

**Note**: `argmax` returns the index of the first maximum value (the first active class). This is appropriate for these functions since they:
1. `_update_label_text`: Maps class indices through text labels for mix transforms
2. `RandomLoadText`: Samples and remaps class indices for text-based models

### Compatibility

- **Torch tensors**: Uses `argmax(dim=1)` for PyTorch tensors
- **NumPy arrays**: Uses `argmax(axis=1)` for NumPy arrays
- **Backward compatibility**: Single-label format `(N, 1)` continues to work exactly as before

## Testing

Created and ran comprehensive tests to verify:

1. ✅ **Bug reproduction**: Confirmed the original issue with multi-label format
2. ✅ **Single-label compatibility**: Verified single-label format still works
3. ✅ **Multi-label fix**: Confirmed multi-label format now works correctly
4. ✅ **Edge cases**: Tested various multi-hot patterns (multiple active classes, all classes active, etc.)
5. ✅ **Function simulation**: Simulated actual function behavior with both formats
6. ✅ **Syntax validation**: Python compilation check passed
7. ✅ **Linter check**: No linter errors introduced

## Impact

- **Scope**: Fixes critical TypeError that would crash training/inference with multi-label data in mix transforms and text-based models
- **Backward compatibility**: ✅ Fully maintained - single-label format works exactly as before
- **Performance**: Minimal overhead - only adds shape checking and conditional logic
- **Correctness**: Uses `argmax` to select the primary class from multi-hot vectors

## Files Modified

- `ultralytics/data/augment.py`:
  - Lines 451-461: Fixed `BaseMixTransform._update_label_text()`
  - Lines 2362-2368: Fixed `RandomLoadText.__call__()`

## Related Documentation

See `MULTI_LABEL_IMPLEMENTATION.md` for comprehensive documentation on multi-label support across the entire YOLO pipeline.
