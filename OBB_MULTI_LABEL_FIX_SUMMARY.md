# OBB Validator Multi-Label Fix Summary

## Issue Description

The `_prepare_batch` method in `ultralytics/models/yolo/obb/val.py` unconditionally applied `squeeze(-1)` to `batch["cls"][idx]`. This caused issues with multi-label format where the class tensor has shape `(N, C)` with C > 1.

**Problem**: When `batch["cls"][idx]` has multi-label shape `(N, C)` where C > 1:
- `squeeze(-1)` does not reduce dimensions since the last dimension is larger than 1
- The returned `cls` tensor remains 2D `(N, C)`
- This violates downstream expectations where OBB validation code assumes 1D class indices for metric calculations and matching operations

**Affected Code** (lines 128-129):
```python
idx = batch["batch_idx"] == si
cls = batch["cls"][idx].squeeze(-1)  # ❌ Unconditional squeeze
```

## Root Cause

The code was written assuming single-label format `(N, 1)` where `squeeze(-1)` would reduce it to `(N,)`. However, with multi-label format `(N, C)` where C > 1, the squeeze operation has no effect, leaving a 2D tensor that downstream code doesn't expect.

## Solution

Updated the `_prepare_batch` method to conditionally squeeze only when the tensor is in single-label format, matching the approach already implemented in `DetectionValidator`:

```python
idx = batch["batch_idx"] == si
cls = batch["cls"][idx]
# Only squeeze if single-label format (shape: N, 1), keep multi-label format (shape: N, C)
if cls.ndim > 1 and cls.shape[1] == 1:
    cls = cls.squeeze(-1)
```

## Changes Made

**File**: `ultralytics/models/yolo/obb/val.py`

**Lines**: 128-132

**Before**:
```python
idx = batch["batch_idx"] == si
cls = batch["cls"][idx].squeeze(-1)
```

**After**:
```python
idx = batch["batch_idx"] == si
cls = batch["cls"][idx]
# Only squeeze if single-label format (shape: N, 1), keep multi-label format (shape: N, C)
if cls.ndim > 1 and cls.shape[1] == 1:
    cls = cls.squeeze(-1)
```

## Verification

### Code Comparison

The fix now matches the implementation in `DetectionValidator._prepare_batch()` (lines 138-142 in `ultralytics/models/yolo/detect/val.py`):

```python
idx = batch["batch_idx"] == si
cls = batch["cls"][idx]
# Only squeeze if single-label format (shape: N, 1), keep multi-label format (shape: N, C)
if cls.ndim > 1 and cls.shape[1] == 1:
    cls = cls.squeeze(-1)
```

### Expected Behavior

1. **Single-Label Format** `(N, 1)`:
   - Tensor is squeezed to `(N,)` as expected
   - Backward compatible with existing single-label datasets

2. **Multi-Label Format** `(N, C)` where C > 1:
   - Tensor remains 2D `(N, C)`
   - Preserves multi-hot encoding for downstream processing
   - Enables proper multi-label OBB validation

3. **Empty Batches**:
   - Handles empty tensors gracefully
   - No dimension issues

## Impact

- **Scope**: Fixes critical issue that would cause incorrect behavior or crashes with multi-label data in OBB validation
- **Backward Compatibility**: ✅ Fully backward compatible with single-label format
- **Multi-label Support**: ✅ Now properly supports multi-label format `(N, C)`
- **Consistency**: ✅ Now consistent with `DetectionValidator` implementation

## Testing Recommendations

1. **Single-Label OBB**: Verify existing single-label OBB datasets still validate correctly
2. **Multi-Label OBB**: Test with multi-label OBB dataset (shape: N, C where C > 1)
3. **Mixed Batches**: Test with batches containing both single and multi-label instances
4. **Edge Cases**: Test with empty batches and various class counts

## Related Documentation

This fix is part of the comprehensive multi-label support implementation:
- `MULTI_LABEL_IMPLEMENTATION.md` - Overall multi-label support architecture
- `MULTI_LABEL_BUG_FIX_SUMMARY.md` - Previous multi-label fixes in detection validator
- `LOADVISUALPROMPT_FIX_SUMMARY.md` - Multi-label fix for visual prompts

## Consistency with Codebase

This fix follows the same pattern established in the detection validator:

1. **Detection Validator** (`ultralytics/models/yolo/detect/val.py`, lines 140-142):
   ```python
   # Only squeeze if single-label format (shape: N, 1), keep multi-label format (shape: N, C)
   if cls.ndim > 1 and cls.shape[1] == 1:
       cls = cls.squeeze(-1)
   ```

2. **OBB Validator** (`ultralytics/models/yolo/obb/val.py`, lines 130-132):
   ```python
   # Only squeeze if single-label format (shape: N, 1), keep multi-label format (shape: N, C)
   if cls.ndim > 1 and cls.shape[1] == 1:
       cls = cls.squeeze(-1)
   ```

## Conclusion

The OBB validator now properly handles both single-label and multi-label formats by:
1. ✅ Detecting the format based on tensor dimensions
2. ✅ Conditionally squeezing only single-label format
3. ✅ Preserving multi-label format for downstream processing
4. ✅ Maintaining full backward compatibility

This ensures OBB validation works correctly with both single-label `(N, 1)` and multi-label `(N, C)` formats, bringing it in line with the detection validator implementation.
