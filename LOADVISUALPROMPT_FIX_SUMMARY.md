# LoadVisualPrompt Multi-Label Fix Summary

## Issue Description

The `LoadVisualPrompt.__call__()` method in `ultralytics/data/augment.py` incorrectly handled multi-label format data at line 2217.

**Problem**: When `labels["cls"]` has multi-label shape `(N, C)` where C > 1:
- `squeeze(-1)` has no effect (cannot squeeze a dimension with size > 1)
- The resulting 2D tensor `(N, C)` is passed to `get_visuals()`
- `get_visuals()` expects a 1D tensor of class indices for its `category` parameter
- At line 2257, `torch.unique(category, sorted=True, return_inverse=True)` is called, which expects 1D input or will flatten the tensor incorrectly

## Affected Code Location

**File**: `ultralytics/data/augment.py`  
**Lines**: 2217-2224 (after fix)  
**Class**: `LoadVisualPrompt`  
**Method**: `__call__()`

## Solution

Added conditional logic to detect and handle both single-label and multi-label formats:

### Before (Line 2217):
```python
cls = labels["cls"].squeeze(-1).to(torch.int)
```

### After (Lines 2217-2224):
```python
# Handle both single-label (N, 1) and multi-label (N, C) formats
cls_tensor = labels["cls"]
if cls_tensor.ndim > 1 and cls_tensor.shape[1] > 1:
    # Multi-label format: convert multi-hot to class indices using argmax
    cls = cls_tensor.argmax(dim=1).to(torch.int)
else:
    # Single-label format: squeeze and convert to int
    cls = cls_tensor.squeeze(-1).to(torch.int)
```

## Implementation Details

### Multi-Label Handling Strategy

For multi-label format `(N, C)`, the fix uses `argmax(dim=1)` to convert the multi-hot vector to a single class index:
- **Single-label format**: `[[1], [0], [2]]` → squeeze works → `[1, 0, 2]`
- **Multi-label format**: `[[1,0,0], [0,1,0], [0,0,1]]` → argmax used → `[0, 1, 2]`

**Note**: `argmax` returns the index of the first maximum value (the first active class). This is appropriate for visual prompt generation since:
1. Visual prompts are used to provide spatial guidance to the model
2. The primary class (first active class) is sufficient for generating the visual mask
3. Multi-label predictions are still supported downstream via NMS with `multi_label=True`

### Compatibility

- **Backward compatibility**: Single-label format `(N, 1)` continues to work exactly as before
- **Multi-label support**: Multi-label format `(N, C)` now works correctly
- **PyTorch tensors**: Uses `argmax(dim=1)` for proper dimension handling

## Testing

Created and ran comprehensive tests (`test_visual_prompt_fix.py`) to verify:

1. ✅ **squeeze(-1) behavior**: Confirmed the original issue with multi-label format
2. ✅ **torch.unique behavior**: Demonstrated that 2D tensors cause incorrect flattening
3. ✅ **Fix logic**: Verified the conditional logic works for both formats
4. ✅ **Integration test**: Tested LoadVisualPrompt with both single-label and multi-label data
5. ✅ **Multiple active classes**: Tested multi-label with multiple classes active per box
6. ✅ **Backward compatibility**: Verified single-label format still works
7. ✅ **Linter check**: No linter errors introduced

### Test Results

All tests passed successfully:
```
✅ Single-label format (N, 1): Success! visuals shape: torch.Size([3, 80, 80])
✅ Multi-label format (N, C): Success! visuals shape: torch.Size([3, 80, 80])
✅ Multi-label with multiple active classes: Success! visuals shape: torch.Size([2, 80, 80])
```

## Impact

- **Scope**: Fixes critical issue that would cause incorrect behavior with multi-label data in visual prompt loading
- **Backward compatibility**: ✅ Fully maintained - single-label format works exactly as before
- **Performance**: Minimal overhead - only adds shape checking and conditional logic
- **Correctness**: Uses `argmax` to select the primary class from multi-hot vectors for visual prompt generation

## Context

This fix is part of the comprehensive multi-label bounding box support implementation documented in `MULTI_LABEL_IMPLEMENTATION.md`. It follows the same pattern as previous fixes:

1. **`BaseMixTransform._update_label_text()`** (lines 451-461) - Fixed in MULTI_LABEL_BUG_FIX_SUMMARY.md
2. **`RandomLoadText.__call__()`** (lines 2362-2368) - Fixed in MULTI_LABEL_BUG_FIX_SUMMARY.md
3. **`LoadVisualPrompt.__call__()`** (lines 2217-2224) - **This fix**

## Files Modified

- `ultralytics/data/augment.py`:
  - Lines 2217-2224: Fixed `LoadVisualPrompt.__call__()`

## Related Documentation

- `MULTI_LABEL_IMPLEMENTATION.md` - Comprehensive documentation on multi-label support
- `MULTI_LABEL_BUG_FIX_SUMMARY.md` - Previous multi-label bug fixes
- `test_visual_prompt_fix.py` - Test script for this fix

## Conclusion

This fix ensures `LoadVisualPrompt` correctly handles both single-label `(N, 1)` and multi-label `(N, C)` formats by converting multi-hot vectors to class indices before passing to `get_visuals()`. The fix maintains full backward compatibility while enabling proper multi-label support.
