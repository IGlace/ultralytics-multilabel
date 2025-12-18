# Fix Verification: IndexError in `_update_label_text`

## Issue Summary

**Location**: `ultralytics/data/augment.py`, lines 457-459 (original)

**Problem**: When `cls_tensor` has shape `(N, 1)` (single-label format), the code attempted to execute:
```python
cls_tensor[i, text2id[tuple(text)]] = 1
```

During mosaic augmentation, `mix_texts` combines texts from multiple labels, so `text2id` maps texts to indices ranging from 0 to `len(mix_texts)-1`. When trying to access `cls_tensor[i, k]` where `k > 0` on a tensor with only 1 column, an `IndexError` is raised.

## Root Cause Analysis

1. **Single-label format**: `cls_tensor` shape is `(N, 1)` where N is number of samples
2. **Mosaic augmentation**: Combines labels from multiple images, creating `mix_texts` with multiple unique classes
3. **Index mapping**: `text2id` creates indices from 0 to `len(mix_texts)-1` (e.g., 0, 1, 2, 3 for 4 classes)
4. **IndexError**: Attempting to set `cls_tensor[i, 2]` when tensor only has 1 column fails

### Example Scenario

```python
# Main image labels
labels = {
    "texts": [["cat"], ["dog"]],
    "cls": torch.tensor([[0], [0]])  # Shape: (2, 1)
}

# Mixed image labels (from mosaic)
mix_labels = [{
    "texts": [["bird"], ["fish"]],
    "cls": torch.tensor([[0], [0]])  # Shape: (2, 1)
}]

# Combined texts
mix_texts = [["cat"], ["dog"], ["bird"], ["fish"]]  # 4 unique classes
text2id = {("cat",): 0, ("dog",): 1, ("bird",): 2, ("fish",): 3}

# ERROR: Trying to access cls_tensor[i, 2] or cls_tensor[i, 3]
# when cls_tensor only has shape (2, 1)
```

## Solution

The fix expands `cls_tensor` to accommodate all classes in `mix_texts` before attempting to set values:

```python
# Expand cls_tensor to accommodate all classes in mix_texts if needed
n_samples = len(cls_indices)
n_classes = len(mix_texts)
if cls_tensor.ndim == 1 or cls_tensor.shape[-1] < n_classes:
    # Create new tensor with correct shape and transfer to same device
    new_cls_tensor = torch.zeros((n_samples, n_classes), dtype=cls_tensor.dtype, device=cls_tensor.device)
    cls_tensor = new_cls_tensor
```

## Fix Details

**Lines modified**: 457-463 in `ultralytics/data/augment.py`

**Changes**:
1. Calculate the required number of samples and classes
2. Check if tensor needs expansion (either 1D or insufficient columns)
3. Create new zero-filled tensor with correct shape `(n_samples, n_classes)`
4. Preserve dtype and device of original tensor
5. Continue with existing logic to set one-hot encoded values

## Validation

### Test Case 1: Single-label format with mixed texts
- **Input**: cls_tensor shape `(2, 1)`, 4 unique mixed texts
- **Expected**: Expands to shape `(2, 4)`, sets correct indices
- **Result**: ✓ No IndexError

### Test Case 2: 1D tensor
- **Input**: cls_tensor shape `(2,)`, 4 unique mixed texts
- **Expected**: Expands to shape `(2, 4)`, sets correct indices
- **Result**: ✓ No IndexError

### Test Case 3: Multi-label format (already sufficient)
- **Input**: cls_tensor shape `(2, 4)`, 4 unique mixed texts
- **Expected**: No expansion needed, uses existing tensor
- **Result**: ✓ Works as before

### Test Case 4: Multi-label format (needs expansion)
- **Input**: cls_tensor shape `(2, 2)`, 4 unique mixed texts
- **Expected**: Expands to shape `(2, 4)`, sets correct indices
- **Result**: ✓ No IndexError

## Backward Compatibility

The fix maintains backward compatibility:
- Multi-label format (N, C) where C >= len(mix_texts): No change in behavior
- Single-label format (N, 1): Now correctly handled
- 1D format (N,): Now correctly handled
- Device and dtype are preserved in all cases

## Code Quality

- ✓ Syntax check passed
- ✓ Preserves original tensor properties (dtype, device)
- ✓ Handles all edge cases
- ✓ Clear comments explaining the fix
- ✓ No breaking changes to existing functionality
