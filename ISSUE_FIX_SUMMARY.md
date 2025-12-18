# Issue Fix Summary: IndexError in `_update_label_text`

## Status: ✓ FIXED

## Issue Details

**File**: `ultralytics/data/augment.py`  
**Function**: `BaseMixTransform._update_label_text`  
**Lines Affected**: 457-463 (modified)

### Problem Description

When `cls_tensor` has shape `(N, 1)` (single-label format), the code attempted to execute:
```python
cls_tensor[i, text2id[tuple(text)]] = 1
```

During mosaic augmentation, `mix_texts` combines texts from multiple labels, so `text2id` maps texts to indices ranging from 0 to `len(mix_texts)-1`. Attempting to access `cls_tensor[i, k]` where `k > 0` on a tensor with only 1 column caused an **IndexError**.

### Example Scenario That Triggered the Bug

```python
# Main image has 2 labels with single-label format
labels = {
    "texts": [["cat"], ["dog"]],
    "cls": torch.tensor([[0], [0]])  # Shape: (2, 1)
}

# Mixed image (from mosaic) has 2 more labels
mix_labels = [{
    "texts": [["bird"], ["fish"]],
    "cls": torch.tensor([[0], [0]])  # Shape: (2, 1)
}]

# After combining, we have 4 unique classes
mix_texts = [["cat"], ["dog"], ["bird"], ["fish"]]
text2id = {("cat",): 0, ("dog",): 1, ("bird",): 2, ("fish",): 3}

# ERROR: Trying to access cls_tensor[i, 2] when cls_tensor.shape = (2, 1)
```

## Solution

Added tensor expansion logic before attempting to set one-hot encoded values:

```python
# Expand cls_tensor to accommodate all classes in mix_texts if needed
n_samples = len(cls_indices)
n_classes = len(mix_texts)
if cls_tensor.ndim == 1 or cls_tensor.shape[-1] < n_classes:
    # Create new tensor with correct shape and transfer to same device
    new_cls_tensor = torch.zeros((n_samples, n_classes), dtype=cls_tensor.dtype, device=cls_tensor.device)
    cls_tensor = new_cls_tensor
```

### Key Features of the Fix

1. **Detects when expansion is needed**: Checks if tensor is 1D or has insufficient columns
2. **Creates properly sized tensor**: Expands to `(n_samples, n_classes)` shape
3. **Preserves tensor properties**: Maintains original dtype and device
4. **Backward compatible**: Doesn't affect already-sufficient tensors

## Changes Made

**File**: `ultralytics/data/augment.py`

**Before** (lines 450-471):
```python
for label in [labels] + labels["mix_labels"]:
    cls_tensor = label["cls"]
    if cls_tensor.ndim > 1 and cls_tensor.shape[-1] > 1:
        cls_indices = cls_tensor.argmax(-1)
    else:
        cls_indices = cls_tensor.squeeze(-1)
    for i, cls_val in enumerate(cls_indices.tolist()):
        text = label["texts"][int(cls_val)]
        cls_tensor[i] = 0
        cls_tensor[i, text2id[tuple(text)]] = 1  # ← IndexError here!
    label["cls"] = cls_tensor
    label["texts"] = mix_texts
```

**After** (lines 450-471):
```python
for label in [labels] + labels["mix_labels"]:
    cls_tensor = label["cls"]
    if cls_tensor.ndim > 1 and cls_tensor.shape[-1] > 1:
        cls_indices = cls_tensor.argmax(-1)
    else:
        cls_indices = cls_tensor.squeeze(-1)
    
    # Expand cls_tensor to accommodate all classes in mix_texts if needed
    n_samples = len(cls_indices)
    n_classes = len(mix_texts)
    if cls_tensor.ndim == 1 or cls_tensor.shape[-1] < n_classes:
        # Create new tensor with correct shape and transfer to same device
        new_cls_tensor = torch.zeros((n_samples, n_classes), dtype=cls_tensor.dtype, device=cls_tensor.device)
        cls_tensor = new_cls_tensor
    
    for i, cls_val in enumerate(cls_indices.tolist()):
        text = label["texts"][int(cls_val)]
        cls_tensor[i] = 0
        cls_tensor[i, text2id[tuple(text)]] = 1  # ← Now works correctly!
    label["cls"] = cls_tensor
    label["texts"] = mix_texts
```

## Verification

### Test Results

✓ **Syntax Check**: Passed  
✓ **Original Bug Reproduction**: Confirmed IndexError in original code  
✓ **Fix Validation**: Confirmed fix prevents IndexError  
✓ **Edge Cases**: All 5 edge cases pass  

### Tested Scenarios

1. **Single-label format (N, 1) with multiple mixed classes** ✓
   - Input: shape (2, 1), 4 classes
   - Output: shape (2, 4)
   - Result: No IndexError

2. **1D tensor format** ✓
   - Input: shape (2,), 4 classes
   - Output: shape (2, 4)
   - Result: No IndexError

3. **Multi-label format (sufficient columns)** ✓
   - Input: shape (2, 4), 4 classes
   - Output: shape (2, 4)
   - Result: No expansion needed, works as before

4. **Multi-label format (insufficient columns)** ✓
   - Input: shape (2, 2), 4 classes
   - Output: shape (2, 4)
   - Result: Properly expanded

5. **Single class (no expansion needed)** ✓
   - Input: shape (2, 1), 1 class
   - Output: shape (2, 1)
   - Result: No expansion needed, works correctly

## Impact

- **Fixes**: IndexError during mosaic augmentation with single-label format
- **Affects**: Training pipelines using mosaic augmentation with text labels
- **Breaking Changes**: None
- **Performance**: Minimal overhead (only creates new tensor when needed)

## Related Code

This is the only location in the codebase where `cls_tensor[i, text2id[...]]` pattern is used.

## Additional Files

- `FIX_VERIFICATION.md`: Detailed technical analysis of the fix
