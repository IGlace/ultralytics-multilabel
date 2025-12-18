# Multi-Label Bounding Box Support Implementation

## Overview

This document describes the comprehensive implementation of end-to-end multi-label bounding box support in Ultralytics YOLO. This allows each bounding box to have multiple class labels (multi-hot encoding) instead of a single class label.

## Label Format

### Single-Label Format (Previous)
```
<class_id> <x_center> <y_center> <width> <height>
```
Example: `0 0.5 0.5 0.3 0.4` (one class per box)

### Multi-Label Format (New)
```
<cls_0> <cls_1> ... <cls_N-1> <x_center> <y_center> <width> <height>
```
Example (80 classes): `1 0 0 1 0 ... 0 0.5 0.5 0.3 0.4` (multiple classes per box)

Where:
- `cls_i` is 0 or 1 (binary multi-hot encoding)
- N is the total number of classes in the dataset
- At least one class must be active (value=1) per box

## Implementation Details

### 1. Data Parsing & Caching (`ultralytics/data/utils.py`, `ultralytics/data/dataset.py`)

**Changes:**
- `verify_image_label()`: Now detects and validates multi-label format
  - Checks if labels have `num_classes + 4` columns (multi-label) or 5 columns (single-label)
  - Validates that multi-hot classes are binary (0 or 1)
  - Ensures each instance has at least one active class

- `YOLODataset.cache_labels()`: Stores cls data based on format
  - Single-label: stores as `(N, 1)` with integer class IDs
  - Multi-label: stores as `(N, num_classes)` with multi-hot vectors

**Internal Representation:**
- Single-label: `cls` shape is `(N, 1)` - integer class IDs
- Multi-label: `cls` shape is `(N, C)` - multi-hot binary vectors

### 2. Data Augmentation (`ultralytics/data/augment.py`, `ultralytics/data/base.py`)

**Changes:**
- `Format.__call__()`: Handles both single-label and multi-label cls tensors
  - Preserves multi-label shape `(N, C)` when converting to PyTorch tensors
  
- `BaseDataset.update_labels()`: Updated filtering logic
  - Single-label: filters using equality check on class IDs
  - Multi-label: filters boxes that have ANY of the included classes active
  - `single_cls` mode: converts all instances to class 0

### 3. Loss Functions (`ultralytics/utils/loss.py`)

**Changes:**
- `v8DetectionLoss.preprocess()`: Added `num_cls_cols` parameter
  - Handles variable-width class columns (1 for single-label, C for multi-label)
  - Correctly scales bbox coordinates after class columns

- `v8DetectionLoss.__call__()`: Detects format and splits targets appropriately
  - Automatically determines number of class columns from batch data
  - Passes `num_cls_cols` to preprocessing and target splitting

- `v8SegmentationLoss.__call__()`: Same updates as detection loss
- `v8PoseLoss.__call__()`: Same updates as detection loss  
- `v8OBBLoss.__call__()` and `v8OBBLoss.preprocess()`: 
  - Custom handling for 5-coord oriented bounding boxes
  - Correctly extracts class columns before bbox columns

### 4. Task-Aligned Assigner (`ultralytics/utils/tal.py`)

**Changes:**
- `TaskAlignedAssigner.get_box_metrics()`: Handles multi-hot gt_labels
  - Single-label: Uses class index to gather prediction scores
  - Multi-label: Computes max score across all active classes per GT box
  
- `TaskAlignedAssigner.get_targets()`: Creates appropriate target_scores
  - Single-label: Creates one-hot encoding via scatter operation
  - Multi-label: Uses multi-hot labels directly as target scores

### 5. Non-Maximum Suppression (`ultralytics/utils/nms.py`, `ultralytics/models/yolo/detect/predict.py`)

**Changes:**
- NMS already supported multi-label via `multi_label=True` parameter
- `DetectionPredictor.postprocess()`: Now explicitly passes `multi_label=True`
- When enabled, NMS generates multiple detections per box (one per class above threshold)

### 6. Validation (`ultralytics/models/yolo/detect/val.py`, `ultralytics/engine/validator.py`)

**Changes:**
- `DetectionValidator._prepare_batch()`: Conditionally squeezes cls
  - Only squeezes single-label format `(N, 1)` → `(N,)`
  - Preserves multi-label format `(N, C)`

- `DetectionValidator.update_metrics()`: Extracts unique classes correctly
  - Single-label: Uses `np.unique()` on 1D array
  - Multi-label: Finds class indices where sum > 0 across instances

- `BaseValidator.match_predictions()`: Updated class matching logic
  - Single-label: Direct equality comparison
  - Multi-label: Checks if prediction class is active in GT multi-hot vector using gather operation

### 7. Statistics & Metrics (`ultralytics/data/utils.py`)

**Changes:**
- `HUBDatasetStats.get_json()`: Updated class counting
  - Single-label: Uses `np.bincount()` on flattened class IDs
  - Multi-label: Sums multi-hot vectors across instances

## Backward Compatibility

The implementation maintains **full backward compatibility** with single-label datasets:

1. **Automatic Format Detection**: All components detect whether cls is `(N, 1)` or `(N, C)`
2. **Conditional Logic**: Different code paths for single-label vs multi-label
3. **Default Behavior**: Without multi-label data, everything works as before

## Usage Examples

### Training with Multi-Label Data

```python
from ultralytics import YOLO

# Dataset should have labels in multi-label format
# Label file example (for 80 classes):
# 1 0 0 1 0 ... 0 0.5 0.5 0.3 0.4
model = YOLO('yolo11n.pt')
model.train(data='multilabel_dataset.yaml', epochs=100)
```

### Prediction

```python
# Predictions will return multiple classes per box
results = model.predict('image.jpg')
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = box.cls  # Can have multiple detections per box
        conf = box.conf
        xyxy = box.xyxy
```

### Creating Multi-Label Dataset

Convert your annotations to multi-label format:

```python
import numpy as np

num_classes = 80
# Example: object has classes 0 and 3 active
cls_multihot = np.zeros(num_classes)
cls_multihot[[0, 3]] = 1

# Label line: <cls_0> <cls_1> ... <cls_79> <x> <y> <w> <h>
label_line = list(cls_multihot) + [0.5, 0.5, 0.3, 0.4]
```

## Files Modified

### Core Data Pipeline
- `ultralytics/data/utils.py` - Label parsing and validation
- `ultralytics/data/dataset.py` - Label caching
- `ultralytics/data/augment.py` - Data formatting
- `ultralytics/data/base.py` - Label filtering

### Training Pipeline
- `ultralytics/utils/loss.py` - All loss functions
- `ultralytics/utils/tal.py` - Task-aligned assignment

### Inference Pipeline
- `ultralytics/utils/nms.py` - Already supported (no changes needed)
- `ultralytics/models/yolo/detect/predict.py` - Enable multi_label in NMS

### Validation Pipeline
- `ultralytics/models/yolo/detect/val.py` - Detection validator
- `ultralytics/engine/validator.py` - Base validator
- `ultralytics/models/yolo/yoloe/val.py` - Fixed sample counting bug

## Testing Recommendations

1. **Single-Label Regression Test**: Verify existing single-label datasets still work
2. **Multi-Label Training**: Test with synthetic multi-label dataset
3. **Mixed Batches**: Test with mix of single and multi-label instances
4. **Edge Cases**: 
   - All classes inactive (should error)
   - All classes active
   - Single active class (should match single-label behavior)

## Performance Considerations

- **Memory**: Multi-label format increases label storage from `(N, 1)` to `(N, C)`
- **Computation**: Minimal overhead in loss/assignment (linear in C)
- **NMS**: Multi-label NMS can produce more detections per image

## Future Enhancements

1. **Flexible Label Format**: Support comma-separated class lists in label files
2. **Class Weights**: Per-class weights for multi-label BCE loss
3. **Metrics**: Multi-label specific metrics (Hamming loss, etc.)
4. **Visualization**: Better visualization for multi-label predictions

## Conclusion

This implementation provides comprehensive, backward-compatible multi-label support across the entire YOLO pipeline. All major components (data loading, training, validation, inference) now support both single-label and multi-label formats seamlessly.
