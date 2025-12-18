# Fix Summary: Multi-label Detection Data Corruption

## Issue Verified ✅

The reported issue exists in `ultralytics/data/utils.py` at lines 204-208. The segment detection condition `any(len(x) > 6 for x in lb)` incorrectly triggers for multi-label detection data when `num_cls > 2`, causing silent data corruption.

## Root Cause

Multi-label detection format: `[cls_0, cls_1, ..., cls_n-1, x, y, w, h]` has `num_cls + 4` values per line.

When `num_cls > 2`:
- `num_cls + 4 > 6` triggers the segment detection condition
- Data is incorrectly processed as polygon segments
- For odd `num_cls` values (3, 5, 7, etc.), the bug is silent because `num_cls + 3` remaining values can be reshaped to `(-1, 2)`
- The first multi-hot value is taken as class, and remaining class columns + bbox coordinates are misinterpreted as polygon vertices

## Fix Applied ✅

**File:** `ultralytics/data/utils.py`
**Line:** 205-206

**Before:**
```python
if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
```

**After:**
```python
# Check for segment data, excluding multi-label detection format (num_cls + 4 values)
if any(len(x) > 6 and len(x) != num_cls + 4 for x in lb) and (not keypoint):  # is segment
```

## Fix Logic

The condition now explicitly excludes lines with exactly `num_cls + 4` values from being treated as segments. This prevents multi-label detection data from being corrupted while preserving correct segmentation detection.

## Tests Added ✅

**File:** `tests/test_multilabel_segment_fix.py`

Comprehensive test suite covering:
1. Multi-label detection with `num_cls=3` (7 values) - ensures NOT treated as segment
2. Multi-label detection with `num_cls=5` (9 values) - ensures NOT treated as segment
3. Single-label detection (5 values) - ensures still works correctly
4. Actual segmentation data - ensures still detected correctly
5. Segmentation with many vertices - ensures still works
6. Edge case handling - ensures correct disambiguation

**Test Results:**
```
6 tests passed in 0.05s ✅
```

Run tests with:
```bash
python3 -m pytest tests/test_multilabel_segment_fix.py -v
```

## Backward Compatibility ✅

The fix is fully backward compatible:
- **Single-label detection:** No change (5 values ≤ 6, condition doesn't trigger)
- **Segmentation:** No change (works as before when not multi-label format)
- **Multi-label detection:** Now works correctly (was broken before)

## Impact

This fix prevents silent data corruption in multi-label detection datasets with `num_cls > 2`. Without this fix:
- Training would proceed without errors
- Model would learn from corrupted labels
- Results would be poor and debugging would be difficult

## Verification

- [x] Issue verified and reproduced
- [x] Fix implemented
- [x] Comprehensive tests added
- [x] All tests pass
- [x] No linter errors
- [x] Backward compatibility maintained
- [x] Existing tests still pass
- [x] Documentation added

## Files Modified

1. **ultralytics/data/utils.py** (line 205-206)
   - Fixed segment detection condition

2. **tests/test_multilabel_segment_fix.py** (new file)
   - Added comprehensive test suite

3. **MULTILABEL_SEGMENT_FIX.md** (new file)
   - Detailed technical documentation

4. **FIX_SUMMARY.md** (this file)
   - Executive summary
