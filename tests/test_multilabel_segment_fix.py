"""
Test for the multi-label detection vs segmentation data fix.

This test verifies that multi-label detection data with num_cls > 2 is not
incorrectly treated as segmentation data.

Issue: The segment detection condition `any(len(x) > 6 for x in lb)` triggers
for multi-label detection data when `num_cls > 2`, since multi-label format
has `num_cls + 4` values per line.
"""

import tempfile
import os
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from ultralytics.data.utils import verify_image_label


def create_test_image():
    """Create a simple test image."""
    img = Image.new('RGB', (640, 480), color='red')
    return img


def test_multilabel_not_treated_as_segment():
    """Test that multi-label detection data is not treated as segmentation data."""
    with tempfile.TemporaryDirectory() as test_dir:
        # Create test image
        img_path = os.path.join(test_dir, "test.jpg")
        img = create_test_image()
        img.save(img_path)
        
        # Test case 1: Multi-label detection with num_cls=3 (7 values total)
        label_path = os.path.join(test_dir, "test.txt")
        with open(label_path, 'w') as f:
            # Multi-hot encoding: [1, 0, 1, x, y, w, h]
            f.write("1 0 1 0.5 0.5 0.3 0.4\n")
            f.write("0 1 0 0.2 0.3 0.2 0.2\n")
        
        args = (img_path, label_path, "", False, 3, 0, 2, False)
        result = verify_image_label(args)
        im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg = result
        
        # Verify that segments list is empty (not treated as segment data)
        assert not segments, "Multi-label data with num_cls=3 should NOT be treated as segments"
        
        # Verify the label shape is correct (num_cls + 4 = 7 columns)
        assert lb.shape[1] == 7, f"Expected 7 columns, got {lb.shape[1]}"
        
        # Verify multi-hot encoding is preserved
        assert np.allclose(lb[0, :3], [1, 0, 1]), "First row class encoding corrupted"
        assert np.allclose(lb[1, :3], [0, 1, 0]), "Second row class encoding corrupted"


def test_multilabel_num_cls_5():
    """Test multi-label detection with num_cls=5 (9 values total)."""
    with tempfile.TemporaryDirectory() as test_dir:
        # Create test image
        img_path = os.path.join(test_dir, "test.jpg")
        img = create_test_image()
        img.save(img_path)
        
        # Multi-label with num_cls=5
        label_path = os.path.join(test_dir, "test.txt")
        with open(label_path, 'w') as f:
            # Multi-hot encoding: [1, 0, 1, 0, 1, x, y, w, h]
            f.write("1 0 1 0 1 0.5 0.5 0.3 0.4\n")
        
        args = (img_path, label_path, "", False, 5, 0, 2, False)
        result = verify_image_label(args)
        im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg = result
        
        # Verify not treated as segment
        assert not segments, "Multi-label data with num_cls=5 should NOT be treated as segments"
        
        # Verify shape
        assert lb.shape[1] == 9, f"Expected 9 columns, got {lb.shape[1]}"
        
        # Verify encoding
        assert np.allclose(lb[0, :5], [1, 0, 1, 0, 1]), "Class encoding corrupted"


def test_single_label_still_works():
    """Test that single-label detection (5 values) still works correctly."""
    with tempfile.TemporaryDirectory() as test_dir:
        # Create test image
        img_path = os.path.join(test_dir, "test.jpg")
        img = create_test_image()
        img.save(img_path)
        
        # Single-label format
        label_path = os.path.join(test_dir, "test.txt")
        with open(label_path, 'w') as f:
            f.write("0 0.5 0.5 0.3 0.4\n")
            f.write("1 0.2 0.3 0.2 0.2\n")
        
        args = (img_path, label_path, "", False, 80, 0, 2, False)
        result = verify_image_label(args)
        im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg = result
        
        # Verify not treated as segment (5 values is not > 6)
        assert not segments, "Single-label data should NOT be treated as segments"
        
        # Verify shape
        assert lb.shape[1] == 5, f"Expected 5 columns, got {lb.shape[1]}"


def test_actual_segment_still_detected():
    """Test that actual segmentation data is still correctly detected."""
    with tempfile.TemporaryDirectory() as test_dir:
        # Create test image
        img_path = os.path.join(test_dir, "test.jpg")
        img = create_test_image()
        img.save(img_path)
        
        # Segmentation format: [cls, x1, y1, x2, y2, x3, y3, x4, y4]
        label_path = os.path.join(test_dir, "test.txt")
        with open(label_path, 'w') as f:
            # 4-vertex polygon (9 values: 1 class + 8 coordinates)
            f.write("0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n")
        
        # Using num_cls=80 (COCO), so 80+4=84 != 9, will be treated as segment
        args = (img_path, label_path, "", False, 80, 0, 2, False)
        result = verify_image_label(args)
        im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg = result
        
        # Verify it IS treated as segment
        assert segments, "Actual segment data should be detected as segments"
        assert len(segments) == 1, "Should have 1 segment"
        
        # Verify converted to bbox format (5 columns)
        assert lb.shape[1] == 5, f"Segment should be converted to bbox (5 columns), got {lb.shape[1]}"


def test_segment_with_many_vertices():
    """Test segmentation with many vertices (more than 8 coordinates)."""
    with tempfile.TemporaryDirectory() as test_dir:
        # Create test image
        img_path = os.path.join(test_dir, "test.jpg")
        img = create_test_image()
        img.save(img_path)
        
        # Segmentation with 6 vertices (13 values: 1 class + 12 coordinates)
        label_path = os.path.join(test_dir, "test.txt")
        with open(label_path, 'w') as f:
            f.write("0 0.1 0.1 0.3 0.1 0.5 0.2 0.5 0.4 0.3 0.5 0.1 0.4\n")
        
        # Using num_cls=80, so 80+4=84 != 13, will be treated as segment
        args = (img_path, label_path, "", False, 80, 0, 2, False)
        result = verify_image_label(args)
        im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg = result
        
        # Verify it IS treated as segment
        assert segments, "Segment with many vertices should be detected"
        
        # Verify converted to bbox format
        assert lb.shape[1] == 5, "Should be converted to bbox format"


def test_edge_case_multilabel_vs_segment():
    """
    Test edge case where multi-label with num_cls=5 (9 values) could be
    confused with a 4-vertex polygon (also 9 values).
    
    With the fix, when num_cls=5, a line with 9 values will be treated as
    multi-label (not segment) because 9 == num_cls + 4.
    """
    with tempfile.TemporaryDirectory() as test_dir:
        # Create test image
        img_path = os.path.join(test_dir, "test.jpg")
        img = create_test_image()
        img.save(img_path)
        
        # Ambiguous case: 9 values
        label_path = os.path.join(test_dir, "test.txt")
        with open(label_path, 'w') as f:
            # These values are formatted as multi-label with binary values
            f.write("1 0 1 0 1 0.5 0.5 0.3 0.4\n")
        
        # Test with num_cls=5: should be treated as multi-label
        args = (img_path, label_path, "", False, 5, 0, 2, False)
        result = verify_image_label(args)
        im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg = result
        
        assert not segments, "With num_cls=5, 9 values should be multi-label, not segment"
        assert lb.shape[1] == 9, "Should preserve multi-label format"
        
        # Now test with different num_cls where 9 != num_cls+4: should be treated as segment
        # For example, num_cls=3 means multi-label would be 7 values, so 9 values = segment
        with open(label_path, 'w') as f:
            # Write valid polygon coordinates
            f.write("0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n")
        
        args = (img_path, label_path, "", False, 3, 0, 2, False)
        result = verify_image_label(args)
        im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg = result
        
        assert segments, "With num_cls=3, 9 values should be segment"
        assert lb.shape[1] == 5, "Should be converted to bbox format"


if __name__ == "__main__":
    # Run tests
    print("Running multi-label vs segment fix tests...\n")
    
    test_multilabel_not_treated_as_segment()
    print("✅ Test 1 passed: Multi-label (num_cls=3) not treated as segment")
    
    test_multilabel_num_cls_5()
    print("✅ Test 2 passed: Multi-label (num_cls=5) not treated as segment")
    
    test_single_label_still_works()
    print("✅ Test 3 passed: Single-label detection still works")
    
    test_actual_segment_still_detected()
    print("✅ Test 4 passed: Actual segmentation data still detected")
    
    test_segment_with_many_vertices()
    print("✅ Test 5 passed: Segmentation with many vertices works")
    
    test_edge_case_multilabel_vs_segment()
    print("✅ Test 6 passed: Edge case handled correctly")
    
    print("\n" + "="*70)
    print("All tests passed! ✅")
    print("="*70)
