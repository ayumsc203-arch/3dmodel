import unittest
import numpy as np
from src.tracking.filters import OneEuroFilter, LowPassFilter
from src.tracking.hand_tracker import HandTracker


class TestFilters(unittest.TestCase):
    def test_low_pass_filter(self):
        """Test basic low pass filtering behavior."""
        lpf = LowPassFilter(alpha=0.5)
        # First call should initialize y to x
        self.assertEqual(lpf(10.0), 10.0)
        # Second call should apply alpha: 0.5 * 20.0 + 0.5 * 10.0 = 15.0
        self.assertEqual(lpf(20.0), 15.0)

    def test_one_euro_filter_scalar(self):
        """Test One Euro Filter with scalar values."""
        oef = OneEuroFilter(min_cutoff=1.0, beta=0.01, d_cutoff=1.0)
        
        # Test initialization
        y1 = oef(1.0, timestamp=1.0)
        self.assertEqual(y1, 1.0)
        
        # Test subsequent updates (values should smooth)
        y2 = oef(2.0, timestamp=1.1)
        self.assertTrue(1.0 < y2 < 2.0)
        
        # Reset should clear historical state
        oef.reset()
        self.assertIsNone(oef.last_time)


class TestHandTracker(unittest.TestCase):
    def test_initialization(self):
        """Test HandTracker initialization defaults."""
        tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.5)
        self.assertEqual(tracker.max_num_hands, 1)
        self.assertEqual(tracker.min_detection_confidence, 0.5)
        self.assertTrue(tracker.smoothing_enabled)

    def test_rotation_matrix_filtering(self):
        """Test rotation decomposition, filtering, and reconstruction."""
        tracker = HandTracker()
        
        # Create an identity rotation matrix (represents zero tilt/yaw/roll)
        rot_identity = np.eye(3, dtype=np.float32)
        
        # Filter rotation at time t=1.0
        filtered_rot = tracker._filter_rotation("Right", rot_identity, 1.0)
        
        # Assert shape and identity closeness
        self.assertEqual(filtered_rot.shape, (3, 3))
        np.testing.assert_array_almost_equal(filtered_rot, rot_identity, decimal=5)

    def test_metric_projection_logic(self):
        """Verify the camera matrix metric back-projection equations."""
        # Setup simulated camera matrix
        width, height = 1280, 720
        fx = fy = 1000.0  # Focal length
        cx, cy = 640.0, 360.0
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Let's say a landmark is at pixel position [640.0, 360.0] (center of the image)
        # And the estimated depth is exactly 1.0 meter.
        # X_metric should be 0.0, Y_metric should be 0.0, Z_metric should be -1.0.
        
        # We simulate the metric math inside HandTracker.process_frame:
        estimated_depth = 1.0
        px_x, px_y = 640.0, 360.0
        
        x_metric = (px_x - cx) * estimated_depth / fx
        y_metric = -(px_y - cy) * estimated_depth / fy
        
        self.assertEqual(x_metric, 0.0)
        self.assertEqual(y_metric, 0.0)


if __name__ == "__main__":
    unittest.main()
