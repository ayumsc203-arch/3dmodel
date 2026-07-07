import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.camera.camera_source import CameraSource


class TestCameraSource(unittest.TestCase):
    def test_initialization(self):
        """Test that CameraSource initializes properties correctly."""
        cam = CameraSource(device_index=1, width=640, height=480, target_fps=30, mirror=False)
        self.assertEqual(cam.device_index, 1)
        self.assertEqual(cam.target_width, 640)
        self.assertEqual(cam.target_height, 480)
        self.assertEqual(cam.target_fps, 30)
        self.assertFalse(cam.mirror)
        self.assertFalse(cam.running)
        self.assertIsNone(cam.frame)

    def test_calculate_intrinsics(self):
        """Test camera matrix calibration calculations."""
        cam = CameraSource(width=1280, height=720)
        cam.width = 1280
        cam.height = 720
        cam._calculate_intrinsics()

        self.assertIsNotNone(cam.camera_matrix)
        self.assertEqual(cam.camera_matrix.shape, (3, 3))
        # fx should equal fy
        self.assertEqual(cam.camera_matrix[0, 0], cam.camera_matrix[1, 1])
        # Principal points should be width/2, height/2
        self.assertEqual(cam.camera_matrix[0, 2], 640.0)
        self.assertEqual(cam.camera_matrix[1, 2], 360.0)

    @patch('src.camera.camera_source.cv2.VideoCapture')
    def test_mock_capture_loop(self, mock_video_capture):
        """Test thread-based capture loop using mocked cv2.VideoCapture."""
        # Setup mock instance
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Simulate successful read returning a dummy frame (100x100 RGB)
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap.get.side_effect = lambda prop: {
            2: 1280.0,  # Width
            3: 720.0,   # Height
            5: 60.0     # FPS
        }.get(prop, 0.0)

        cam = CameraSource(width=1280, height=720, target_fps=60, mirror=True)
        
        # Test initialization
        success = cam.initialize_capture()
        self.assertTrue(success)
        mock_video_capture.assert_called_once()
        
        # Start capture thread
        cam.start()
        self.assertTrue(cam.running)
        
        # Wait briefly for thread to run at least one loop iteration
        import time
        time.sleep(0.1)
        
        # Verify a frame was successfully read and mirrored
        read_success, frame = cam.read()
        self.assertTrue(read_success)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (100, 100, 3))
        
        # Stop thread
        cam.stop()
        self.assertFalse(cam.running)


if __name__ == "__main__":
    unittest.main()
