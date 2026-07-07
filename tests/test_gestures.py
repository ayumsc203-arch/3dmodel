import unittest
import numpy as np
from src.tracking.hand_tracker import HandData
from src.tracking.gesture_detector import GestureDetector, Gesture


class TestGestureDetector(unittest.TestCase):
    def setUp(self):
        self.detector = GestureDetector(swipe_threshold=1.0)
        
    def _create_mock_hand(self, landmarks_3d: np.ndarray, label: str = "Right") -> HandData:
        """Helper to build a HandData object with mock values."""
        return HandData(
            label=label,
            landmarks_2d=np.zeros((21, 2), dtype=np.float32),
            landmarks_3d=landmarks_3d,
            palm_center=np.mean(landmarks_3d[[0, 5, 9, 17]], axis=0),
            rotation_matrix=np.eye(3, dtype=np.float32),
            scale=1.0
        )

    def test_open_palm(self):
        """Test open palm recognition (all joints extended)."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        landmarks[0] = [0.0, 0.0, 0.0]  # Wrist
        landmarks[17] = [0.05, 0.05, 0.0]  # Pinky MCP
        
        # Thumb tip extended far from pinky MCP
        landmarks[2] = [0.01, 0.01, 0.0]
        landmarks[4] = [0.08, 0.08, 0.0] 
        
        # Fingers: MCP=5+(f-1)*4, PIP=6+(f-1)*4, TIP=8+(f-1)*4
        for f in range(1, 5):
            mcp = 5 + (f - 1) * 4
            pip = 6 + (f - 1) * 4
            tip = 8 + (f - 1) * 4
            landmarks[mcp] = [0.0, 0.02 * f, 0.0]
            landmarks[pip] = [0.0, 0.04 * f, 0.0]
            landmarks[tip] = [0.0, 0.08 * f, 0.0]  # Tip is further than PIP
            
        hand = self._create_mock_hand(landmarks)
        gesture = self.detector.update(hand)
        self.assertEqual(gesture, Gesture.OPEN_PALM)

    def test_closed_fist(self):
        """Test closed fist recognition (all joints folded)."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        landmarks[0] = [0.0, 0.0, 0.0]
        landmarks[17] = [0.05, 0.05, 0.0]
        
        # Thumb folded near pinky MCP
        landmarks[2] = [0.01, 0.01, 0.0]
        landmarks[4] = [0.01, 0.01, 0.0]
        
        for f in range(1, 5):
            mcp = 5 + (f - 1) * 4
            pip = 6 + (f - 1) * 4
            tip = 8 + (f - 1) * 4
            landmarks[mcp] = [0.0, 0.05 * f, 0.0]
            landmarks[pip] = [0.0, 0.06 * f, 0.0]
            landmarks[tip] = [0.0, 0.03 * f, 0.0]  # Tip is folded back towards Wrist (closer than PIP)
            
        hand = self._create_mock_hand(landmarks)
        gesture = self.detector.update(hand)
        self.assertEqual(gesture, Gesture.CLOSED_FIST)

    def test_pointing(self):
        """Test pointing recognition (only index finger extended)."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        landmarks[0] = [0.0, 0.0, 0.0]
        landmarks[17] = [0.05, 0.05, 0.0]
        
        # Folded thumb
        landmarks[2] = [0.01, 0.01, 0.0]
        landmarks[4] = [0.01, 0.01, 0.0]

        # Index extended (f=1)
        landmarks[5] = [0.0, 0.02, 0.0]
        landmarks[6] = [0.0, 0.04, 0.0]  # PIP
        landmarks[8] = [0.0, 0.08, 0.0]  # TIP
        
        # Middle, Ring, Pinky folded (f=2,3,4)
        for f in [2, 3, 4]:
            mcp = 5 + (f - 1) * 4
            pip = 6 + (f - 1) * 4
            tip = 8 + (f - 1) * 4
            landmarks[mcp] = [0.0, 0.05 * f, 0.0]
            landmarks[pip] = [0.0, 0.06 * f, 0.0]
            landmarks[tip] = [0.0, 0.03 * f, 0.0]  # Tip closer than PIP
            
        hand = self._create_mock_hand(landmarks)
        gesture = self.detector.update(hand)
        self.assertEqual(gesture, Gesture.POINTING)

    def test_peace_sign(self):
        """Test peace sign recognition (Index and Middle fingers extended)."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        landmarks[0] = [0.0, 0.0, 0.0]
        landmarks[17] = [0.05, 0.05, 0.0]
        
        # Folded thumb
        landmarks[2] = [0.01, 0.01, 0.0]
        landmarks[4] = [0.01, 0.01, 0.0]

        # Index extended (f=1)
        landmarks[5] = [0.0, 0.02, 0.0]
        landmarks[6] = [0.0, 0.04, 0.0]
        landmarks[8] = [0.0, 0.08, 0.0]
        
        # Middle extended (f=2)
        landmarks[9] = [0.0, 0.02, 0.0]
        landmarks[10] = [0.0, 0.04, 0.0]
        landmarks[12] = [0.0, 0.08, 0.0]
        
        # Ring and Pinky folded (f=3,4)
        for f in [3, 4]:
            mcp = 5 + (f - 1) * 4
            pip = 6 + (f - 1) * 4
            tip = 8 + (f - 1) * 4
            landmarks[mcp] = [0.0, 0.05 * f, 0.0]
            landmarks[pip] = [0.0, 0.06 * f, 0.0]
            landmarks[tip] = [0.0, 0.03 * f, 0.0]
            
        hand = self._create_mock_hand(landmarks)
        gesture = self.detector.update(hand)
        self.assertEqual(gesture, Gesture.PEACE)

    def test_pinch(self):
        """Test pinch recognition (Thumb TIP and Index TIP are close)."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        landmarks[0] = [0.0, 0.0, 0.0]
        landmarks[17] = [0.05, 0.05, 0.0]
        
        # Thumb TIP (4) and Index TIP (8) placed at the exact same location
        landmarks[4] = [0.02, 0.02, 0.0]
        landmarks[8] = [0.02, 0.02, 0.0]
        
        # Index PIP (6) placed further away so Index is not folded
        landmarks[6] = [0.01, 0.01, 0.0]
        
        # Keep other fingers extended to prevent a closed fist trigger
        for f in [2, 3, 4]:
            mcp = 5 + (f - 1) * 4
            pip = 6 + (f - 1) * 4
            tip = 8 + (f - 1) * 4
            landmarks[mcp] = [0.0, 0.02 * f, 0.0]
            landmarks[pip] = [0.0, 0.04 * f, 0.0]
            landmarks[tip] = [0.0, 0.08 * f, 0.0]

        hand = self._create_mock_hand(landmarks)
        gesture = self.detector.update(hand)
        self.assertEqual(gesture, Gesture.PINCH)

    def test_swipe_detection(self):
        """Test swiping velocity calculations."""
        # Initialize an open palm structure for simulated hand
        landmarks = np.zeros((21, 3), dtype=np.float32)
        landmarks[0] = [0.0, 0.0, 0.0]
        landmarks[17] = [0.05, 0.05, 0.0]
        landmarks[4] = [0.08, 0.08, 0.0]
        for f in range(1, 5):
            landmarks[8 + (f-1)*4] = [0.0, 0.08 * f, 0.0]

        hand1 = self._create_mock_hand(landmarks)
        # Position 1 at origin
        hand1 = hand1._replace(palm_center=np.array([0.0, 0.0, 0.0]))
        
        hand2 = self._create_mock_hand(landmarks)
        # Position 2 offset right (0.2m)
        hand2 = hand2._replace(palm_center=np.array([0.2, 0.0, 0.0]))

        # First update (sets initial time)
        g1 = self.detector.update(hand1)
        # Should detect Open Palm
        self.assertEqual(g1, Gesture.OPEN_PALM)
        
        # Fast update (triggers swipe right)
        import time
        t_start = time.time()
        self.detector.palm_history["Right"] = [(t_start, 0.0)]
        self.detector.last_swipe_times["Right"] = 0.0  # Clear cooldown
        
        # Simulate fast update
        self.detector.palm_history["Right"].append((t_start + 0.05, 0.2))
        swipe = self.detector._detect_swipe("Right", np.array([0.2, 0.0, 0.0]), t_start + 0.05)
        self.assertEqual(swipe, Gesture.SWIPE_RIGHT)


if __name__ == "__main__":
    unittest.main()
