import numpy as np
import time
from enum import Enum
from typing import Dict, List, Optional
from src.tracking.hand_tracker import HandData


class Gesture(Enum):
    UNKNOWN = "unknown"
    PINCH = "pinch"
    OPEN_PALM = "open_palm"
    CLOSED_FIST = "closed_fist"
    POINTING = "pointing"
    PEACE = "peace"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"


class GestureDetector:
    """
    Analyzes HandData landmarks to recognize gestures.
    Maintains a gesture state machine with temporal smoothing and cooldowns.
    """

    def __init__(self, swipe_threshold: float = 1.0, swipe_cooldown: float = 0.6):
        self.swipe_threshold = swipe_threshold  # Velocity threshold in m/s
        self.swipe_cooldown = swipe_cooldown    # Cooldown duration in seconds

        # Hand states history: keys are hand labels ("Left", "Right")
        self.active_gestures: Dict[str, Gesture] = {"Left": Gesture.UNKNOWN, "Right": Gesture.UNKNOWN}
        self.last_gesture_times: Dict[str, float] = {"Left": 0.0, "Right": 0.0}
        
        # Swiping history: list of tuples (timestamp, palm_center_x)
        self.palm_history: Dict[str, List[tuple[float, float]]] = {"Left": [], "Right": []}
        self.last_swipe_times: Dict[str, float] = {"Left": 0.0, "Right": 0.0}

    def _is_finger_extended(self, hand_data: HandData, finger_idx: int) -> bool:
        """
        Determines if a specific finger is extended.
        Uses distance from the wrist compared to intermediate joints to ensure rotation invariance.
        Fingers: 1=Index, 2=Middle, 3=Ring, 4=Pinky
        """
        landmarks = hand_data.landmarks_3d
        wrist = landmarks[0]

        # Map finger index to PIP and TIP landmarks
        # Index: PIP=6, TIP=8
        # Middle: PIP=10, TIP=12
        # Ring: PIP=14, TIP=16
        # Pinky: PIP=18, TIP=20
        pip_idx = 2 + finger_idx * 4
        tip_idx = 4 + finger_idx * 4

        d_tip = np.linalg.norm(landmarks[tip_idx] - wrist)
        d_pip = np.linalg.norm(landmarks[pip_idx] - wrist)

        # Extended if the tip is further from the wrist than the PIP joint
        return d_tip > d_pip

    def _is_thumb_extended(self, hand_data: HandData) -> bool:
        """Determines if the thumb is extended based on its distance to the wrist."""
        landmarks = hand_data.landmarks_3d
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        
        # Extended if thumb tip is further from the wrist than the thumb base (MCP)
        return np.linalg.norm(thumb_tip - wrist) > np.linalg.norm(thumb_mcp - wrist)

    def _detect_swipe(self, label: str, palm_center: np.ndarray, current_time: float) -> Optional[Gesture]:
        """Tracks palm center history to detect high-speed swiping gestures."""
        history = self.palm_history[label]
        
        # Append current position
        history.append((current_time, palm_center[0]))
        
        # Prune old history (keep only last 0.2 seconds)
        history = [pt for pt in history if current_time - pt[0] < 0.2]
        self.palm_history[label] = history

        # Check cooldown
        if current_time - self.last_swipe_times[label] < self.swipe_cooldown:
            return None

        if len(history) < 3:
            return None

        # Calculate average velocity: delta_x / delta_t
        dt = history[-1][0] - history[0][0]
        if dt > 0.03:  # Minimum time window to avoid division by noisy time increments
            dx = history[-1][1] - history[0][1]
            velocity = dx / dt

            if abs(velocity) > self.swipe_threshold:
                self.last_swipe_times[label] = current_time
                self.palm_history[label] = []  # Clear history after trigger
                if velocity > 0.0:
                    # Moving right in OpenGL space
                    return Gesture.SWIPE_RIGHT
                else:
                    return Gesture.SWIPE_LEFT

        return None

    def update(self, hand_data: HandData) -> Gesture:
        """
        Processes hand data, evaluates gestures, runs the state machine,
        and returns the active or newly triggered gesture.
        """
        label = hand_data.label
        current_time = time.time()

        # 1. Check for transient Swipe gesture first (velocity-based)
        swipe = self._detect_swipe(label, hand_data.palm_center, current_time)
        if swipe is not None:
            self.active_gestures[label] = swipe
            self.last_gesture_times[label] = current_time
            return swipe

        # If a transient swipe was triggered recently, keep it active for 0.25 seconds for visual feedback
        if (self.active_gestures[label] in [Gesture.SWIPE_LEFT, Gesture.SWIPE_RIGHT] and 
                current_time - self.last_gesture_times[label] < 0.25):
            return self.active_gestures[label]

        # 2. Evaluate static finger states
        thumb_ext = self._is_thumb_extended(hand_data)
        index_ext = self._is_finger_extended(hand_data, 1)
        middle_ext = self._is_finger_extended(hand_data, 2)
        ring_ext = self._is_finger_extended(hand_data, 3)
        pinky_ext = self._is_finger_extended(hand_data, 4)

        # 3. Check for Pinch (Thumb TIP + Index TIP close together in meters)
        thumb_tip = hand_data.landmarks_3d[4]
        index_tip = hand_data.landmarks_3d[8]
        pinch_dist = np.linalg.norm(thumb_tip - index_tip)

        # Pinch threshold: ~3.5 cm (0.035m)
        is_pinching = pinch_dist < 0.035

        # 4. Resolve static gestures with prioritized rules
        detected = Gesture.UNKNOWN

        if not (index_ext or middle_ext or ring_ext or pinky_ext or thumb_ext):
            detected = Gesture.CLOSED_FIST
        elif is_pinching:
            detected = Gesture.PINCH
        elif index_ext and middle_ext and ring_ext and pinky_ext and thumb_ext:
            detected = Gesture.OPEN_PALM
        elif index_ext and middle_ext and not (ring_ext or pinky_ext):
            detected = Gesture.PEACE
        elif index_ext and not (middle_ext or ring_ext or pinky_ext):
            detected = Gesture.POINTING

        # State Machine update: apply simple temporal hysteresis (requires gesture to be stable)
        # For simplicity, we update the active gesture immediately, but static gestures don't override transient swipes
        self.active_gestures[label] = detected
        self.last_gesture_times[label] = current_time

        return detected

    def get_gesture(self, label: str) -> Gesture:
        """Returns the last active gesture for the specified hand."""
        return self.active_gestures.get(label, Gesture.UNKNOWN)
