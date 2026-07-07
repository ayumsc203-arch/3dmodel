import cv2
import mediapipe as mp
import numpy as np
import time
import logging
from typing import Dict, List, NamedTuple, Optional, Tuple
from src.tracking.filters import OneEuroFilter

logger = logging.getLogger("HandTracker")


class HandData(NamedTuple):
    """Clean data structure representing tracked hand information."""
    label: str                   # "Left" or "Right"
    landmarks_2d: np.ndarray     # 21x2 shape: pixel coordinates (X, Y)
    landmarks_3d: np.ndarray     # 21x3 shape: filtered metric space coordinates (X, Y, Z) in camera frame
    palm_center: np.ndarray      # 3D coordinate (X, Y, Z) representing palm center
    rotation_matrix: np.ndarray  # 3x3 rotation matrix representing hand orientation
    scale: float                 # Calculated relative scale of the hand


class HandTracker:
    """
    Integrates MediaPipe Hands with stabilization filters.
    Computes 3D coordinates, orientation matrices, and provides dual-hand tracking.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        smoothing_enabled: bool = True,
        filter_min_cutoff: float = 0.8,
        filter_beta: float = 0.02
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smoothing_enabled = smoothing_enabled

        # One Euro Filter parameters
        self.filter_min_cutoff = filter_min_cutoff
        self.filter_beta = filter_beta

        # MediaPipe Solutions
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            model_complexity=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

        # Filters dictionary: keys are strings like "Left_0", "Right_20" representing hand side + landmark index
        self.filters: Dict[str, OneEuroFilter] = {}
        # Filters for palm center and orientation (Euler angles)
        self.palm_filters: Dict[str, OneEuroFilter] = {}

        # Estimated physical constants
        self.PHYSICAL_PALM_SIZE = 0.085  # Average distance in meters between Wrist (0) and Index MCP (5)

    def _get_filter(self, key: str) -> OneEuroFilter:
        """Retrieves or creates a One Euro Filter instance for a specific landmark/parameter."""
        if key not in self.filters:
            self.filters[key] = OneEuroFilter(
                min_cutoff=self.filter_min_cutoff,
                beta=self.filter_beta
            )
        return self.filters[key]

    def _get_palm_filter(self, key: str) -> OneEuroFilter:
        """Retrieves or creates a One Euro Filter instance for palm metrics."""
        if key not in self.palm_filters:
            self.palm_filters[key] = OneEuroFilter(
                min_cutoff=self.filter_min_cutoff * 0.5,  # Smoother orientation
                beta=self.filter_beta
            )
        return self.palm_filters[key]

    def process_frame(
        self, 
        frame: np.ndarray, 
        camera_matrix: np.ndarray
    ) -> List[HandData]:
        """
        Processes a camera frame, extracts hands, estimates 3D metric coordinates,
        applies One Euro Filters, and computes spatial transforms.
        """
        height, width, _ = frame.shape
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Disable writing on image to optimize memory performance
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        tracked_hands: List[HandData] = []

        if not results.multi_hand_landmarks or not results.multi_handedness:
            # No hands detected, reset filter states to prevent lag jumps when hands reappear
            self.reset_filters()
            return tracked_hands

        # Extract intrinsic parameters
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # MediaPipe hands label is mirrored inside standard mirror modes
            # We obtain standard labels: "Left" or "Right"
            label = handedness.classification[0].label

            # 1. Gather raw 2D pixel coordinates and depth estimate proxy
            landmarks_2d = np.zeros((21, 2), dtype=np.float32)
            for idx, lm in enumerate(hand_landmarks.landmark):
                landmarks_2d[idx] = [lm.x * width, lm.y * height]

            # Estimate depth (Z) based on physical palm size in pixels
            wrist_px = landmarks_2d[0]
            index_mcp_px = landmarks_2d[5]
            palm_dist_px = np.linalg.norm(wrist_px - index_mcp_px)
            
            # Avoid division by zero
            if palm_dist_px < 1.0:
                palm_dist_px = 1.0

            # Z = (Physical size * focal length) / distance in pixels
            estimated_depth = (self.PHYSICAL_PALM_SIZE * fx) / palm_dist_px

            # 2. Convert landmarks to 3D metric space (in meters relative to camera)
            raw_landmarks_3d = np.zeros((21, 3), dtype=np.float32)
            for idx, lm in enumerate(hand_landmarks.landmark):
                # Backproject: X = (x_px - cx) * Z / fx
                px_x = lm.x * width
                px_y = lm.y * height
                
                x_metric = (px_x - cx) * estimated_depth / fx
                y_metric = -(px_y - cy) * estimated_depth / fy  # Invert Y to match OpenGL space (up is positive)
                # MediaPipe's Z is relative and scaled similarly to X.
                # We combine our calculated base depth with MediaPipe's relative depth
                # MediaPipe's Z coordinate is scaled such that it represents relative depth in meters
                z_metric = -estimated_depth + (lm.z * self.PHYSICAL_PALM_SIZE)
                
                raw_landmarks_3d[idx] = [x_metric, y_metric, z_metric]

            # 3. Apply One Euro filtering to 3D landmarks
            t = time.time()
            landmarks_3d = np.zeros((21, 3), dtype=np.float32)
            for idx in range(21):
                raw_coord = raw_landmarks_3d[idx]
                if self.smoothing_enabled:
                    filt_x = self._get_filter(f"{label}_{idx}_x")(raw_coord[0], t)
                    filt_y = self._get_filter(f"{label}_{idx}_y")(raw_coord[1], t)
                    filt_z = self._get_filter(f"{label}_{idx}_z")(raw_coord[2], t)
                    landmarks_3d[idx] = [filt_x, filt_y, filt_z]
                else:
                    landmarks_3d[idx] = raw_coord

            # 4. Calculate Palm orientation frame (Wrist(0), Index MCP(5), Pinky MCP(17))
            wrist_3d = landmarks_3d[0]
            index_mcp_3d = landmarks_3d[5]
            pinky_mcp_3d = landmarks_3d[17]

            # Local coordinates vectors
            u = index_mcp_3d - wrist_3d
            v = pinky_mcp_3d - wrist_3d

            # Orthonormalization (Gram-Schmidt)
            u_norm = u / np.linalg.norm(u)
            # Normal vector pointing out of palm (cross product)
            # We adapt cross product order for left vs right hand orientation alignment
            if label == "Right":
                normal = np.cross(u_norm, v)
            else:
                normal = np.cross(v, u_norm)
                
            normal_norm = normal / np.linalg.norm(normal)
            
            # Tangent direction (orthogonal to u and normal)
            tangent = np.cross(normal_norm, u_norm)
            tangent_norm = tangent / np.linalg.norm(tangent)

            # Rotation matrix: columns are X (tangent), Y (up/u_norm), Z (normal)
            raw_rot_matrix = np.column_stack((tangent_norm, u_norm, normal_norm))

            # Smooth rotation matrix by converting to Euler angles, filtering, and converting back
            rot_matrix = self._filter_rotation(label, raw_rot_matrix, t)

            # Palm center calculation (centroid of wrist, index, middle, and pinky MCPs)
            middle_mcp_3d = landmarks_3d[9]
            raw_palm_center = (wrist_3d + index_mcp_3d + middle_mcp_3d + pinky_mcp_3d) / 4.0
            
            if self.smoothing_enabled:
                filt_cx = self._get_palm_filter(f"{label}_center_x")(raw_palm_center[0], t)
                filt_cy = self._get_palm_filter(f"{label}_center_y")(raw_palm_center[1], t)
                filt_cz = self._get_palm_filter(f"{label}_center_z")(raw_palm_center[2], t)
                palm_center = np.array([filt_cx, filt_cy, filt_cz], dtype=np.float32)
            else:
                palm_center = raw_palm_center

            # Scale estimation relative to target palm size (for dynamic model scaling)
            current_scale = np.linalg.norm(index_mcp_3d - wrist_3d) / self.PHYSICAL_PALM_SIZE

            tracked_hands.append(
                HandData(
                    label=label,
                    landmarks_2d=landmarks_2d,
                    landmarks_3d=landmarks_3d,
                    palm_center=palm_center,
                    rotation_matrix=rot_matrix,
                    scale=current_scale
                )
            )

        return tracked_hands

    def _filter_rotation(self, label: str, rot_mat: np.ndarray, t: float) -> np.ndarray:
        """Extracts Euler angles, applies One Euro filter, and reconstructs rotation matrix."""
        # Deconstruct rotation matrix to Euler angles (yaw, pitch, roll)
        # R = Rz(roll) * Ry(pitch) * Rx(yaw)
        pitch = np.arcsin(-rot_mat[2, 1])
        if np.abs(np.cos(pitch)) > 1e-6:
            yaw = np.arctan2(rot_mat[2, 0], rot_mat[2, 2])
            roll = np.arctan2(rot_mat[0, 1], rot_mat[1, 1])
        else:
            yaw = np.arctan2(-rot_mat[0, 2], rot_mat[0, 0])
            roll = 0.0

        if self.smoothing_enabled:
            filt_yaw = self._get_palm_filter(f"{label}_rot_yaw")(yaw, t)
            filt_pitch = self._get_palm_filter(f"{label}_rot_pitch")(pitch, t)
            filt_roll = self._get_palm_filter(f"{label}_rot_roll")(roll, t)
        else:
            filt_yaw, filt_pitch, filt_roll = yaw, pitch, roll

        # Reconstruct rotation matrix
        # Rx
        rx = np.array([
            [1, 0, 0],
            [0, np.cos(filt_yaw), -np.sin(filt_yaw)],
            [0, np.sin(filt_yaw), np.cos(filt_yaw)]
        ])
        # Ry
        ry = np.array([
            [np.cos(filt_pitch), 0, np.sin(filt_pitch)],
            [0, 1, 0],
            [-np.sin(filt_pitch), 0, np.cos(filt_pitch)]
        ])
        # Rz
        rz = np.array([
            [np.cos(filt_roll), -np.sin(filt_roll), 0],
            [np.sin(filt_roll), np.cos(filt_roll), 0],
            [0, 0, 1]
        ])

        # R = Rz * Ry * Rx
        return rz @ ry @ rx

    def draw_skeleton(self, frame: np.ndarray, hand_data: HandData) -> None:
        """Helper to draw stabilized skeleton lines and joint points on a frame."""
        # Convert landmarks back to 2D tuples
        points = [tuple(pt.astype(int)) for pt in hand_data.landmarks_2d]
        
        # Color coding: Green for Right hand, Red for Left hand
        color = (0, 255, 0) if hand_data.label == "Right" else (0, 0, 255)
        
        # Draw connections
        for connection in self.mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx], color, 2)
                
        # Draw joints
        for pt in points:
            cv2.circle(frame, pt, 5, (255, 255, 255), -1)
            cv2.circle(frame, pt, 5, color, 1)

    def reset_filters(self) -> None:
        """Resets all active One Euro Filter states."""
        for filt in self.filters.values():
            filt.reset()
        for filt in self.palm_filters.values():
            filt.reset()

    def release(self) -> None:
        """Clean up MediaPipe resources."""
        self.hands.close()
