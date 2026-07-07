import cv2
import numpy as np
import threading
import time
import logging
from typing import Tuple, Optional

logger = logging.getLogger("CameraSource")


class CameraSource:
    """
    Handles camera capture using OpenCV in a background thread for optimal performance.
    Provides mirror mode, thread-safe frame access, FPS measurement, and basic calibration parameters.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 60,
        mirror: bool = True
    ):
        self.device_index = device_index
        self.target_width = width
        self.target_height = height
        self.target_fps = target_fps
        self.mirror = mirror

        # OpenCV Capture Object
        self.cap: Optional[cv2.VideoCapture] = None

        # Threading control
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Thread outputs
        self.ret = False
        self.frame: Optional[np.ndarray] = None
        self.fps = 0.0

        # Calibration parameters
        self.fov = 60.0  # Estimated horizontal field of view in degrees
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

    def initialize_capture(self) -> bool:
        """Initializes the OpenCV VideoCapture object and applies properties."""
        try:
            logger.info(f"Opening camera index {self.device_index}...")
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW if os_is_windows() else cv2.CAP_ANY)
            
            if not self.cap.isOpened():
                # Fallback to standard backend if CAP_DSHOW fails
                self.cap = cv2.VideoCapture(self.device_index)
                if not self.cap.isOpened():
                    logger.error(f"Failed to open camera index {self.device_index}.")
                    return False

            # Set requested dimensions
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Retrieve actual applied dimensions (cameras might not support exact requests)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"Camera opened. Resolution: {self.width}x{self.height} at {self.actual_fps} FPS.")
            self._calculate_intrinsics()
            return True
            
        except Exception as e:
            logger.exception(f"Error initializing camera capture: {e}")
            return False

    def _calculate_intrinsics(self) -> None:
        """Calculates approximate camera intrinsics (calibration matrix) based on FOV."""
        # Simple pinhole model approximation
        cx = self.width / 2.0
        cy = self.height / 2.0
        # f = cx / tan(fov_rad / 2)
        fov_rad = np.radians(self.fov)
        fx = cx / np.tan(fov_rad / 2.0)
        fy = fx  # Assume square pixels

        self.camera_matrix = np.array(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float32
        )
        self.dist_coeffs = np.zeros(5, dtype=np.float32)  # Assume zero distortion
        logger.info(f"Calculated camera matrix: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    def start(self) -> None:
        """Starts the background thread for capturing frames."""
        if self.running:
            logger.warning("CameraSource is already running.")
            return

        if self.cap is None and not self.initialize_capture():
            logger.error("Cannot start CameraSource because capture initialization failed.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, name="CameraCaptureThread", daemon=True)
        self.thread.start()
        logger.info("Camera capture background thread started.")

    def _capture_loop(self) -> None:
        """Thread loop that continuously grabs and pre-processes frames."""
        frame_count = 0
        start_time = time.time()

        while self.running:
            if self.cap is None:
                break

            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to grab frame from camera source.")
                with self.lock:
                    self.ret = False
                time.sleep(0.01)
                continue

            # Process frame (mirroring)
            if self.mirror:
                frame = cv2.flip(frame, 1)

            # Store frame thread-safely
            with self.lock:
                self.ret = True
                self.frame = frame

            # Calculate actual capture FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # Slight sleep to match target frame rate and prevent CPU hogging
            time.sleep(1.0 / (self.target_fps * 1.5))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the latest frame in a thread-safe manner."""
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def get_fps(self) -> float:
        """Returns the current measured capture FPS."""
        return self.fps

    def stop(self) -> None:
        """Stops the background thread and releases resources."""
        logger.info("Stopping camera capture...")
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        logger.info("Camera capture stopped.")


def os_is_windows() -> bool:
    """Helper to detect if OS is Windows."""
    import platform
    return platform.system() == "Windows"
