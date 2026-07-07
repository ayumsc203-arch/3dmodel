#!/usr/bin/env python3
"""
Launcher and Setup Script for the Hand Tracking 3D VFX System.
Provides validation of dependencies, initialization of folders, and starting the system.
"""

import os
import sys
import argparse
import logging
import importlib
from pathlib import Path

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Launcher")

# List of required external packages
REQUIRED_PACKAGES = {
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "moderngl": "moderngl",
    "moderngl_window": "moderngl-window",
    "dearpygui.dearpygui": "dearpygui",
    "glm": "PyGLM",
    "yaml": "pyyaml",
    "numpy": "numpy"
}

# Directories that need to exist
PROJECT_DIRS = [
    "config",
    "src",
    "src/camera",
    "src/tracking",
    "src/renderer",
    "src/renderer/shaders",
    "src/assets",
    "src/assets/models",
    "src/particles",
    "src/effects",
    "src/ui",
    "tests"
]


def check_dependencies() -> bool:
    """Checks if all required python dependencies are installed."""
    logger.info("Checking system dependencies...")
    missing_packages = []
    
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
            logger.info(f"  [OK] Found dependency: {package_name}")
        except ImportError:
            logger.error(f"  [MISSING] Dependency: {package_name}")
            missing_packages.append(package_name)
            
    if missing_packages:
        logger.warning(
            "Some dependencies are missing! You can install them using:\n"
            f"  pip install -r requirements.txt"
        )
        return False
        
    logger.info("All dependencies verified successfully.")
    return True


def create_directory_structure() -> None:
    """Creates the necessary folder structure for the project."""
    logger.info("Verifying project folder structure...")
    base_dir = Path(__file__).parent.resolve()
    
    for relative_path in PROJECT_DIRS:
        dir_path = base_dir / relative_path
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  [CREATED] Directory: {relative_path}")
        else:
            logger.info(f"  [EXISTS] Directory: {relative_path}")
            
    # Create empty __init__.py files where needed for modules
    module_roots = ["src", "src/camera", "src/tracking", "src/renderer", "src/assets", "src/particles", "src/effects", "src/ui", "tests"]
    for relative_path in module_roots:
        init_file = base_dir / relative_path / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            logger.info(f"  [CREATED] __init__.py in {relative_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand Tracking 3D VFX System Launcher")
    parser.add_argument(
        "--setup", 
        action="store_true", 
        help="Run system check, create directory layout, and verify environment."
    )
    parser.add_argument(
        "--run", 
        action="store_true", 
        help="Start the application (starts dummy loop in Module 1)."
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, default to setup first
    if not (args.setup or args.run):
        parser.print_help()
        sys.exit(0)
        
    if args.setup:
        logger.info("--- Starting System Setup and Verification ---")
        create_directory_structure()
        check_dependencies()
        logger.info("--- Setup Verification Completed ---")
        
    if args.run:
        logger.info("--- Booting Hand Tracking 3D VFX System ---")
        
        # Check dependencies first
        if not check_dependencies():
            logger.critical("Cannot run application because of missing dependencies.")
            sys.exit(1)
            
        # Verify directories exist
        create_directory_structure()
        
        # Check configuration
        config_path = Path(__file__).parent.resolve() / "config" / "settings.yaml"
        if not config_path.exists():
            logger.critical(f"Configuration file not found at: {config_path}")
            sys.exit(1)
            
        logger.info(f"Loaded configuration from: {config_path}")
        
        import yaml
        import cv2
        from src.camera.camera_source import CameraSource
        from src.tracking.hand_tracker import HandTracker
        from src.tracking.gesture_detector import GestureDetector

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        # Initialize Camera System (Module 2)
        cam_conf = config.get("camera", {})
        cam = CameraSource(
            device_index=cam_conf.get("device_index", 0),
            width=cam_conf.get("width", 1280),
            height=cam_conf.get("height", 720),
            target_fps=cam_conf.get("fps", 60),
            mirror=cam_conf.get("mirror", True)
        )
        
        if not cam.initialize_capture():
            logger.critical("Failed to initialize camera source.")
            sys.exit(1)
            
        cam.start()

        # Initialize Hand Tracker (Module 3)
        track_conf = config.get("tracking", {})
        tracker = HandTracker(
            max_num_hands=track_conf.get("max_num_hands", 2),
            min_detection_confidence=track_conf.get("min_detection_confidence", 0.7),
            min_tracking_confidence=track_conf.get("min_tracking_confidence", 0.7),
            smoothing_enabled=track_conf.get("smoothing", {}).get("enabled", True),
            filter_min_cutoff=track_conf.get("smoothing", {}).get("min_cutoff", 0.8),
            filter_beta=track_conf.get("smoothing", {}).get("beta", 0.02)
        )

        # Initialize Gesture Detector (Module 4)
        gesture_detector = GestureDetector()
        
        logger.info("Press 'ESC' or 'q' in the camera window to exit.")
        try:
            while True:
                ret, frame = cam.read()
                if ret and frame is not None:
                    # Process hand tracking
                    hands = tracker.process_frame(frame, cam.camera_matrix)
                    for hand in hands:
                        # Draw skeletal overlays
                        tracker.draw_skeleton(frame, hand)
                        
                        # Process gestures
                        gesture = gesture_detector.update(hand)
                        
                        # Draw overlay statistics and active gesture near the wrist joint
                        wrist_pos = hand.landmarks_2d[0]
                        cv2.putText(
                            frame,
                            f"Gesture: {gesture.value.upper()}",
                            (int(wrist_pos[0]), int(wrist_pos[1]) - 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),  # Cyan
                            2,
                            cv2.LINE_AA
                        )
                        
                        cv2.putText(
                            frame,
                            f"{hand.label} (Scale: {hand.scale:.2f}, Z: {hand.palm_center[2]:.2f}m)",
                            (int(wrist_pos[0]), int(wrist_pos[1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

                    # Draw FPS overlay
                    fps = cam.get_fps()
                    cv2.putText(
                        frame, 
                        f"Camera FPS: {fps:.1f}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2, 
                        cv2.LINE_AA
                    )
                    cv2.imshow("Hand Tracking VFX System - Camera Debug View", frame)
                    
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            tracker.release()
            cam.stop()
            cv2.destroyAllWindows()
            logger.info("System shut down cleanly.")


if __name__ == "__main__":
    main()
