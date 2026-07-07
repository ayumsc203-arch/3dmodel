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
        import glm
        import moderngl_window as mglw
        from src.camera.camera_source import CameraSource
        from src.tracking.hand_tracker import HandTracker
        from src.tracking.gesture_detector import GestureDetector, Gesture
        from src.renderer.engine import RendererEngine

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        # Initialize Window Settings (Module 5)
        render_conf = config.get("rendering", {})
        win_w = render_conf.get("window_width", 1280)
        win_h = render_conf.get("window_height", 720)
        vsync = render_conf.get("vsync", True)

        logger.info("Initializing ModernGL window...")
        mglw.setup_basic_logging(logger.level)
        window = mglw.create_window_from_settings({
            "class": "moderngl_window.context.pyglet.Window",
            "title": "Hand Tracking 3D VFX System - ModernGL Viewport",
            "size": (win_w, win_h),
            "fullscreen": False,
            "resizable": False,
            "vsync": vsync,
            "gl_version": (3, 3)
        })
        ctx = window.ctx

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

        # Initialize ModernGL Renderer Engine (Module 5)
        renderer = RendererEngine(ctx, win_w, win_h)
        
        logger.info("Starting ModernGL VFX loop. Press 'ESC' or close the viewport to exit.")
        try:
            while not window.is_closing:
                ret, frame = cam.read()
                if ret and frame is not None:
                    # Update webcam background texture (MediaPipe/OpenGL wants RGB)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    renderer.update_background_texture(frame_rgb)
                    
                    # Bind HDR FBO and clear with camera quad background
                    renderer.prepare_scene()
                    
                    # Track hands
                    hands = tracker.process_frame(frame, cam.camera_matrix)
                    
                    # Compute Projection Matrix from Camera Intrinsics
                    fx = cam.camera_matrix[0, 0]
                    fy = cam.camera_matrix[1, 1]
                    cx = cam.camera_matrix[0, 2]
                    cy = cam.camera_matrix[1, 2]
                    w, h = cam.width, cam.height
                    near, far = 0.05, 20.0
                    
                    proj = glm.mat4(0.0)
                    proj[0][0] = 2.0 * fx / w
                    proj[1][1] = 2.0 * fy / h
                    proj[2][0] = (w - 2.0 * cx) / w
                    proj[2][1] = (2.0 * cy - h) / h
                    proj[2][2] = -(far + near) / (far - near)
                    proj[2][3] = -1.0
                    proj[3][2] = -(2.0 * far * near) / (far - near)
                    
                    view = glm.mat4(1.0) # Identity View Matrix
                    
                    # Render 3D components for active hands
                    for hand in hands:
                        gesture = gesture_detector.update(hand)
                        
                        # Extract 3D position
                        pos = glm.vec3(hand.palm_center[0], hand.palm_center[1], hand.palm_center[2])
                        
                        # Extract Orientation Rotation Matrix
                        rot = glm.mat4(1.0)
                        for col in range(3):
                            for row in range(3):
                                rot[col][row] = hand.rotation_matrix[col][row]
                                
                        # Scale factor of hand
                        scale = hand.scale * 0.07  # ~7cm default cube scale
                        
                        # Map gestures to colors and glow emissive intensities
                        emissive = glm.vec3(0.0)
                        box_color = glm.vec3(0.0, 1.0, 1.0)  # Default: Cyan
                        
                        if gesture == Gesture.CLOSED_FIST:
                            # Fist shrinks/hides the 3D overlay completely
                            continue
                        elif gesture == Gesture.PINCH:
                            # Pinch yields hot red glow
                            emissive = glm.vec3(3.0, 0.1, 0.1)
                            box_color = glm.vec3(1.0, 0.1, 0.1)
                        elif gesture == Gesture.PEACE:
                            # Peace yields electric blue glow
                            emissive = glm.vec3(0.1, 0.1, 3.0)
                            box_color = glm.vec3(0.1, 0.1, 1.0)
                        elif gesture == Gesture.POINTING:
                            # Pointing yields green glow
                            emissive = glm.vec3(0.1, 3.0, 0.1)
                            box_color = glm.vec3(0.1, 1.0, 0.1)
                        else:
                            # Default open palm yields soft white glow
                            emissive = glm.vec3(0.4, 0.4, 0.4)
                            
                        # Render wireframe box (Module 5 Visual calibration)
                        renderer.render_bounding_box(pos, rot, scale, proj, view, box_color)
                        
                        # Render lit 3D Cube
                        renderer.render_cube(pos, rot, scale, proj, view, emissive)
                        
                    # Perform bloom post-processing, exposure tone-mapping, and composite to screen
                    bloom_conf = render_conf.get("bloom", {})
                    bloom_intensity = bloom_conf.get("intensity", 1.2)
                    
                    tone_conf = render_conf.get("tonemapping", {})
                    exposure = tone_conf.get("exposure", 1.0)
                    gamma = tone_conf.get("gamma", 2.2)
                    
                    renderer.resolve_post_processing(
                        exposure=exposure,
                        bloom_intensity=bloom_intensity,
                        gamma=gamma
                    )
                    
                    # Swap buffers
                    window.swap_buffers()
                    
                window.process_events()
                
        except KeyboardInterrupt:
            pass
        finally:
            renderer.release()
            tracker.release()
            cam.stop()
            window.destroy()
            logger.info("System shut down cleanly.")


if __name__ == "__main__":
    main()
