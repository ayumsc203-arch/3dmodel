import dearpygui.dearpygui as dpg
import logging
from typing import Tuple

logger = logging.getLogger("ControlPanel")


class UIState:
    """Stores shared telemetry and settings variables between ModernGL loop and Dear PyGui."""

    def __init__(self):
        # Settings
        self.bloom_intensity = 1.2
        self.exposure = 1.0
        self.gamma = 2.2
        
        self.active_model_idx = 0
        self.animate_wings = True
        
        self.min_detection_confidence = 0.7
        self.smoothing_enabled = True
        
        # Telemetry / Stats
        self.camera_fps = 0.0
        self.active_hands_count = 0
        self.left_gesture = "None"
        self.right_gesture = "None"
        self.left_palm_pos = (0.0, 0.0, 0.0)
        self.right_palm_pos = (0.0, 0.0, 0.0)


class ControlPanel:
    """Implements a sleek Dear PyGui control window for real-time visual adjustments."""

    def __init__(self, state: UIState):
        self.state = state
        self.context_created = False

    def setup(self, width: int = 380, height: int = 680) -> None:
        """Initializes DPG context, layouts, styles, and viewport settings."""
        logger.info("Setting up Dear PyGui control panel...")
        
        dpg.create_context()
        self.context_created = True
        
        dpg.create_viewport(
            title="VFX System Control Panel",
            width=width,
            height=height,
            resizable=False,
            decorated=True
        )
        dpg.setup_dearpygui()

        # Define Layout
        with dpg.window(label="System Controller", tag="main_window", width=width, height=height, no_resize=True, no_move=True, no_close=True):
            # Header Title
            dpg.add_text("HAND TRACKING VFX", color=[0, 255, 255])
            dpg.add_separator()
            dpg.add_spacer(height=5)

            # 1. Telemetry Section
            with dpg.collapsing_header(label="SYSTEM TELEMETRY", default_open=True):
                dpg.add_text("Camera FPS: 0.0", tag="fps_text")
                dpg.add_text("Active Hands: 0", tag="hands_count_text")
                dpg.add_spacer(height=3)
                
                dpg.add_text("Left Gesture: None", tag="left_gesture_text", color=[0, 255, 0])
                dpg.add_text("Left Palm: (0.00, 0.00, 0.00)", tag="left_pos_text")
                dpg.add_spacer(height=3)
                
                dpg.add_text("Right Gesture: None", tag="right_gesture_text", color=[255, 0, 255])
                dpg.add_text("Right Palm: (0.00, 0.00, 0.00)", tag="right_pos_text")

            dpg.add_spacer(height=5)

            # 2. Post Processing Section
            with dpg.collapsing_header(label="POST PROCESSING (GLSL)", default_open=True):
                dpg.add_slider_float(
                    label="Bloom Intensity",
                    tag="bloom_intensity_slider",
                    default_value=self.state.bloom_intensity,
                    min_value=0.0,
                    max_value=4.0,
                    format="%.2f"
                )
                dpg.add_slider_float(
                    label="Exposure (HDR)",
                    tag="exposure_slider",
                    default_value=self.state.exposure,
                    min_value=0.1,
                    max_value=2.5,
                    format="%.2f"
                )
                dpg.add_slider_float(
                    label="Gamma Correction",
                    tag="gamma_slider",
                    default_value=self.state.gamma,
                    min_value=1.0,
                    max_value=3.0,
                    format="%.2f"
                )

            dpg.add_spacer(height=5)

            # 3. 3D Asset Settings Section
            with dpg.collapsing_header(label="3D ASSETS SELECTION", default_open=True):
                models_items = ["Orchid Plant", "Fantasy Wing", "Butterfly Swarm"]
                dpg.add_combo(
                    items=models_items,
                    label="Active Asset",
                    tag="active_model_combo",
                    default_value=models_items[self.state.active_model_idx]
                )
                dpg.add_checkbox(
                    label="Animate Wings (Butterfly)",
                    tag="animate_wings_checkbox",
                    default_value=self.state.animate_wings
                )

            dpg.add_spacer(height=5)

            # 4. Tracking Settings Section
            with dpg.collapsing_header(label="TRACKING CONFIG", default_open=True):
                dpg.add_slider_float(
                    label="Min Detection Confidence",
                    tag="min_det_conf_slider",
                    default_value=self.state.min_detection_confidence,
                    min_value=0.4,
                    max_value=0.9,
                    format="%.2f"
                )
                dpg.add_checkbox(
                    label="Enable One Euro Smoothing",
                    tag="smoothing_checkbox",
                    default_value=self.state.smoothing_enabled
                )

        # Style layout dark theme
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                # Cyan accents dark theme
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, [18, 18, 22])
                dpg.add_theme_color(dpg.mvThemeCol_Header, [28, 48, 58])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, [40, 68, 80])
                dpg.add_theme_color(dpg.mvThemeCol_Button, [32, 60, 72])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [48, 85, 100])
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, [0, 200, 200])
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, [0, 255, 255])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, [24, 28, 32])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, [32, 38, 44])
                
        dpg.bind_theme(global_theme)
        dpg.show_viewport()

    def update(self) -> None:
        """Pulls user inputs from UI widgets to state, updates telemetry labels, and renders frame."""
        if not self.context_created or not dpg.is_dearpygui_running():
            return

        # Synchronize active model combo box with external updates (like swipe gestures)
        model_list = ["Orchid Plant", "Fantasy Wing", "Butterfly Swarm"]
        current_combo_val = dpg.get_value("active_model_combo")
        expected_combo_val = model_list[self.state.active_model_idx]
        if current_combo_val != expected_combo_val:
            dpg.set_value("active_model_combo", expected_combo_val)

        # 1. Update State variables from DPG inputs
        self.state.bloom_intensity = dpg.get_value("bloom_intensity_slider")
        self.state.exposure = dpg.get_value("exposure_slider")
        self.state.gamma = dpg.get_value("gamma_slider")
        
        model_str = dpg.get_value("active_model_combo")
        model_map = {"Orchid Plant": 0, "Fantasy Wing": 1, "Butterfly Swarm": 2}
        self.state.active_model_idx = model_map.get(model_str, 0)
        
        self.state.animate_wings = dpg.get_value("animate_wings_checkbox")
        self.state.min_detection_confidence = dpg.get_value("min_det_conf_slider")
        self.state.smoothing_enabled = dpg.get_value("smoothing_checkbox")

        # 2. Push telemetry status to labels
        dpg.set_value("fps_text", f"Camera FPS: {self.state.camera_fps:.1f}")
        dpg.set_value("hands_count_text", f"Active Hands: {self.state.active_hands_count}")
        
        dpg.set_value("left_gesture_text", f"Left Gesture: {self.state.left_gesture}")
        dpg.set_value("left_pos_text", f"Left Palm: ({self.state.left_palm_pos[0]:.2f}, {self.state.left_palm_pos[1]:.2f}, {self.state.left_palm_pos[2]:.2f})")
        
        dpg.set_value("right_gesture_text", f"Right Gesture: {self.state.right_gesture}")
        dpg.set_value("right_pos_text", f"Right Palm: ({self.state.right_palm_pos[0]:.2f}, {self.state.right_palm_pos[1]:.2f}, {self.state.right_palm_pos[2]:.2f})")

        # 3. Synchronously render UI frame
        dpg.render_dearpygui_frame()

    def set_active_model_combo(self, idx: int) -> None:
        """Deprecated: Synchronization is now handled internally inside update() to ensure thread safety."""
        pass

    def release(self) -> None:
        """Destroys DPG window context."""
        if self.context_created:
            dpg.destroy_context()
            self.context_created = False
            logger.info("Dear PyGui resources released.")
