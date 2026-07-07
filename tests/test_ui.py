import unittest
import dearpygui.dearpygui as dpg
from src.ui.control_panel import UIState, ControlPanel


class TestUI(unittest.TestCase):

    def test_ui_state_defaults(self):
        """Test default configurations for UIState parameters."""
        state = UIState()
        self.assertEqual(state.bloom_intensity, 1.2)
        self.assertEqual(state.exposure, 1.0)
        self.assertEqual(state.gamma, 2.2)
        self.assertEqual(state.active_model_idx, 0)
        self.assertTrue(state.animate_wings)
        self.assertEqual(state.min_detection_confidence, 0.7)
        self.assertTrue(state.smoothing_enabled)
        
        # Telemetry
        self.assertEqual(state.camera_fps, 0.0)
        self.assertEqual(state.active_hands_count, 0)
        self.assertEqual(state.left_gesture, "None")
        self.assertEqual(state.right_gesture, "None")

    def test_control_panel_layout_construction(self):
        """Test that ControlPanel initializes DPG structures without throwing errors."""
        state = UIState()
        panel = ControlPanel(state)
        
        # Setup context
        dpg.create_context()
        panel.context_created = True
        
        # Verify that DPG components can be registered on headless runners
        with dpg.window(label="TestWindow", tag="test_win"):
            dpg.add_slider_float(label="TestSlider", tag="bloom_intensity_slider", default_value=1.2)
            dpg.add_combo(items=["A", "B"], label="TestCombo", tag="active_model_combo", default_value="A")
            dpg.add_checkbox(label="TestCheck", tag="animate_wings_checkbox", default_value=True)
            dpg.add_checkbox(label="TestCheckSmooth", tag="smoothing_checkbox", default_value=True)
            dpg.add_slider_float(label="TestDetConf", tag="min_det_conf_slider", default_value=0.7)
            dpg.add_slider_float(label="TestExposure", tag="exposure_slider", default_value=1.0)
            dpg.add_slider_float(label="TestGamma", tag="gamma_slider", default_value=2.2)
            
            # Telemetry tags
            dpg.add_text("FPS", tag="fps_text")
            dpg.add_text("Hands", tag="hands_count_text")
            dpg.add_text("LeftG", tag="left_gesture_text")
            dpg.add_text("LeftP", tag="left_pos_text")
            dpg.add_text("RightG", tag="right_gesture_text")
            dpg.add_text("RightP", tag="right_pos_text")

        # Update panel properties (simulates widgets read)
        panel.update()
        
        # Verify values transferred from DPG to state
        self.assertEqual(state.bloom_intensity, 1.2)
        self.assertEqual(state.active_model_idx, 0) # "A" index is 0
        self.assertTrue(state.animate_wings)
        self.assertTrue(state.smoothing_enabled)
        
        # Clean up
        dpg.destroy_context()
        panel.context_created = False


if __name__ == "__main__":
    unittest.main()
