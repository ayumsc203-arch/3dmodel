import unittest
import os
import yaml
from pathlib import Path

class TestSetup(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(__file__).parent.parent.resolve()

    def test_directories_exist(self):
        """Verify that all required project directories were created."""
        from run import PROJECT_DIRS
        for relative_path in PROJECT_DIRS:
            dir_path = self.base_dir / relative_path
            self.assertTrue(dir_path.exists(), f"Directory should exist: {relative_path}")

    def test_settings_yaml(self):
        """Verify settings.yaml contains all required config blocks."""
        config_path = self.base_dir / "config" / "settings.yaml"
        self.assertTrue(config_path.exists(), "settings.yaml should exist")
        
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            
        self.assertIn("camera", data)
        self.assertIn("tracking", data)
        self.assertIn("rendering", data)
        self.assertIn("particles", data)
        self.assertIn("vfx", data)

        # Check basic camera configurations
        self.assertIn("device_index", data["camera"])
        self.assertIn("width", data["camera"])
        self.assertIn("height", data["camera"])
        
        # Check basic tracking configurations
        self.assertIn("max_num_hands", data["tracking"])

    def test_requirements_file(self):
        """Verify requirements.txt exists and is not empty."""
        req_path = self.base_dir / "requirements.txt"
        self.assertTrue(req_path.exists(), "requirements.txt should exist")
        self.greaterThan(os.path.getsize(req_path), 0, "requirements.txt should not be empty")

    def greaterThan(self, val1, val2, msg=None):
        self.assertTrue(val1 > val2, msg)

if __name__ == "__main__":
    unittest.main()
