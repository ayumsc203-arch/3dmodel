import unittest
import moderngl
import os
import numpy as np
from pathlib import Path
from src.assets.loader import ObjLoader
from src.assets.procedural import (
    generate_butterfly_model,
    generate_phoenix_model,
    generate_dragon_model
)


class TestAssets(unittest.TestCase):
    def setUp(self):
        self.ctx = moderngl.create_context(standalone=True)
        self.test_dir = Path(__file__).parent.resolve() / "temp"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Clean up temporary test files
        for f in self.test_dir.iterdir():
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(self.test_dir)
        except OSError:
            pass
        self.ctx.release()

    def test_obj_loader_parsing(self):
        """Test parsing of a simple Wavefront OBJ file."""
        # Create a simple OBJ file (a single triangle)
        obj_content = """
        v 0.0 0.0 0.0
        v 1.0 0.0 0.0
        v 0.0 1.0 0.0
        vn 0.0 0.0 1.0
        f 1//1 2//1 3//1
        """
        file_path = self.test_dir / "triangle.obj"
        with open(file_path, "w") as f:
            f.write(obj_content)

        # Create a dummy shader program to pass to the loader
        prog = self.ctx.program(
            vertex_shader="""
                #version 330 core
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord;
                void main() {
                    vec3 p = in_position + in_normal * 0.0001 + vec3(in_texcoord, 0.0) * 0.0001;
                    gl_Position = vec4(p, 1.0);
                }
            """,
            fragment_shader="""
                #version 330 core
                out vec4 color;
                void main() {
                    color = vec4(1.0);
                }
            """
        )

        loader = ObjLoader(self.ctx)
        vbo, vao, vertex_count = loader.load_model(str(file_path), prog)
        
        # Verify
        self.assertEqual(vertex_count, 3) # 1 face * 3 vertices/face = 3 vertices
        self.assertIsNotNone(vao)
        
        # Clean up
        vbo.release()
        vao.release()
        prog.release()
        loader.clean_cache()

    def test_procedural_butterfly_generation(self):
        """Test generating and parsing the procedural butterfly model."""
        file_path = self.test_dir / "test_butterfly.obj"
        generate_butterfly_model(file_path)
        
        self.assertTrue(file_path.exists())
        self.greaterThan(os.path.getsize(file_path), 0)

        # Read the generated OBJ file lines and confirm vertices count
        v_count = 0
        f_count = 0
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    v_count += 1
                elif line.startswith("f "):
                    f_count += 1
                    
        self.greaterThan(v_count, 0)
        self.greaterThan(f_count, 0)

    def test_procedural_phoenix_generation(self):
        """Test generating and parsing the procedural phoenix model."""
        file_path = self.test_dir / "test_phoenix.obj"
        generate_phoenix_model(file_path)
        
        self.assertTrue(file_path.exists())
        self.greaterThan(os.path.getsize(file_path), 0)

        v_count = 0
        f_count = 0
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    v_count += 1
                elif line.startswith("f "):
                    f_count += 1
                    
        self.greaterThan(v_count, 0)
        self.greaterThan(f_count, 0)

    def test_procedural_dragon_generation(self):
        """Test generating and parsing the procedural dragon model."""
        file_path = self.test_dir / "test_dragon.obj"
        generate_dragon_model(file_path)
        
        self.assertTrue(file_path.exists())
        self.greaterThan(os.path.getsize(file_path), 0)

        v_count = 0
        f_count = 0
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    v_count += 1
                elif line.startswith("f "):
                    f_count += 1
                    
        self.greaterThan(v_count, 0)
        self.greaterThan(f_count, 0)

    def greaterThan(self, val1, val2, msg=None):
        self.assertTrue(val1 > val2, msg)


if __name__ == "__main__":
    unittest.main()
