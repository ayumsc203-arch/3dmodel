import unittest
import moderngl
import numpy as np
import glm
from src.renderer.shader_manager import ShaderManager
from src.renderer.framebuffer import FramebufferManager
from src.renderer.engine import RendererEngine


class TestRenderer(unittest.TestCase):
    def setUp(self):
        # Create a standalone OpenGL context (headless)
        self.ctx = moderngl.create_context(standalone=True)
        
    def tearDown(self):
        self.ctx.release()

    def test_framebuffer_allocation(self):
        """Test FBO and texture sizes during allocation and resizing."""
        fbo_manager = FramebufferManager(self.ctx, width=800, height=600, bloom_scale=0.25)
        
        # Verify FBO attachments and texture sizes
        self.assertEqual(fbo_manager.width, 800)
        self.assertEqual(fbo_manager.height, 600)
        self.assertEqual(fbo_manager.color_tex.size, (800, 600))
        self.assertEqual(fbo_manager.bright_tex.size, (800, 600))
        self.assertEqual(fbo_manager.ping_pong_texs[0].size, (200, 150))
        
        # Test resize
        fbo_manager.allocate(1024, 768)
        self.assertEqual(fbo_manager.width, 1024)
        self.assertEqual(fbo_manager.color_tex.size, (1024, 768))
        self.assertEqual(fbo_manager.ping_pong_texs[0].size, (256, 192))
        
        fbo_manager.release()

    def test_shader_compilation(self):
        """Test loading and compiling of shaders via ShaderManager."""
        shader_manager = ShaderManager(self.ctx)
        
        # Compile background shader
        prog = shader_manager.get_program("background", "background.vert", "background.frag")
        self.assertIsNotNone(prog)
        self.assertIn("Texture", prog)
        
        # Verify caching works
        prog2 = shader_manager.get_program("background", "background.vert", "background.frag")
        self.assertEqual(prog, prog2)
        
        shader_manager.cleanup()

    def test_renderer_engine_initialization(self):
        """Test that RendererEngine initializes properties and geometry successfully."""
        engine = RendererEngine(self.ctx, width=800, height=600)
        
        # Check defaults
        self.assertIsNotNone(engine.cube_vao)
        self.assertIsNotNone(engine.box_vao)
        self.assertIsNotNone(engine.line_prog)
        
        # Check lights vectors
        self.assertIsInstance(engine.dir_light_dir, glm.vec3)
        self.assertIsInstance(engine.point_light_color, glm.vec3)
        
        engine.release()


if __name__ == "__main__":
    unittest.main()
