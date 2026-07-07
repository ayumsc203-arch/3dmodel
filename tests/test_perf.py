import unittest
import moderngl
import numpy as np
import time
from pathlib import Path
from src.assets.loader import ObjLoader
from src.assets.procedural import generate_butterfly_model
from src.particles.particle_system import ParticleSystem


class TestPerformance(unittest.TestCase):

    def setUp(self):
        # Setup standalone headless context
        self.ctx = moderngl.create_context(standalone=True)
        self.test_dir = Path(__file__).parent.resolve() / "temp_perf"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Clean up temporary test files
        for f in self.test_dir.iterdir():
            try:
                f.unlink()
            except OSError:
                pass
        try:
            self.test_dir.rmdir()
        except OSError:
            pass
        self.ctx.release()

    def test_asset_loader_cache_performance(self):
        """Verify that asset loading cache hits resolve in sub-millisecond times."""
        # Create a simple OBJ model
        file_path = self.test_dir / "temp_perf.obj"
        generate_butterfly_model(file_path)

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

        # First load: reads from disk, parses, creates GPU buffers
        start_time = time.perf_counter()
        vbo1, vao1, count1 = loader.load_model(str(file_path), prog)
        load1_duration = time.perf_counter() - start_time

        # Second load: retrieves from cache
        start_time = time.perf_counter()
        vbo2, vao2, count2 = loader.load_model(str(file_path), prog)
        load2_duration = time.perf_counter() - start_time

        # Assert second load is extremely fast (cache hit)
        self.assertLess(load2_duration, 0.001, "Cache hit must be sub-millisecond.")
        self.assertLess(load2_duration, load1_duration, "Cache hit must be faster than disk parsing.")
        self.assertEqual(count1, count2)

        # Release
        vao1.release()
        vbo1.release()
        prog.release()
        loader.clean_cache()

    def test_particle_physics_update_performance(self):
        """Verify that updating 3000 active particles takes less than 5ms (vectorized NumPy speed)."""
        ps = ParticleSystem(self.ctx, max_particles=3000)

        # Spawn 3000 active particles
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ps.spawn(origin, emitter_type="fire", count=1500)
        ps.spawn(origin, emitter_type="sparkles", count=1500)

        # Benchmark 100 physics steps
        start_time = time.perf_counter()
        for _ in range(100):
            ps.update(dt=0.016)
        avg_step_duration = (time.perf_counter() - start_time) / 100.0

        # Assert average step duration is under 5ms
        self.assertLess(avg_step_duration, 0.005, "NumPy vectorized update step must be under 5ms.")
        ps.release()


if __name__ == "__main__":
    unittest.main()
