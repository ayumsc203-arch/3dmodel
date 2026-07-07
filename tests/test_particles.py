import unittest
import moderngl
import numpy as np
from src.particles.particle_system import ParticleSystem


class TestParticleSystem(unittest.TestCase):
    def setUp(self):
        # Headless standalone context
        self.ctx = moderngl.create_context(standalone=True)

    def tearDown(self):
        self.ctx.release()

    def test_particle_spawning(self):
        """Test that spawning generates active particles."""
        ps = ParticleSystem(self.ctx, max_particles=100)
        
        # Spawn 10 sparkles
        origin = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ps.spawn(origin, emitter_type="sparkles", count=10, base_color=(1.0, 0.5, 0.0))
        
        # Verify 10 particles are active
        active_count = np.sum(ps.active_mask)
        self.assertEqual(active_count, 10)
        self.assertEqual(len(ps.free_indices), 90)
        
        # Verify positions
        for idx in np.where(ps.active_mask)[0]:
            np.testing.assert_array_almost_equal(ps.positions[idx], origin)
            
        ps.release()

    def test_particle_physics_update(self):
        """Test particle movement and lifetime countdown."""
        ps = ParticleSystem(self.ctx, max_particles=10)
        
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ps.spawn(origin, emitter_type="fire", count=5)
        
        # Save active velocities
        active_indices = np.where(ps.active_mask)[0]
        vels = ps.velocities[active_indices].copy()
        
        # Update system with dt = 0.1s
        ps.update(dt=0.1)
        
        # Verify position = velocity * dt (for simple linear fire)
        for i, idx in enumerate(active_indices):
            expected_pos = vels[i] * 0.1
            np.testing.assert_array_almost_equal(ps.positions[idx], expected_pos)
            
        ps.release()

    def test_particle_recycling(self):
        """Test that dead particles are returned to the free pool."""
        ps = ParticleSystem(self.ctx, max_particles=10)
        
        ps.spawn(np.array([0.0, 0.0, 0.0]), emitter_type="fire", count=5)
        
        # Manually force lifetimes of 3 particles to 0
        active_indices = np.where(ps.active_mask)[0]
        ps.lifetimes[active_indices[0]] = -0.1
        ps.lifetimes[active_indices[1]] = -0.1
        ps.lifetimes[active_indices[2]] = -0.1
        
        # Run update
        ps.update(dt=0.01)
        
        # 3 should have been recycled, leaving 2 active
        active_count = np.sum(ps.active_mask)
        self.assertEqual(active_count, 2)
        self.assertEqual(len(ps.free_indices), 8)
        
        ps.release()


if __name__ == "__main__":
    unittest.main()
