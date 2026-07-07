import unittest
import moderngl
import numpy as np
from src.particles.particle_system import ParticleSystem
from src.effects.vfx_manager import VfxManager, ButterflySwarmMember
from src.tracking.hand_tracker import HandData
from src.tracking.gesture_detector import Gesture


class TestEffects(unittest.TestCase):
    def setUp(self):
        # Headless context
        self.ctx = moderngl.create_context(standalone=True)
        self.particles = ParticleSystem(self.ctx, max_particles=100)

    def tearDown(self):
        self.particles.release()
        self.ctx.release()

    def test_transition_scale(self):
        """Test that VfxManager transitions scales smoothly towards target."""
        vfx = VfxManager(self.particles)
        
        # Initial scale
        self.assertEqual(vfx.transition_scale, 0.0)
        
        # Mock hand data
        hand = HandData(
            label="Right",
            landmarks_2d=np.zeros((21, 2)),
            landmarks_3d=np.zeros((21, 3)),
            palm_center=np.array([0.0, 0.0, -0.5]),
            rotation_matrix=np.eye(3),
            scale=1.0
        )
        
        # Tick update with dt = 0.1s
        vfx.update(dt=0.1, hand=hand, active_model_name="orchid", gesture=Gesture.OPEN_PALM)
        
        # Scale should have risen
        self.assertGreater(vfx.transition_scale, 0.0)
        self.assertLessEqual(vfx.transition_scale, 1.0)
        
        # Simulate several ticks to reach target 1.0
        for _ in range(10):
            vfx.update(dt=0.1, hand=hand, active_model_name="orchid", gesture=Gesture.OPEN_PALM)
            
        self.assertEqual(vfx.transition_scale, 1.0)
        
        # Switch gesture to Closed Fist (should scale down to 0)
        vfx.update(dt=0.1, hand=hand, active_model_name="orchid", gesture=Gesture.CLOSED_FIST)
        self.assertLess(vfx.transition_scale, 1.0)

    def test_butterfly_swarm_movement(self):
        """Test that secondary swarm member angles and relative coordinates update."""
        member = ButterflySwarmMember(orbit_id=0)
        initial_angle = member.angle
        initial_pos = member.rel_pos.copy()
        
        # Update
        member.update(dt=0.1)
        
        # Angle should change
        self.assertNotEqual(member.angle, initial_angle)
        # Position should change
        self.assertFalse(np.array_equal(member.rel_pos, initial_pos))


if __name__ == "__main__":
    unittest.main()
