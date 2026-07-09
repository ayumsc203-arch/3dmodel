import numpy as np
import glm
import random
import time
import logging
from typing import Dict, List, Tuple
from src.tracking.hand_tracker import HandData
from src.tracking.gesture_detector import Gesture
from src.particles.particle_system import ParticleSystem

logger = logging.getLogger("VfxManager")


class ButterflySwarmMember:
    """Represents a single tiny secondary butterfly orbiting the palm."""

    def __init__(self, orbit_id: int):
        self.orbit_id = orbit_id
        # Orbital properties
        self.angle = random.uniform(0.0, 2.0 * np.pi)
        self.speed = random.uniform(2.5, 4.5)
        self.radius = random.uniform(0.12, 0.28)
        self.height_offset = random.uniform(-0.1, 0.1)
        self.scale = random.uniform(0.06, 0.12)
        
        # Local position relative to palm center
        self.rel_pos = np.zeros(3, dtype=np.float32)

    def update(self, dt: float) -> None:
        """Updates orbital angle to swirl around the center."""
        self.angle += self.speed * dt
        # Spiral orbit coordinates
        self.rel_pos[0] = self.radius * np.cos(self.angle)
        self.rel_pos[1] = self.height_offset + 0.04 * np.sin(self.angle * 2.0)
        self.rel_pos[2] = self.radius * np.sin(self.angle)


class VfxManager:
    """
    Orchestrates advanced asset-particle visual effects.
    Manages transition scaling, Orchid sprouting sparks, wing trails, and secondary butterfly swarms.
    """

    def __init__(self, particle_system: ParticleSystem):
        self.particles = particle_system
        
        # Smooth scaling transitions
        self.transition_scale = 0.0
        self.target_scale = 1.0
        self.transition_speed = 3.5  # Reaches max scale in ~0.3 seconds
        
        # Tracks current active model to detect switches
        self.last_model_name = ""

        # Secondary Butterfly Swarm particles
        self.swarm: List[ButterflySwarmMember] = [
            ButterflySwarmMember(i) for i in range(5)
        ]

    def update(self, dt: float, hand: HandData, active_model_name: str, gesture: Gesture) -> None:
        """Updates transitions, spawns trailing particles, and updates swarm members."""
        # 1. Detect active model switches to trigger scale resets
        if active_model_name != self.last_model_name:
            self.transition_scale = 0.0  # Reset scale to animate sprouting
            self.last_model_name = active_model_name
            logger.info(f"VFX transition triggered for model: {active_model_name}")

        # 2. Adjust target scale based on hand gesture (Fist hides the overlay)
        if gesture == Gesture.CLOSED_FIST:
            self.target_scale = 0.0
        else:
            self.target_scale = 1.0

        # Interpolate transition scale smoothly using linear Euler step
        diff = self.target_scale - self.transition_scale
        if abs(diff) > 0.01:
            self.transition_scale += np.sign(diff) * self.transition_speed * dt
            self.transition_scale = max(0.0, min(self.transition_scale, 1.0))

        # If model is scaled down (hidden), bypass emitter ticks
        if self.transition_scale <= 0.01:
            return

        palm_pos = hand.palm_center
        rot_mat = hand.rotation_matrix
        hand_scale = hand.scale * 0.4 * self.transition_scale

        # 3. Spawn specialized particles based on active model and gestures
        if active_model_name == "orchid":
            # Orchid Blooming Sprout effect: Spawn green/pink magic sparks during scaling up
            if self.transition_scale < 0.98:
                # Sprouting sparks burst from wrist area to fingertips
                for _ in range(2):
                    offset = np.array([
                        random.uniform(-0.05, 0.05),
                        random.uniform(0.0, 0.15),
                        random.uniform(-0.05, 0.05)
                    ]) * self.transition_scale
                    self.particles.spawn(palm_pos + offset, emitter_type="magic", count=1, base_color=(0.3, 0.9, 0.3))

        elif active_model_name == "wing":
            # Fiery Wing Trail effect: Emits trailing embers along the wing span
            # Wing span lies along the hand's local X-axis (left/right)
            for _ in range(3):
                # Pick a random point along the local wing span: -1.0 to 1.0 X axis
                local_span = random.uniform(-0.8, 0.8)
                local_pos = np.array([local_span * 0.8, 0.0, -0.05], dtype=np.float32)
                
                # Transform local point to world camera coordinates: palm + rot_mat * local_pos * scale
                world_pos = palm_pos + rot_mat @ (local_pos * hand_scale * 2.5)
                
                # Spawn fire particles at the world coordinates
                self.particles.spawn(world_pos, emitter_type="fire", count=1, base_color=(1.0, 0.3, 0.1))

        elif active_model_name == "butterfly":
            # Butterfly Swarm update:
            for member in self.swarm:
                member.update(dt)
                
                # Spawn tiny sparkles trailing the secondary swarm members in world space
                member_world_pos = palm_pos + rot_mat @ (member.rel_pos * hand_scale * 2.0)
                if random.random() < 0.35:
                    self.particles.spawn(member_world_pos, emitter_type="sparkles", count=1, base_color=(0.2, 0.7, 1.0))

        elif active_model_name == "phoenix":
            # Phoenix effect: Fiery Wing Trail and Golden Tail Sparkles
            for _ in range(2):
                # Wing sparkles
                local_span = random.uniform(-1.0, 1.0)
                local_pos = np.array([local_span * 0.9, 0.0, -0.05], dtype=np.float32)
                world_pos = palm_pos + rot_mat @ (local_pos * hand_scale * 2.2)
                self.particles.spawn(world_pos, emitter_type="fire", count=1, base_color=(1.0, 0.4, 0.1))
                
            if random.random() < 0.5:
                # Tail sparkles
                tail_t = random.uniform(0.3, 0.9)
                local_pos = np.array([random.uniform(-0.1, 0.1), -tail_t * 0.7, -tail_t * 0.4], dtype=np.float32)
                world_pos = palm_pos + rot_mat @ (local_pos * hand_scale * 2.0)
                self.particles.spawn(world_pos, emitter_type="energy", count=1, base_color=(1.0, 0.8, 0.2))

        elif active_model_name == "dragon":
            # Dragon effect: Neon purple wing trails and tail sparks
            for _ in range(2):
                # Wing sparks
                local_span = random.uniform(-1.0, 1.0)
                local_pos = np.array([local_span * 1.2, 0.0, -0.2], dtype=np.float32)
                world_pos = palm_pos + rot_mat @ (local_pos * hand_scale * 2.0)
                self.particles.spawn(world_pos, emitter_type="magic", count=1, base_color=(0.6, 0.0, 0.9))
                
            if random.random() < 0.4:
                # Tail tip sparks
                local_pos = np.array([0.0, 0.0, -0.9], dtype=np.float32)
                world_pos = palm_pos + rot_mat @ (local_pos * hand_scale * 2.0)
                self.particles.spawn(world_pos, emitter_type="energy", count=1, base_color=(0.8, 0.1, 0.8))

        elif active_model_name == "energy_ball":
            # Energy Ball: Cyan/blue plasma particles drifting upward and outward from palm centroid
            for _ in range(4):
                offset = np.array([
                    random.uniform(-0.04, 0.04),
                    random.uniform(-0.04, 0.04),
                    random.uniform(-0.04, 0.04)
                ], dtype=np.float32)
                world_pos = palm_pos + rot_mat @ (offset * hand_scale)
                self.particles.spawn(world_pos, emitter_type="plasma", count=1, base_color=(0.0, 0.8, 1.0))

    def render_swarm(
        self, 
        renderer, 
        proj_matrix: glm.mat4, 
        view_matrix: glm.mat4, 
        palm_pos: glm.vec3, 
        rot_mat: glm.mat4, 
        hand_scale: float, 
        time_val: float
    ) -> None:
        """Renders secondary tiny butterflies swirling around the palm center."""
        if self.last_model_name != "butterfly" or self.transition_scale <= 0.01:
            return

        # Path to butterfly model
        model_path = "src/assets/models/butterfly.obj"
        
        # Draw each member of the swarm
        for member in self.swarm:
            # Local member offset relative to palm
            local_offset = glm.vec3(member.rel_pos[0], member.rel_pos[1], member.rel_pos[2])
            
            # Rotate local offset to match hand rotation
            world_offset = glm.vec3(rot_mat * glm.vec4(local_offset, 1.0))
            
            # Position of swarm member in world camera coordinates
            member_pos = palm_pos + world_offset * hand_scale * 2.0
            
            # Orientation: keep them slightly tilted towards flight directions
            # We add a small rotating rotation around local Y axis
            member_rot = rot_mat * glm.rotate(glm.mat4(1.0), member.angle, glm.vec3(0.0, 1.0, 0.0))
            
            # Scale of member relative to hand scale
            member_scale = hand_scale * member.scale * self.transition_scale
            
            # Soft neon blue glow
            emissive_color = glm.vec3(0.0, 0.3, 0.8)
            object_color = glm.vec3(0.1, 0.7, 1.0)

            renderer.render_model(
                model_path,
                member_pos,
                member_rot,
                member_scale,
                proj_matrix,
                view_matrix,
                emissive_color=emissive_color,
                object_color=object_color,
                animate_wings=True,
                time_val=time_val + member.orbit_id * 1.5 # phase offset in fluttering
            )
