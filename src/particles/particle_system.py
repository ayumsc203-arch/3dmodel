import numpy as np
import moderngl
import random
import logging
from typing import Tuple

logger = logging.getLogger("ParticleSystem")


class ParticleSystem:
    """
    High-performance NumPy-backed GPU instanced particle simulator.
    Supports Fire, Smoke, Sparkles, Magic, and Energy emitters.
    """

    def __init__(self, ctx: moderngl.Context, max_particles: int = 3000):
        self.ctx = ctx
        self.max_particles = max_particles

        # Particle state arrays (pre-allocated pool for O(1) recycle speed)
        self.positions = np.zeros((self.max_particles, 3), dtype=np.float32)
        self.velocities = np.zeros((self.max_particles, 3), dtype=np.float32)
        self.colors = np.zeros((self.max_particles, 4), dtype=np.float32)
        self.sizes = np.zeros(self.max_particles, dtype=np.float32)
        self.lifetimes = np.zeros(self.max_particles, dtype=np.float32)
        self.max_lifetimes = np.ones(self.max_particles, dtype=np.float32)
        self.active_mask = np.zeros(self.max_particles, dtype=bool)

        # Particle type labels (to apply custom updates if needed)
        self.types = ["sparkle"] * self.max_particles 

        # Free indices list
        self.free_indices = list(range(self.max_particles))

        # ModernGL Shaders & Buffers
        self._init_gpu_buffers()

    def _init_gpu_buffers(self) -> None:
        """Initializes quad buffers and dynamic instanced buffers on the GPU."""
        # 1. Compile Shader program
        from src.renderer.shader_manager import ShaderManager
        # We assume program compilation is managed, but to keep ParticleSystem self-contained,
        # we check the shader manager or compile in-place.
        # Actually, let's load it from the engine's shader cache or compile a standard one:
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330 core
                uniform mat4 Projection;
                uniform mat4 View;
                in vec2 in_quad;
                in vec3 in_pos;
                in vec4 in_color;
                in float in_size;
                out vec2 v_texcoord;
                out vec4 v_color;
                void main() {
                    vec4 viewPos = View * vec4(in_pos, 1.0);
                    viewPos.xy += in_quad * in_size;
                    v_texcoord = in_quad + vec2(0.5);
                    v_color = in_color;
                    gl_Position = Projection * viewPos;
                }
            """,
            fragment_shader="""
                #version 330 core
                layout (location = 0) out vec4 FragColor;
                layout (location = 1) out vec4 BrightColor;
                in vec2 v_texcoord;
                in vec4 v_color;
                uniform float BloomThreshold;
                void main() {
                    vec2 temp_uv = v_texcoord - vec2(0.5);
                    float dist = length(temp_uv);
                    if (dist > 0.5) discard;
                    float intensity = smoothstep(0.5, 0.0, dist);
                    float alpha = intensity * v_color.a;
                    vec4 color = vec4(v_color.rgb, alpha);
                    FragColor = color;
                    float luminance = dot(color.rgb * alpha, vec3(0.2126, 0.7152, 0.0722));
                    if (luminance > BloomThreshold) {
                        BrightColor = vec4(color.rgb * alpha * 2.0, 1.0);
                    } else {
                        BrightColor = vec4(0.0, 0.0, 0.0, 1.0);
                    }
                }
            """
        )

        # 2. Local Quad Vertex Buffer: standard Triangle Strip quad
        quad_vertices = np.array([
            -0.5,  0.5,
            -0.5, -0.5,
             0.5,  0.5,
             0.5, -0.5
        ], dtype="f4")
        self.quad_vbo = self.ctx.buffer(quad_vertices.tobytes())

        # 3. Preallocate empty Dynamic Instance Buffer (interleaved positions(3f), colors(4f), sizes(1f) = 8f per particle)
        self.instance_data = np.zeros((self.max_particles, 8), dtype=np.float32)
        self.instance_vbo = self.ctx.buffer(reserve=self.max_particles * 8 * 4, dynamic=True)

        # 4. Create instanced VAO
        # Quad is bound per vertex (attrib 0: in_quad)
        # Positions, colors, sizes are bound per instance (attrib divisors = 1)
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.quad_vbo, "2f", "in_quad"),
                (self.instance_vbo, "3f 4f 1f/i", "in_pos", "in_color", "in_size")
            ]
        )

    def spawn(
        self, 
        origin: np.ndarray, 
        emitter_type: str = "sparkles", 
        count: int = 20, 
        base_color: tuple = (1.0, 1.0, 1.0)
    ) -> None:
        """Spawns a burst of particles from an origin coordinate."""
        if len(self.free_indices) < count:
            # Recycle oldest particles if we exceed limit
            dead_candidates = np.where(~self.active_mask)[0]
            if len(dead_candidates) + len(self.free_indices) < count:
                # Force kill oldest active particles
                active_indices = np.where(self.active_mask)[0]
                if len(active_indices) > 0:
                    kill_count = count - len(self.free_indices)
                    for k in active_indices[:kill_count]:
                        self.active_mask[k] = False
                        self.free_indices.append(k)

        # Retrieve available slots
        spawn_slots = [self.free_indices.pop() for _ in range(min(count, len(self.free_indices)))]

        for idx in spawn_slots:
            self.active_mask[idx] = True
            self.positions[idx] = origin
            self.max_lifetimes[idx] = random.uniform(0.4, 1.2)
            self.lifetimes[idx] = self.max_lifetimes[idx]
            self.types[idx] = emitter_type

            # Setup physics parameters based on emitter type
            if emitter_type == "fire":
                # Fire: shoots upwards, orange/yellow
                self.velocities[idx] = [
                    random.uniform(-0.15, 0.15),
                    random.uniform(0.3, 0.7),
                    random.uniform(-0.15, 0.15)
                ]
                self.colors[idx] = [base_color[0], base_color[1], base_color[2], 0.9]
                self.sizes[idx] = random.uniform(0.04, 0.12)
                self.max_lifetimes[idx] = random.uniform(0.3, 0.6)
                self.lifetimes[idx] = self.max_lifetimes[idx]

            elif emitter_type == "smoke":
                # Smoke: drifts slowly, grey/dark
                self.velocities[idx] = [
                    random.uniform(-0.08, 0.08),
                    random.uniform(0.1, 0.3),
                    random.uniform(-0.08, 0.08)
                ]
                self.colors[idx] = [0.4, 0.4, 0.4, 0.5]
                self.sizes[idx] = random.uniform(0.08, 0.22)
                self.max_lifetimes[idx] = random.uniform(0.8, 1.5)
                self.lifetimes[idx] = self.max_lifetimes[idx]

            elif emitter_type == "sparkles":
                # Sparkles: spherical burst, gravity drop
                angle = random.uniform(0.0, 2.0 * np.pi)
                phi = np.arccos(random.uniform(-1.0, 1.0))
                speed = random.uniform(0.4, 1.0)
                
                self.velocities[idx] = [
                    speed * np.sin(phi) * np.cos(angle),
                    speed * np.cos(phi),
                    speed * np.sin(phi) * np.sin(angle)
                ]
                self.colors[idx] = [base_color[0], base_color[1], base_color[2], 1.0]
                self.sizes[idx] = random.uniform(0.015, 0.04)

            elif emitter_type == "magic":
                # Magic: spiral orbital swirls
                self.velocities[idx] = [
                    random.uniform(-0.25, 0.25),
                    random.uniform(-0.25, 0.25),
                    random.uniform(-0.25, 0.25)
                ]
                self.colors[idx] = [base_color[0], base_color[1], base_color[2], 0.95]
                self.sizes[idx] = random.uniform(0.02, 0.07)
                
            elif emitter_type == "energy":
                # Energy: fast radial expansion
                angle = random.uniform(0.0, 2.0 * np.pi)
                speed = random.uniform(0.5, 1.2)
                self.velocities[idx] = [
                    speed * np.cos(angle),
                    random.uniform(-0.1, 0.1),
                    speed * np.sin(angle)
                ]
                self.colors[idx] = [base_color[0], base_color[1], base_color[2], 0.8]
                self.sizes[idx] = random.uniform(0.03, 0.09)

    def update(self, dt: float, gravity: np.ndarray = np.array([0.0, -1.0, 0.0])) -> int:
        """
        Updates the physics equations of all active particles using NumPy.
        Returns the number of active particles.
        """
        active_indices = np.where(self.active_mask)[0]
        if len(active_indices) == 0:
            return 0

        # Update lifetimes
        self.lifetimes[active_indices] -= dt

        # Recycle dead particles
        dead_mask = self.lifetimes[active_indices] <= 0.0
        if np.any(dead_mask):
            dead_indices = active_indices[dead_mask]
            self.active_mask[dead_indices] = False
            self.free_indices.extend(list(dead_indices))
            # Refresh active indices
            active_indices = np.where(self.active_mask)[0]
            if len(active_indices) == 0:
                return 0

        # Euler Physics Integration (vectorized)
        self.positions[active_indices] += self.velocities[active_indices] * dt
        
        # Apply custom emitter modifiers
        for idx in active_indices:
            p_type = self.types[idx]
            
            # Apply gravity to sparkles
            if p_type == "sparkles":
                self.velocities[idx] += gravity * dt
                
            # Fire particles scale down over lifetime and fade color from yellow to red
            elif p_type == "fire":
                ratio = self.lifetimes[idx] / self.max_lifetimes[idx]
                self.sizes[idx] *= 0.97
                # Fade color: Yellow -> Red -> Black
                self.colors[idx][0] = 1.0
                self.colors[idx][1] = max(0.0, ratio - 0.2)
                self.colors[idx][2] = 0.0
                
            # Smoke scales up (expands)
            elif p_type == "smoke":
                self.sizes[idx] += dt * 0.1
                
            # Magic spirals around origin
            elif p_type == "magic":
                # Add swirling orbital force
                pos = self.positions[idx]
                r_vec = np.array([-pos[2], 0.0, pos[0]])  # Perpendicular tangential vector
                r_norm = np.linalg.norm(r_vec)
                if r_norm > 0.0:
                    self.velocities[idx] += (r_vec / r_norm) * 0.4 * dt

        # Alpha fadeout over lifetime
        ratios = self.lifetimes[active_indices] / self.max_lifetimes[active_indices]
        self.colors[active_indices, 3] = ratios  # Alpha matches life ratio

        return len(active_indices)

    def render(self, projection_matrix: np.ndarray, view_matrix: np.ndarray) -> None:
        """Assembles active particle buffer, writes to VBO, and renders instanced billboard geometry."""
        active_indices = np.where(self.active_mask)[0]
        active_count = len(active_indices)
        
        if active_count == 0:
            return

        # Interleave active particle states: pos(3), color(4), size(1)
        self.instance_data[:active_count, 0:3] = self.positions[active_indices]
        self.instance_data[:active_count, 3:7] = self.colors[active_indices]
        self.instance_data[:active_count, 7] = self.sizes[active_indices]

        # Upload instance buffer to GPU
        self.instance_vbo.write(self.instance_data[:active_count].tobytes())

        # Setup Blend state
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.ctx.disable(moderngl.DEPTH_TEST)  # Additive blending transparency overlay

        # Bind shader and matrices
        self.prog["Projection"].write(projection_matrix)
        self.prog["View"].write(view_matrix)
        self.prog["BloomThreshold"].value = 0.7

        # Render active instances
        self.vao.render(moderngl.TRIANGLE_STRIP, instances=active_count)

        # Reset state
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def release(self) -> None:
        """Releases GPU buffers."""
        self.quad_vbo.release()
        self.instance_vbo.release()
        self.vao.release()
        self.prog.release()
        logger.info("ParticleSystem resources released.")
