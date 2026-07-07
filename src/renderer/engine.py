import moderngl
import numpy as np
import glm
import logging
from pathlib import Path
from typing import Optional, Tuple
from src.renderer.shader_manager import ShaderManager
from src.renderer.framebuffer import FramebufferManager

logger = logging.getLogger("RendererEngine")


class RendererEngine:
    """
    Coordinates context-level operations in ModernGL.
    Sets up background textures, basic geometry VAOs (Cube, Wireframe Box),
    configures lights, and runs the Bloom / HDR / Tone-mapping post-processing pipeline.
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height

        # Enable global depth testing and face culling
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Managers
        self.shader_manager = ShaderManager(self.ctx)
        self.fbo_manager = FramebufferManager(self.ctx, self.width, self.height)

        # Shaders compilation (must run first to bind programs)
        self._init_programs()

        # Background webcam texture
        self.bg_texture: Optional[moderngl.Texture] = None
        self._init_background_quad()

        # Geometry setup
        self._init_cube_geometry()
        self._init_box_geometry()

        # Lighting defaults
        self.dir_light_dir = glm.vec3(-0.3, -1.0, -0.5)
        self.dir_light_color = glm.vec3(1.0, 1.0, 0.95)
        self.point_light_color = glm.vec3(1.0, 0.5, 0.8)  # Glowing pink/magenta default

    def _init_programs(self) -> None:
        """Compiles shader programs using ShaderManager."""
        self.bg_prog = self.shader_manager.get_program(
            "background", "background.vert", "background.frag"
        )
        self.model_prog = self.shader_manager.get_program(
            "model", "model.vert", "model.frag"
        )
        self.bloom_prog = self.shader_manager.get_program(
            "bloom", "bloom.vert", "bloom.frag"
        )

    def _init_background_quad(self) -> None:
        """Creates VAO for full-screen background quad rendering."""
        # Quad vertices in NDC (Normalized Device Coordinates) and TexCoords
        # We flip Y in texture coordinates to account for OpenCV's top-left origin
        vertices = np.array([
            # Position(X, Y), Texture(U, V)
            -1.0,  1.0,  0.0, 0.0,
            -1.0, -1.0,  0.0, 1.0,
             1.0,  1.0,  1.0, 0.0,
            -1.0, -1.0,  0.0, 1.0,
             1.0, -1.0,  1.0, 1.0,
             1.0,  1.0,  1.0, 0.0,
        ], dtype="f4")

        self.bg_vbo = self.ctx.buffer(vertices.tobytes())
        # Bind attributes: in_position (2 floats), in_texcoord (2 floats)
        self.bg_vao = self.ctx.vertex_array(
            self.bg_prog,
            [
                (self.bg_vbo, "2f 2f", "in_position", "in_texcoord")
            ]
        )

    def _init_cube_geometry(self) -> None:
        """Creates standard 3D Cube VAO with positions, normals, and texture coordinates."""
        # 36 vertices (6 faces * 2 triangles * 3 vertices)
        # Format: Position (3f), Normal (3f), Texcoord (2f)
        cube_vertices = np.array([
            # Back face
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 0.0,
             0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 1.0,
             0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 0.0,
             0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 1.0,
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 0.0,
            -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 1.0,
            # Front face
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 0.0,
             0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 0.0,
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 1.0,
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 1.0,
            -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 1.0,
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 0.0,
            # Left face
            -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 0.0,
            -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,  1.0, 1.0,
            -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 1.0,
            -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 1.0,
            -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,  0.0, 0.0,
            -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 0.0,
            # Right face
             0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  1.0, 0.0,
             0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  0.0, 1.0,
             0.5,  0.5, -0.5,  1.0,  0.0,  0.0,  1.0, 1.0,
             0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  0.0, 1.0,
             0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  1.0, 0.0,
             0.5, -0.5,  0.5,  1.0,  0.0,  0.0,  0.0, 0.0,
            # Bottom face
            -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0, 1.0,
             0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  1.0, 1.0,
             0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0, 0.0,
             0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0, 0.0,
            -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  0.0, 0.0,
            -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0, 1.0,
            # Top face
            -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0, 1.0,
            -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  0.0, 0.0,
             0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0, 0.0,
             0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0, 0.0,
             0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  1.0, 1.0,
            -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0, 1.0,
        ], dtype="f4")

        self.cube_vbo = self.ctx.buffer(cube_vertices.tobytes())
        self.cube_vao = self.ctx.vertex_array(
            self.model_prog,
            [
                (self.cube_vbo, "3f 3f 2f", "in_position", "in_normal", "in_texcoord")
            ]
        )

    def _init_box_geometry(self) -> None:
        """Creates wireframe 3D bounding box geometry (for hand enclosing boxes)."""
        # A simple unit wireframe cuboid from [-0.5, -0.5, -0.5] to [0.5, 0.5, 0.5]
        # 12 edges * 2 vertices/edge = 24 vertices
        box_corners = np.array([
            # Lower Ring (Z = -0.5)
            -0.5, -0.5, -0.5,   0.5, -0.5, -0.5,
             0.5, -0.5, -0.5,   0.5,  0.5, -0.5,
             0.5,  0.5, -0.5,  -0.5,  0.5, -0.5,
            -0.5,  0.5, -0.5,  -0.5, -0.5, -0.5,
            # Upper Ring (Z = 0.5)
            -0.5, -0.5,  0.5,   0.5, -0.5,  0.5,
             0.5, -0.5,  0.5,   0.5,  0.5,  0.5,
             0.5,  0.5,  0.5,  -0.5,  0.5,  0.5,
            -0.5,  0.5,  0.5,  -0.5, -0.5,  0.5,
            # Pillars connecting rings
            -0.5, -0.5, -0.5,  -0.5, -0.5,  0.5,
             0.5, -0.5, -0.5,   0.5, -0.5,  0.5,
             0.5,  0.5, -0.5,   0.5,  0.5,  0.5,
            -0.5,  0.5, -0.5,  -0.5,  0.5,  0.5,
        ], dtype="f4")

        self.box_vbo = self.ctx.buffer(box_corners.tobytes())
        # We can reuse the model program, but since a wireframe has no texture or normal shading,
        # we will render it using simple color or a line pass.
        # We set input attributes; in_normal and in_texcoord are not needed but modernGL vertex arrays
        # can bind dummy/empty placeholders or we can compile a simple wireframe shader.
        # To avoid extra shaders, we compile a simple flat shader or use model shader with dummy values
        # using buffer layout.
        # Actually, wireframes are easiest drawn using model_prog with null normals/textures:
        # Since line drawing doesn't need face lighting, we can define normal as vec3(0,1,0) and tex as vec2(0)
        # or compile a tiny line program. A line program is much safer!
        # Let's write simple line shaders in code to avoid disk file dependencies:
        self.line_prog = self.ctx.program(
            vertex_shader="""
                #version 330 core
                uniform mat4 Mvp;
                in vec3 in_position;
                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330 core
                uniform vec3 Color;
                out vec4 f_color;
                void main() {
                    f_color = vec4(Color, 1.0);
                }
            """
        )
        self.box_vao = self.ctx.vertex_array(
            self.line_prog,
            [
                (self.box_vbo, "3f", "in_position")
            ]
        )

    def update_background_texture(self, frame_rgb: np.ndarray) -> None:
        """Updates the GPU texture data with a new webcam frame."""
        height, width, _ = frame_rgb.shape
        
        # Initialize texture on the first frame
        if self.bg_texture is None:
            self.bg_texture = self.ctx.texture((width, height), 3, data=frame_rgb.tobytes())
            self.bg_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.bg_texture.repeat_x = False
            self.bg_texture.repeat_y = False
        else:
            # Check if resolution changed
            if self.bg_texture.size != (width, height):
                self.bg_texture.release()
                self.bg_texture = self.ctx.texture((width, height), 3, data=frame_rgb.tobytes())
                self.bg_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            else:
                self.bg_texture.write(frame_rgb.tobytes())

    def draw_webcam_background(self) -> None:
        """Renders the webcam texture as a full-screen background quad."""
        if self.bg_texture is None:
            return

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.bg_texture.use(0)
        self.bg_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def prepare_scene(self) -> None:
        """Starts rendering on the HDR FBO, clears it, and draws the camera background quad."""
        # 1. Bind the HDR FBO
        self.fbo_manager.hdr_fbo.use()
        
        # Clear color and depth attachments
        self.fbo_manager.hdr_fbo.clear(0.0, 0.0, 0.0, 0.0)
        
        # 2. Draw webcam background quad as the first layer in HDR space
        self.draw_webcam_background()

    def render_cube(
        self, 
        position: glm.vec3, 
        rotation: glm.mat4, 
        scale: float, 
        proj_matrix: glm.mat4, 
        view_matrix: glm.mat4,
        emissive_color: glm.vec3 = glm.vec3(0.0)
    ) -> None:
        """Renders a fully-lit 3D Cube at the hand's position with spec/ambient shading and emissive bloom."""
        # Model matrix: Translation * Rotation * Scale
        model_matrix = glm.translate(glm.mat4(1.0), position)
        model_matrix = model_matrix * rotation
        model_matrix = glm.scale(model_matrix, glm.vec3(scale))

        # View-Projection matrices
        mvp = proj_matrix * view_matrix * model_matrix
        normal_matrix = glm.inverseTranspose(glm.mat3(model_matrix))

        # Bind uniforms in model program
        self.model_prog["Mvp"].write(mvp)
        self.model_prog["Model"].write(model_matrix)
        self.model_prog["NormalMatrix"].write(normal_matrix)
        
        self.model_prog["CameraPos"].write(glm.vec3(0.0, 0.0, 0.0))  # Camera is at origin in standard space
        self.model_prog["ObjectColor"].write(glm.vec3(1.0, 0.8, 0.2)) # Yellow/gold base
        self.model_prog["EmissiveColor"].write(emissive_color)
        self.model_prog["UseTexture"].value = False
        self.model_prog["BloomThreshold"].value = 0.7

        # Set Global overhead directional lighting
        self.model_prog["DirLightDir"].write(self.dir_light_dir)
        self.model_prog["DirLightColor"].write(self.dir_light_color)

        # Set Dynamic Point lighting (follows hand palm)
        self.model_prog["PointLightPos"].write(position)
        self.model_prog["PointLightColor"].write(self.point_light_color)

        # Draw Cube
        self.cube_vao.render(moderngl.TRIANGLES)

    def render_bounding_box(
        self, 
        position: glm.vec3, 
        rotation: glm.mat4, 
        scale: float, 
        proj_matrix: glm.mat4, 
        view_matrix: glm.mat4,
        color: glm.vec3 = glm.vec3(0.0, 1.0, 1.0)
    ) -> None:
        """Renders a 3D wireframe bounding box around the model (matches reference video)."""
        # Bounding box is slightly larger than the model: 2.2 times hand scale
        box_size = scale * 2.2
        model_matrix = glm.translate(glm.mat4(1.0), position)
        model_matrix = model_matrix * rotation
        model_matrix = glm.scale(model_matrix, glm.vec3(box_size))

        mvp = proj_matrix * view_matrix * model_matrix

        # Render wireframe
        self.line_prog["Mvp"].write(mvp)
        self.line_prog["Color"].write(color)
        self.box_vao.render(moderngl.LINES)

    def resolve_post_processing(
        self, 
        exposure: float = 1.0, 
        bloom_intensity: float = 1.2,
        gamma: float = 2.2
    ) -> None:
        """
        Executes the Bloom blur iterations and final tone-mapping blend pass.
        Renders the result directly to the screen.
        """
        # --- Step 1: Perform Horizontal and Vertical Gaussian Blurs ---
        # Copy the extracted bright regions (bright_tex) from HDR FBO into the first downsampled ping-pong FBO
        self.fbo_manager.ping_pong_fbos[0].use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # We need a VAO to draw full-screen texture passes. We can reuse bg_vao (which draws a full screen quad)!
        # Draw bright regions into ping_pong_fbo_0
        self.fbo_manager.bright_tex.use(0)
        self.bg_vao.render(moderngl.TRIANGLES)

        # Run Gaussian blurs
        horizontal = True
        first_iteration = True
        iterations = 4

        self.bloom_prog["BlendPass"].value = False

        for i in range(iterations * 2):
            target_fbo = self.fbo_manager.ping_pong_fbos[1 if horizontal else 0]
            source_tex = self.fbo_manager.ping_pong_texs[0 if horizontal else 1]

            target_fbo.use()
            target_fbo.clear(0.0, 0.0, 0.0, 0.0)

            # Bind uniforms
            self.bloom_prog["Horizontal"].value = horizontal
            source_tex.use(0)
            self.bg_vao.render(moderngl.TRIANGLES)

            horizontal = not horizontal
            first_iteration = False

        # --- Step 2: Composite HDR Scene with Blurred Bloom and Output to Screen ---
        # Bind the final screen canvas (FBO ID 0)
        self.ctx.screen.use()
        self.ctx.screen.clear(0.0, 0.0, 0.0, 1.0)

        # Set blend pass parameters
        self.bloom_prog["BlendPass"].value = True
        self.bloom_prog["Exposure"].value = exposure
        self.bloom_prog["BloomIntensity"].value = bloom_intensity
        self.bloom_prog["Gamma"].value = gamma

        # Bind texture units
        self.fbo_manager.color_tex.use(0)       # Texture unit 0: Scene HDR color
        self.bloom_prog["SceneTex"].value = 0
        
        # The blurred texture is stored in ping_pong_texs[0] (since we finished an even number of passes, horizontal is True)
        self.fbo_manager.ping_pong_texs[0].use(1) # Texture unit 1: Blurred bright regions
        self.bloom_prog["BloomTex"].value = 1

        # Draw final screen quad
        self.bg_vao.render(moderngl.TRIANGLES)

    def resize(self, width: int, height: int) -> None:
        """Handles window resize events."""
        self.width = width
        self.height = height
        self.fbo_manager.allocate(width, height)

    def release(self) -> None:
        """Releases all ModernGL objects."""
        if self.bg_vbo:
            self.bg_vbo.release()
        if self.bg_vao:
            self.bg_vao.release()
        if self.cube_vbo:
            self.cube_vbo.release()
        if self.cube_vao:
            self.cube_vao.release()
        if self.box_vbo:
            self.box_vbo.release()
        if self.box_vao:
            self.box_vao.release()
        if self.bg_texture:
            self.bg_texture.release()
            
        self.line_prog.release()
        self.shader_manager.cleanup()
        self.fbo_manager.release()
