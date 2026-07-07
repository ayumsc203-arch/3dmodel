import moderngl
import logging
from typing import Tuple

logger = logging.getLogger("FramebufferManager")


class FramebufferManager:
    """
    Manages HDR framebuffers and downsampled ping-pong targets for Bloom post-processing.
    Provides multiple render targets (MRT) to separate bright pixels from normal geometry.
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int, bloom_scale: float = 0.25):
        self.ctx = ctx
        self.bloom_scale = bloom_scale

        # HDR main targets
        self.hdr_fbo: moderngl.Framebuffer = None
        self.color_tex: moderngl.Texture = None
        self.bright_tex: moderngl.Texture = None
        self.depth_rb: moderngl.Renderbuffer = None

        # Ping-Pong FBOs for vertical/horizontal blurs (downsampled for speed)
        self.ping_pong_fbos: Tuple[moderngl.Framebuffer, moderngl.Framebuffer] = (None, None)
        self.ping_pong_texs: Tuple[moderngl.Texture, moderngl.Texture] = (None, None)

        self.allocate(width, height)

    def allocate(self, width: int, height: int) -> None:
        """Allocates or re-allocates GPU buffer resources based on viewport size."""
        self.release()
        
        self.width = width
        self.height = height
        self.blur_width = int(width * self.bloom_scale)
        self.blur_height = int(height * self.bloom_scale)

        try:
            logger.info(f"Allocating Framebuffers: Main HDR ({width}x{height}), Blur ({self.blur_width}x{self.blur_height})")

            # 1. Allocate Main HDR textures (16-bit float per component for high dynamic range)
            self.color_tex = self.ctx.texture((width, height), 4, dtype="f2")
            self.bright_tex = self.ctx.texture((width, height), 4, dtype="f2")
            
            # Setup filtering/wrapping modes
            for tex in [self.color_tex, self.bright_tex]:
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                tex.repeat_x = False
                tex.repeat_y = False
                
            # Depth buffer
            self.depth_rb = self.ctx.depth_renderbuffer((width, height))

            # Create main HDR Framebuffer with Multiple Render Targets (MRT)
            self.hdr_fbo = self.ctx.framebuffer(
                color_attachments=[self.color_tex, self.bright_tex],
                depth_attachment=self.depth_rb
            )

            # 2. Allocate Ping-Pong FBOs (16-bit float, downsampled)
            fbos = []
            texs = []
            for i in range(2):
                tex = self.ctx.texture((self.blur_width, self.blur_height), 4, dtype="f2")
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                tex.repeat_x = False
                tex.repeat_y = False
                fbo = self.ctx.framebuffer(color_attachments=[tex])
                texs.append(tex)
                fbos.append(fbo)
                
            self.ping_pong_fbos = (fbos[0], fbos[1])
            self.ping_pong_texs = (texs[0], texs[1])
            
            logger.info("Successfully allocated framebuffers.")

        except Exception as e:
            logger.error(f"Error allocating framebuffer resources: {e}")
            raise e

    def release(self) -> None:
        """Releases GPU buffer resources to prevent leaks."""
        # Release main HDR targets
        if self.hdr_fbo:
            self.hdr_fbo.release()
            self.hdr_fbo = None
        if self.color_tex:
            self.color_tex.release()
            self.color_tex = None
        if self.bright_tex:
            self.bright_tex.release()
            self.bright_tex = None
        if self.depth_rb:
            self.depth_rb.release()
            self.depth_rb = None

        # Release Ping-Pong FBOs
        if self.ping_pong_fbos[0]:
            self.ping_pong_fbos[0].release()
            self.ping_pong_fbos[1].release()
            self.ping_pong_fbos = (None, None)
            
        if self.ping_pong_texs[0]:
            self.ping_pong_texs[0].release()
            self.ping_pong_texs[1].release()
            self.ping_pong_texs = (None, None)
