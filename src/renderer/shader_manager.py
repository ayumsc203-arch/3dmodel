import os
import logging
import moderngl
from pathlib import Path
from typing import Dict

logger = logging.getLogger("ShaderManager")


class ShaderManager:
    """Loads, compiles, and caches GLSL shader programs in ModernGL."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.programs: Dict[str, moderngl.Program] = {}
        self.shader_dir = Path(__file__).parent.resolve() / "shaders"

    def _read_shader_file(self, name: str) -> str:
        """Reads shader source code from disk."""
        path = self.shader_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Shader source file not found: {path}")
        with open(path, "r") as f:
            return f.read()

    def get_program(
        self, 
        name: str, 
        vertex_file: str, 
        fragment_file: str
    ) -> moderngl.Program:
        """
        Compiles and caches a Program if not already compiled.
        Retrieves from cache if available.
        """
        if name in self.programs:
            return self.programs[name]

        try:
            logger.info(f"Compiling shader program '{name}'...")
            vert_src = self._read_shader_file(vertex_file)
            frag_src = self._read_shader_file(fragment_file)
            
            program = self.ctx.program(
                vertex_shader=vert_src,
                fragment_shader=frag_src
            )
            self.programs[name] = program
            logger.info(f"Successfully compiled and cached '{name}'.")
            return program
            
        except moderngl.Error as e:
            logger.error(f"Failed to compile shader '{name}': {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error loading shader '{name}': {e}")
            raise e

    def cleanup(self) -> None:
        """Releases all cached program resources."""
        for name, prog in list(self.programs.items()):
            prog.release()
        self.programs.clear()
        logger.info("ShaderManager resources released.")
