import os
import numpy as np
import moderngl
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger("AssetLoader")


class MeshData:
    """Stores intermediate CPU vertex data parsed from an OBJ file."""

    def __init__(self):
        self.positions: List[Tuple[float, float, float]] = []
        self.normals: List[Tuple[float, float, float]] = []
        self.texcoords: List[Tuple[float, float]] = []
        self.faces: List[List[Tuple[int, int, int]]] = []  # List of triangles: each has 3 vertices of (p_idx, t_idx, n_idx)


class ObjLoader:
    """
    Custom Wavefront OBJ model loader.
    Parses OBJ mesh geometry, computes normals if missing, and creates ModernGL VertexArrays.
    """

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.cache: Dict[str, Tuple[moderngl.Buffer, moderngl.VertexArray, int]] = {}

    def load_model(self, file_path: str, program: moderngl.Program) -> Tuple[moderngl.Buffer, moderngl.VertexArray, int]:
        """
        Loads and compiles an OBJ model. Returns (VBO, VAO, vertex_count).
        Uses cache to prevent duplicate loads.
        """
        resolved_path = str(Path(file_path).resolve())
        if resolved_path in self.cache:
            return self.cache[resolved_path]

        logger.info(f"Loading 3D model from OBJ file: {file_path}")
        
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"3D asset file not found: {resolved_path}")

        mesh = MeshData()

        # Parse file line by line
        with open(resolved_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                prefix = parts[0]

                if prefix == "v":
                    mesh.positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif prefix == "vn":
                    mesh.normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif prefix == "vt":
                    mesh.texcoords.append((float(parts[1]), float(parts[2])))
                elif prefix == "f":
                    # Parse face indices. Formats can be:
                    # v, v//vn, v/vt, v/vt/vn
                    face_vertices = []
                    for vert_str in parts[1:]:
                        v_parts = vert_str.split("/")
                        
                        # 1-based index to 0-based offset, handle negative indices if any
                        p_idx = int(v_parts[0]) - 1 if v_parts[0] else -1
                        
                        t_idx = -1
                        if len(v_parts) > 1 and v_parts[1]:
                            t_idx = int(v_parts[1]) - 1
                            
                        n_idx = -1
                        if len(v_parts) > 2 and v_parts[2]:
                            n_idx = int(v_parts[2]) - 1
                            
                        face_vertices.append((p_idx, t_idx, n_idx))

                    # Triangulate polygons (e.g. quads or fans)
                    for i in range(1, len(face_vertices) - 1):
                        mesh.faces.append([
                            face_vertices[0],
                            face_vertices[i],
                            face_vertices[i + 1]
                        ])

        # Convert face vertex records to flat interleaved float buffer
        flat_vertices = self._assemble_buffer(mesh)
        vertex_count = len(flat_vertices) // 8  # 8 floats per vertex: position(3), normal(3), texcoord(2)

        # Upload to GPU
        vbo = self.ctx.buffer(flat_vertices.tobytes())
        vao = self.ctx.vertex_array(
            program,
            [
                (vbo, "3f 3f 2f", "in_position", "in_normal", "in_texcoord")
            ]
        )

        self.cache[resolved_path] = (vbo, vao, vertex_count)
        logger.info(f"Model successfully loaded. Vertex Count: {vertex_count}")
        return vbo, vao, vertex_count

    def _assemble_buffer(self, mesh: MeshData) -> np.ndarray:
        """Assembles flat vertex buffer, calculating normals if missing."""
        buffer_data = []

        for face in mesh.faces:
            # Face vertices
            v0_rec, v1_rec, v2_rec = face

            p0 = mesh.positions[v0_rec[0]]
            p1 = mesh.positions[v1_rec[0]]
            p2 = mesh.positions[v2_rec[0]]

            # Compute face normal if normal index is missing
            computed_normal = (0.0, 0.0, 0.0)
            if v0_rec[2] == -1 or v1_rec[2] == -1 or v2_rec[2] == -1:
                # Normal = cross(p1 - p0, p2 - p0)
                edge1 = np.array(p1) - np.array(p0)
                edge2 = np.array(p2) - np.array(p0)
                n = np.cross(edge1, edge2)
                norm = np.linalg.norm(n)
                if norm > 0.0:
                    computed_normal = tuple(n / norm)
                else:
                    computed_normal = (0.0, 1.0, 0.0)

            # Assemble each of the 3 vertices for the triangle
            for idx, rec in enumerate([v0_rec, v1_rec, v2_rec]):
                p_idx, t_idx, n_idx = rec

                # Position (always present)
                pos = mesh.positions[p_idx]

                # Normal
                if n_idx != -1 and n_idx < len(mesh.normals):
                    norm = mesh.normals[n_idx]
                else:
                    norm = computed_normal

                # Texcoords (default to 0 if missing)
                if t_idx != -1 and t_idx < len(mesh.texcoords):
                    tex = mesh.texcoords[t_idx]
                else:
                    # Simple procedural mapping mapping X, Y positions
                    tex = (pos[0], pos[1])

                buffer_data.extend([
                    pos[0], pos[1], pos[2],
                    norm[0], norm[1], norm[2],
                    tex[0], tex[1]
                ])

        return np.array(buffer_data, dtype="f4")

    def clean_cache(self) -> None:
        """Releases VBO and VAO cached resources."""
        for vbo, vao, _ in self.cache.values():
            vao.release()
            vbo.release()
        self.cache.clear()
        logger.info("AssetLoader cache cleaned.")
