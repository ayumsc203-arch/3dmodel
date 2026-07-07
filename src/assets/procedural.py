import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger("ProceduralGenerator")


def write_obj(file_path: Path, vertices: np.ndarray, normals: np.ndarray, faces: list) -> None:
    """Helper to write vertex arrays to a standard Wavefront OBJ file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(f"# Procedural OBJ Generation - {file_path.name}\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        # Write normals
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            
        # Write faces (1-indexed, format: v//vn since we write normals but no tex coords)
        for face in faces:
            f.write("f " + " ".join([f"{idx + 1}//{idx + 1}" for idx in face]) + "\n")
            
    logger.info(f"Procedurally generated and saved OBJ model to: {file_path}")


def generate_butterfly_model(file_path: Path) -> None:
    """Generates a detailed 3D Butterfly model with body and wings."""
    vertices = []
    normals = []
    faces = []

    # 1. Generate Body (Cylinder along Y axis)
    body_radius = 0.08
    body_height = 1.0
    segments = 8
    
    # Vertices
    for y in np.linspace(-body_height/2, body_height/2, 5):
        for s in range(segments):
            angle = s * 2.0 * np.pi / segments
            x = body_radius * np.cos(angle)
            z = body_radius * np.sin(angle)
            vertices.append([x, y, z])
            normals.append([np.cos(angle), 0.0, np.sin(angle)])

    # Body Faces
    for r in range(4):
        for s in range(segments):
            # Vertex indices
            i0 = r * segments + s
            i1 = r * segments + (s + 1) % segments
            i2 = (r + 1) * segments + s
            i3 = (r + 1) * segments + (s + 1) % segments
            faces.append([i0, i1, i3])
            faces.append([i0, i3, i2])

    body_vertex_count = len(vertices)

    # 2. Generate Wings (Left & Right flat surfaces with fold angles)
    # Grid of wing points
    wing_res = 6
    for side in [-1, 1]:  # -1 for Left, 1 for Right
        wing_vertex_start = len(vertices)
        
        for u_val in np.linspace(0.0, 1.0, wing_res):
            for v_val in np.linspace(0.0, 1.0, wing_res):
                # Parametric wing layout
                # Span X, Cord Y, Elevation Z
                x = side * (u_val * 1.5)
                # Wing shape: wider in the middle
                y = (v_val - 0.5) * (1.2 - 0.4 * abs(u_val - 0.5))
                # Wing fold: curved upwards
                z = 0.4 * (u_val * u_val)
                
                # Normal vector (pointing roughly upwards)
                norm = np.array([-side * 0.4 * u_val, 0.0, 1.0])
                norm = norm / np.linalg.norm(norm)
                
                vertices.append([x, y, z])
                normals.append(list(norm))

        # Generate wing faces
        for i in range(wing_res - 1):
            for j in range(wing_res - 1):
                idx0 = wing_vertex_start + i * wing_res + j
                idx1 = wing_vertex_start + i * wing_res + (j + 1)
                idx2 = wing_vertex_start + (i + 1) * wing_res + j
                idx3 = wing_vertex_start + (i + 1) * wing_res + (j + 1)
                
                if side == 1:
                    faces.append([idx0, idx2, idx3])
                    faces.append([idx0, idx3, idx1])
                else:
                    faces.append([idx0, idx3, idx2])
                    faces.append([idx0, idx1, idx3])

    write_obj(file_path, np.array(vertices), np.array(normals), faces)


def generate_wing_model(file_path: Path) -> None:
    """Generates a detailed 3D Fantasy Wing model consisting of layered feathers."""
    vertices = []
    normals = []
    faces = []

    # Feathers count
    num_feathers = 10
    
    for f in range(num_feathers):
        v_start = len(vertices)
        
        # Position offset: extend wing along X axis
        angle = f * 0.15
        length = 1.6 - f * 0.1
        width = 0.3 - f * 0.02
        
        origin_x = f * 0.15
        origin_y = f * 0.08
        origin_z = f * 0.04
        
        # Generate single feather plane (curved)
        feather_res = 4
        for i in range(feather_res):
            for j in range(feather_res):
                u = i / (feather_res - 1)
                v = j / (feather_res - 1)
                
                # Rotate local coords
                local_x = u * length
                local_y = (v - 0.5) * width
                local_z = -0.15 * (u * u) # Curved tip
                
                # Apply rotation
                rot_x = local_x * np.cos(angle) - local_y * np.sin(angle)
                rot_y = local_x * np.sin(angle) + local_y * np.cos(angle)
                rot_z = local_z
                
                x = origin_x + rot_x
                y = origin_y + rot_y
                z = origin_z + rot_z
                
                vertices.append([x, y, z])
                normals.append([0.0, 0.0, 1.0])

        # Generate feather faces
        for i in range(feather_res - 1):
            for j in range(feather_res - 1):
                idx0 = v_start + i * feather_res + j
                idx1 = v_start + i * feather_res + (j + 1)
                idx2 = v_start + (i + 1) * feather_res + j
                idx3 = v_start + (i + 1) * feather_res + (j + 1)
                
                faces.append([idx0, idx2, idx3])
                faces.append([idx0, idx3, idx1])

    write_obj(file_path, np.array(vertices), np.array(normals), faces)


def generate_orchid_model(file_path: Path) -> None:
    """Generates a detailed 3D Orchid plant model with a central stem and petals."""
    vertices = []
    normals = []
    faces = []

    # 1. Stem (Spiral Tube going upwards)
    stem_radius = 0.05
    stem_height = 1.2
    stem_segments = 10
    circle_pts = 6

    for i in range(stem_segments):
        h = i * stem_height / (stem_segments - 1)
        # Slight curved bend to make stem look organic
        offset_x = 0.15 * np.sin(h * 3.0)
        offset_z = 0.15 * (1.0 - np.cos(h * 2.0))
        
        for c in range(circle_pts):
            angle = c * 2.0 * np.pi / circle_pts
            x = offset_x + stem_radius * np.cos(angle)
            y = h - stem_height / 2.0
            z = offset_z + stem_radius * np.sin(angle)
            
            vertices.append([x, y, z])
            normals.append([np.cos(angle), 0.0, np.sin(angle)])

    for r in range(stem_segments - 1):
        for s in range(circle_pts):
            i0 = r * circle_pts + s
            i1 = r * circle_pts + (s + 1) % circle_pts
            i2 = (r + 1) * circle_pts + s
            i3 = (r + 1) * circle_pts + (s + 1) % circle_pts
            faces.append([i0, i1, i3])
            faces.append([i0, i3, i2])

    # 2. Add Flowers (Flower shapes along the stem)
    flower_nodes = [3, 6, 9]  # Add flowers at segment heights
    for node in flower_nodes:
        h = node * stem_height / (stem_segments - 1)
        center_x = 0.15 * np.sin(h * 3.0)
        center_z = 0.15 * (1.0 - np.cos(h * 2.0))
        center_y = h - stem_height / 2.0

        # Create 5 petals for the orchid flower
        num_petals = 5
        for p in range(num_petals):
            v_start = len(vertices)
            petal_angle = p * 2.0 * np.pi / num_petals
            
            # Petal quad mesh (3x3 grid)
            petal_res = 3
            for i in range(petal_res):
                for j in range(petal_res):
                    u = i / (petal_res - 1)
                    v = j / (petal_res - 1)
                    
                    # Petal shape (leaf curved outwards)
                    length = 0.35
                    width = 0.15
                    
                    local_x = u * length
                    local_y = (v - 0.5) * width * (1.0 - abs(u - 0.5) * 2.0)
                    local_z = 0.08 * np.sin(u * np.pi)
                    
                    # Rotate petal
                    rot_x = local_x * np.cos(petal_angle) - local_y * np.sin(petal_angle)
                    rot_y = local_x * np.sin(petal_angle) + local_y * np.cos(petal_angle)
                    rot_z = local_z
                    
                    x = center_x + rot_x
                    y = center_y + rot_y
                    z = center_z + rot_z
                    
                    vertices.append([x, y, z])
                    normals.append([np.cos(petal_angle), np.sin(petal_angle), 0.5])

            # Petal faces
            for i in range(petal_res - 1):
                for j in range(petal_res - 1):
                    idx0 = v_start + i * petal_res + j
                    idx1 = v_start + i * petal_res + (j + 1)
                    idx2 = v_start + (i + 1) * petal_res + j
                    idx3 = v_start + (i + 1) * petal_res + (j + 1)
                    
                    faces.append([idx0, idx2, idx3])
                    faces.append([idx0, idx3, idx1])

    write_obj(file_path, np.array(vertices), np.array(normals), faces)


def generate_all_procedural_models() -> None:
    """Generates all orchid, wing, and butterfly files in the assets directory."""
    base_dir = Path(__file__).parent.parent.resolve() / "assets" / "models"
    
    models = {
        "orchid.obj": generate_orchid_model,
        "wing.obj": generate_wing_model,
        "butterfly.obj": generate_butterfly_model
    }

    for name, func in models.items():
        file_path = base_dir / name
        if not file_path.exists():
            logger.info(f"Generating procedural model '{name}'...")
            func(file_path)
        else:
            logger.info(f"Procedural model '{name}' already exists.")
