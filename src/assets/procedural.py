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


def generate_phoenix_model(file_path: Path) -> None:
    """Generates a detailed 3D Phoenix Bird model with body, head, beak, wings, and tail."""
    vertices = []
    normals = []
    faces = []

    # 1. Torso/Body (Ellipsoid along Y-axis)
    body_segments = 8
    body_rings = 6
    body_radius = 0.08
    body_height = 0.4
    
    for r in range(body_rings):
        y_frac = r / (body_rings - 1)
        y = -body_height/2.0 + y_frac * body_height
        # Ellipsoid radius profile
        r_scale = np.sin(y_frac * np.pi)
        current_rad = body_radius * r_scale if r_scale > 0 else 0.005
        
        for s in range(body_segments):
            angle = s * 2.0 * np.pi / body_segments
            x = current_rad * np.cos(angle)
            z = current_rad * np.sin(angle)
            vertices.append([x, y, z])
            
            # Normal calculation (outward from center axis)
            nx = np.cos(angle)
            ny = 0.0
            nz = np.sin(angle)
            normals.append([nx, ny, nz])

    # Torso faces
    for r in range(body_rings - 1):
        for s in range(body_segments):
            i0 = r * body_segments + s
            i1 = r * body_segments + (s + 1) % body_segments
            i2 = (r + 1) * body_segments + s
            i3 = (r + 1) * body_segments + (s + 1) % body_segments
            faces.append([i0, i1, i3])
            faces.append([i0, i3, i2])

    # 2. Head (Sphere at top)
    head_start_idx = len(vertices)
    head_radius = 0.06
    head_center = np.array([0.0, body_height/2.0 + 0.04, 0.05])
    head_rings = 5
    head_segments = 8
    
    for r in range(head_rings):
        phi = r * np.pi / (head_rings - 1)
        for s in range(head_segments):
            theta = s * 2.0 * np.pi / head_segments
            
            x = head_radius * np.sin(phi) * np.cos(theta)
            y = head_radius * np.cos(phi)
            z = head_radius * np.sin(phi) * np.sin(theta)
            
            pos = head_center + np.array([x, y, z])
            vertices.append(list(pos))
            
            n = np.array([x, y, z])
            n_norm = np.linalg.norm(n)
            n = n / n_norm if n_norm > 0 else np.array([0.0, 1.0, 0.0])
            normals.append(list(n))
            
    for r in range(head_rings - 1):
        for s in range(head_segments):
            i0 = head_start_idx + r * head_segments + s
            i1 = head_start_idx + r * head_segments + (s + 1) % head_segments
            i2 = head_start_idx + (r + 1) * head_segments + s
            i3 = head_start_idx + (r + 1) * head_segments + (s + 1) % head_segments
            faces.append([i0, i1, i3])
            faces.append([i0, i3, i2])

    # 3. Beak (Cone extending along Z)
    beak_start_idx = len(vertices)
    beak_tip = head_center + np.array([0.0, -0.01, head_radius + 0.07])
    vertices.append(list(beak_tip))
    normals.append([0.0, 0.0, 1.0])
    
    # Beak base circle
    beak_base_segments = 6
    beak_base_center = head_center + np.array([0.0, -0.01, head_radius - 0.01])
    beak_base_radius = 0.025
    
    for s in range(beak_base_segments):
        angle = s * 2.0 * np.pi / beak_base_segments
        x = beak_base_radius * np.cos(angle)
        y = beak_base_radius * np.sin(angle)
        z = 0.0
        pos = beak_base_center + np.array([x, y, z])
        vertices.append(list(pos))
        normals.append([np.cos(angle), np.sin(angle), 0.2])
        
    for s in range(beak_base_segments):
        i0 = beak_start_idx
        i1 = beak_start_idx + 1 + s
        i2 = beak_start_idx + 1 + (s + 1) % beak_base_segments
        faces.append([i0, i2, i1])

    # 4. Tail Feathers (Curved ribbons at the bottom back)
    tail_ribbons = 3
    tail_ribbon_res = 6
    
    for rib in range(tail_ribbons):
        rib_start_idx = len(vertices)
        
        # Splay direction
        rib_splay = (rib - 1) * 0.15 # -0.15, 0.0, 0.15
        
        for i in range(tail_ribbon_res):
            t = i / (tail_ribbon_res - 1)
            
            # Parametric path for tail feather
            y = -body_height/2.0 - t * 0.7
            x = rib_splay * t
            z = -body_radius * 0.8 - (t * t) * 0.4
            
            # Ribbon width
            w = 0.04 * (1.0 - t * 0.7)
            
            # Add left and right points of the ribbon width (oriented perpendicular to length direction)
            p1 = np.array([x - w, y, z])
            p2 = np.array([x + w, y, z])
            
            vertices.append(list(p1))
            vertices.append(list(p2))
            normals.append([0.0, 0.0, -1.0])
            normals.append([0.0, 0.0, -1.0])
            
        for i in range(tail_ribbon_res - 1):
            idx0 = rib_start_idx + 2 * i
            idx1 = rib_start_idx + 2 * i + 1
            idx2 = rib_start_idx + 2 * (i + 1)
            idx3 = rib_start_idx + 2 * (i + 1) + 1
            
            faces.append([idx0, idx2, idx3])
            faces.append([idx0, idx3, idx1])

    # 5. Wings (Layered feathers extending along X)
    wing_res = 4
    num_feathers = 6
    
    for side in [-1, 1]:  # Left, Right
        for f in range(num_feathers):
            wing_start_idx = len(vertices)
            
            # Position offset: extend wing along X axis
            # Feather angle (spread like a fan)
            f_angle = f * 0.18 * side + (0.1 if side == 1 else np.pi - 0.1)
            
            # Length and width
            length = 0.65 - f * 0.05
            width = 0.12 - f * 0.01
            
            # Origin along the body
            origin_x = side * body_radius * 0.8
            origin_y = body_height * 0.1 - f * 0.04
            origin_z = -0.02
            
            # Generate single feather plane
            for i in range(wing_res):
                for j in range(wing_res):
                    u = i / (wing_res - 1)
                    v = j / (wing_res - 1)
                    
                    # Local coordinates
                    local_x = u * length
                    local_y = (v - 0.5) * width * (1.0 - u * 0.5)
                    local_z = -0.05 * (u * u) # Curved tip
                    
                    # Rotate local coordinates
                    rot_x = local_x * np.cos(f_angle) - local_y * np.sin(f_angle)
                    rot_y = local_x * np.sin(f_angle) + local_y * np.cos(f_angle)
                    rot_z = local_z
                    
                    # Absolute coords (offset to be past the flapping boundary of x=0.08)
                    x = origin_x + rot_x
                    # Ensure wings are fully beyond absolute x > 0.08 to animate
                    if side == 1 and x < 0.09:
                        x = 0.09 + (x - origin_x)
                    elif side == -1 and x > -0.09:
                        x = -0.09 + (x - origin_x)
                        
                    y = origin_y + rot_y
                    z = origin_z + rot_z
                    
                    vertices.append([x, y, z])
                    normals.append([0.0, 1.0, 0.0] if side == 1 else [0.0, -1.0, 0.0])
                    
            # Generate faces for the feather plane
            for i in range(wing_res - 1):
                for j in range(wing_res - 1):
                    idx0 = wing_start_idx + i * wing_res + j
                    idx1 = wing_start_idx + i * wing_res + (j + 1)
                    idx2 = wing_start_idx + (i + 1) * wing_res + j
                    idx3 = wing_start_idx + (i + 1) * wing_res + (j + 1)
                    
                    if side == 1:
                        faces.append([idx0, idx2, idx3])
                        faces.append([idx0, idx3, idx1])
                    else:
                        faces.append([idx0, idx3, idx2])
                        faces.append([idx0, idx1, idx3])

    write_obj(file_path, np.array(vertices), np.array(normals), faces)


def generate_dragon_model(file_path: Path) -> None:
    """Generates a detailed 3D Enchanted Dark Dragon model with body/spine, head, horns, tail, wings, and legs."""
    vertices = []
    normals = []
    faces = []

    # 1. Serpentine Body/Spine Tube along the Z-axis (from tail at z=-0.8 to neck/head at z=0.5)
    spine_res = 16
    circle_res = 8
    
    # Path along Z
    z_coords = np.linspace(-0.8, 0.5, spine_res)
    
    # Generate tube vertices
    for idx, z in enumerate(z_coords):
        # S-curve shape for serpentine look
        x_offset = 0.07 * np.sin(z * 4.5)
        y_offset = 0.04 * np.cos(z * 3.0)
        
        # Scale/thickness: thick in center, thin at tail, taper slightly at neck
        # z goes from -0.8 to 0.5 (t ranges from 0 to 1)
        t = (z + 0.8) / 1.3
        # Thickness profile
        if t < 0.2: # tail taper
            radius = 0.02 + 0.05 * (t / 0.2)
        elif t > 0.8: # neck taper
            radius = 0.07 - 0.03 * ((t - 0.8) / 0.2)
        else: # torso
            radius = 0.07
            
        for s in range(circle_res):
            angle = s * 2.0 * np.pi / circle_res
            x = x_offset + radius * np.cos(angle)
            y = y_offset + radius * np.sin(angle)
            
            vertices.append([x, y, z])
            normals.append([np.cos(angle), np.sin(angle), 0.0])

    # Tube faces
    for r in range(spine_res - 1):
        for s in range(circle_res):
            i0 = r * circle_res + s
            i1 = r * circle_res + (s + 1) % circle_res
            i2 = (r + 1) * circle_res + s
            i3 = (r + 1) * circle_res + (s + 1) % circle_res
            faces.append([i0, i1, i3])
            faces.append([i0, i3, i2])

    # 2. Head (at the end of neck, around z=0.5)
    head_start_idx = len(vertices)
    neck_pos = np.array([0.07 * np.sin(0.5 * 4.5), 0.04 * np.cos(0.5 * 3.0), 0.5])
    
    # Let's construct a sleek dragon head (mesh)
    head_verts_local = [
        [0.0, 0.05, 0.05],   # 0: top-back
        [-0.04, 0.02, 0.05], # 1: left-back
        [0.04, 0.02, 0.05],  # 2: right-back
        [0.0, -0.03, 0.05],  # 3: bottom-back
        [0.0, 0.03, 0.18],   # 4: snout-top
        [-0.02, 0.0, 0.16],  # 5: snout-left
        [0.02, 0.0, 0.16],   # 6: snout-right
        [0.0, -0.02, 0.16]   # 7: snout-bottom
    ]
    
    for v in head_verts_local:
        pos = neck_pos + np.array(v)
        vertices.append(list(pos))
        normals.append(list(np.array(v) / np.linalg.norm(v)))
        
    head_faces_local = [
        [0, 1, 4], [1, 5, 4], # Left top
        [0, 4, 2], [2, 4, 6], # Right top
        [1, 3, 5], [3, 7, 5], # Left bottom
        [2, 6, 3], [3, 6, 7], # Right bottom
        [4, 5, 7], [4, 7, 6]  # Snout front
    ]
    
    for hf in head_faces_local:
        faces.append([head_start_idx + hf[0], head_start_idx + hf[1], head_start_idx + hf[2]])

    # 3. Horns (curving back from the head, left and right)
    for side in [-1, 1]:
        horn_start_idx = len(vertices)
        horn_segments = 5
        base_pos = neck_pos + np.array([side * 0.03, 0.04, 0.05])
        
        for i in range(horn_segments):
            u = i / (horn_segments - 1)
            
            offset_z = -u * 0.18
            offset_y = u * 0.12 + 0.03 * (u * u)
            offset_x = side * 0.01 + side * u * 0.04
            
            rad = 0.018 * (1.0 - u * 0.8) # tapering
            
            for s in range(4):
                angle = s * 2.0 * np.pi / 4
                hx = offset_x + rad * np.cos(angle)
                hy = offset_y + rad * np.sin(angle)
                hz = offset_z
                
                pos = base_pos + np.array([hx, hy, hz])
                vertices.append(list(pos))
                normals.append([side * np.cos(angle), np.sin(angle), -0.5])
                
        for r in range(horn_segments - 1):
            for s in range(4):
                i0 = horn_start_idx + r * 4 + s
                i1 = horn_start_idx + r * 4 + (s + 1) % 4
                i2 = horn_start_idx + (r + 1) * 4 + s
                i3 = horn_start_idx + (r + 1) * 4 + (s + 1) % 4
                faces.append([i0, i1, i3])
                faces.append([i0, i3, i2])

    # 4. Tail Tip (Spade/Arrow shape at the end of the tail z=-0.8)
    tail_start_idx = len(vertices)
    tail_pos = np.array([0.07 * np.sin(-0.8 * 4.5), 0.04 * np.cos(-0.8 * 3.0), -0.8])
    
    spade_verts = [
        [0.0, 0.0, 0.0],          # 0: base
        [-0.08, 0.0, -0.08],      # 1: left tip
        [0.0, 0.0, -0.16],        # 2: end tip
        [0.08, 0.0, -0.08],       # 3: right tip
        [0.0, 0.03, -0.08],       # 4: top center
        [0.0, -0.03, -0.08]       # 5: bottom center
    ]
    
    for v in spade_verts:
        pos = tail_pos + np.array(v)
        vertices.append(list(pos))
        normals.append([0.0, 0.0, -1.0])
        
    spade_faces = [
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], # Top pyramid
        [0, 5, 1], [1, 5, 2], [2, 5, 3], [3, 5, 0]  # Bottom pyramid
    ]
    for sf in spade_faces:
        faces.append([tail_start_idx + sf[0], tail_start_idx + sf[1], tail_start_idx + sf[2]])

    # 5. Wings (Large bat-like wings on left and right)
    wing_joint_z = 0.0
    wing_joint_x = 0.07
    wing_joint_y = 0.0
    
    for side in [-1, 1]:
        wing_start_idx = len(vertices)
        joint_pos = np.array([side * wing_joint_x, wing_joint_y, wing_joint_z])
        num_wing_pts = 4
        
        for finger in range(3):
            if finger == 0:
                dir_vec = np.array([side * 1.2, 0.5, -0.1])
            elif finger == 1:
                dir_vec = np.array([side * 1.3, 0.0, -0.3])
            else:
                dir_vec = np.array([side * 0.9, -0.4, -0.5])
                
            for col in range(num_wing_pts):
                u = col / (num_wing_pts - 1)
                curve_offset = np.array([0.0, -0.08 * (u * u), 0.05 * np.sin(u * np.pi)])
                pos = joint_pos + u * dir_vec + curve_offset
                
                x = pos[0]
                if side == 1 and x < 0.09:
                    x = 0.09
                elif side == -1 and x > -0.09:
                    x = -0.09
                pos[0] = x
                
                vertices.append(list(pos))
                normals.append([0.0, 1.0, 0.0] if side == 1 else [0.0, -1.0, 0.0])
                
        for f in range(2):
            for col in range(num_wing_pts - 1):
                idx0 = wing_start_idx + f * num_wing_pts + col
                idx1 = wing_start_idx + f * num_wing_pts + (col + 1)
                idx2 = wing_start_idx + (f + 1) * num_wing_pts + col
                idx3 = wing_start_idx + (f + 1) * num_wing_pts + (col + 1)
                
                if side == 1:
                    faces.append([idx0, idx2, idx3])
                    faces.append([idx0, idx3, idx1])
                else:
                    faces.append([idx0, idx3, idx2])
                    faces.append([idx0, idx1, idx3])

    write_obj(file_path, np.array(vertices), np.array(normals), faces)


def generate_sphere_model(file_path: Path) -> None:
    """Generates a detailed 3D UV Sphere model."""
    vertices = []
    normals = []
    faces = []
    
    rings = 16
    segments = 16
    radius = 0.12
    
    for r in range(rings):
        phi = r * np.pi / (rings - 1)
        for s in range(segments):
            theta = s * 2.0 * np.pi / segments
            
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.cos(phi)
            z = radius * np.sin(phi) * np.sin(theta)
            
            vertices.append([x, y, z])
            normals.append([x/radius, y/radius, z/radius])
            
    for r in range(rings - 1):
        for s in range(segments):
            i0 = r * segments + s
            i1 = r * segments + (s + 1) % segments
            i2 = (r + 1) * segments + s
            i3 = (r + 1) * segments + (s + 1) % segments
            faces.append([i0, i1, i3])
            faces.append([i0, i3, i2])
            
    write_obj(file_path, np.array(vertices), np.array(normals), faces)


def generate_all_procedural_models() -> None:
    """Generates all orchid, wing, butterfly, phoenix, dragon, and sphere files in the assets directory."""
    base_dir = Path(__file__).parent.parent.resolve() / "assets" / "models"
    
    models = {
        "orchid.obj": generate_orchid_model,
        "wing.obj": generate_wing_model,
        "butterfly.obj": generate_butterfly_model,
        "phoenix.obj": generate_phoenix_model,
        "dragon.obj": generate_dragon_model,
        "sphere.obj": generate_sphere_model
    }

    for name, func in models.items():
        file_path = base_dir / name
        if not file_path.exists():
            logger.info(f"Generating procedural model '{name}'...")
            func(file_path)
        else:
            logger.info(f"Procedural model '{name}' already exists.")
