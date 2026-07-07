#version 330 core

uniform mat4 Projection;
uniform mat4 View;

in vec2 in_quad;          // Per-vertex quad offset: [-0.5, -0.5] to [0.5, 0.5]
in vec3 in_pos;           // Per-instance particle 3D position
in vec4 in_color;         // Per-instance particle color
in float in_size;         // Per-instance particle size

out vec2 v_texcoord;
out vec4 v_color;

void main() {
    // Project the 3D position into view space
    vec4 viewPos = View * vec4(in_pos, 1.0);
    
    // Offset view coordinates by quad offsets, scaling by particle size
    viewPos.xy += in_quad * in_size;
    
    v_texcoord = in_quad + vec2(0.5); // Map [-0.5, 0.5] to [0.0, 1.0] for fragment sampling
    v_color = in_color;
    gl_Position = Projection * viewPos;
}
