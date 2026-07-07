#version 330 core

uniform mat4 Mvp;
uniform mat4 Model;
uniform mat3 NormalMatrix;
uniform float Time;
uniform bool AnimateWings;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord;

out vec3 v_pos;
out vec3 v_normal;
out vec2 v_texcoord;

void main() {
    vec3 pos = in_position;
    
    // Shader-based animation for fluttering wings (e.g. Butterfly wings)
    if (AnimateWings && abs(pos.x) > 0.08) {
        // Wing rotation angle oscillates with high frequency
        float angle = sin(Time * 24.0) * 0.7 * abs(pos.x);
        float c = cos(angle);
        float s = sin(angle);
        
        // Rotate around Y-axis (wing joint lies along Y-axis)
        float x_new = pos.x * c - pos.z * s;
        float z_new = pos.x * s + pos.z * c;
        pos.x = x_new;
        pos.z = z_new;
    }
    
    v_pos = (Model * vec4(pos, 1.0)).xyz;
    v_normal = normalize(NormalMatrix * in_normal);
    v_texcoord = in_texcoord;
    gl_Position = Mvp * vec4(pos, 1.0);
}
