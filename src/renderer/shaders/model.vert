#version 330 core

uniform mat4 Mvp;
uniform mat4 Model;
uniform mat3 NormalMatrix;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord;

out vec3 v_pos;
out vec3 v_normal;
out vec2 v_texcoord;

void main() {
    v_pos = (Model * vec4(in_position, 1.0)).xyz;
    v_normal = normalize(NormalMatrix * in_normal);
    v_texcoord = in_texcoord;
    gl_Position = Mvp * vec4(in_position, 1.0);
}
