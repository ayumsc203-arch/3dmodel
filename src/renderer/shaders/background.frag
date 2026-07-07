#version 330 core

uniform sampler2D Texture;

in vec2 v_texcoord;
out vec4 f_color;

void main() {
    // Sample texture; standard webcam texture has origin at top-left.
    // ModernGL textures might require Y coordinate to be flipped if uploaded raw.
    // We sample straight from the coordinates.
    f_color = texture(Texture, v_texcoord);
}
