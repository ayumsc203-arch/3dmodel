#version 330 core

layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

in vec2 v_texcoord;
in vec4 v_color;

uniform float BloomThreshold;

void main() {
    // Generate soft radial dropoff for particle shapes
    vec2 temp_uv = v_texcoord - vec2(0.5);
    float dist = length(temp_uv);
    
    if (dist > 0.5) {
        discard;
    }
    
    // Soft dropoff from center to edge
    float intensity = smoothstep(0.5, 0.0, dist);
    float alpha = intensity * v_color.a;
    
    vec4 color = vec4(v_color.rgb, alpha);
    FragColor = color;
    
    // Output glowing particles to Bloom bright target
    float luminance = dot(color.rgb * alpha, vec3(0.2126, 0.7152, 0.0722));
    if (luminance > BloomThreshold) {
        // Boost emissive component for high-intensity glow
        BrightColor = vec4(color.rgb * alpha * 2.0, 1.0);
    } else {
        BrightColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
