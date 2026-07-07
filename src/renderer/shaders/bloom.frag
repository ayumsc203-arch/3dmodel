#version 330 core

in vec2 v_texcoord;
out vec4 f_color;

uniform bool BlendPass;

// Blur Pass Uniforms
uniform bool Horizontal;
uniform sampler2D Image;

// Blend Pass Uniforms
uniform sampler2D SceneTex;
uniform sampler2D BloomTex;
uniform float Exposure;
uniform float BloomIntensity;
uniform float Gamma;

// Weight factors for 9-tap Gaussian blur
const float weights[5] = float[](0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162);

void main() {
    if (!BlendPass) {
        // --- 1. Gaussian Blur Pass ---
        vec2 texOffset = 1.0 / textureSize(Image, 0); // Size of 1 texel
        vec3 result = texture(Image, v_texcoord).rgb * weights[0];
        
        if (Horizontal) {
            for (int i = 1; i < 5; ++i) {
                result += texture(Image, v_texcoord + vec2(texOffset.x * i, 0.0)).rgb * weights[i];
                result += texture(Image, v_texcoord - vec2(texOffset.x * i, 0.0)).rgb * weights[i];
            }
        } else {
            for (int i = 1; i < 5; ++i) {
                result += texture(Image, v_texcoord + vec2(0.0, texOffset.y * i)).rgb * weights[i];
                result += texture(Image, v_texcoord - vec2(0.0, texOffset.y * i)).rgb * weights[i];
            }
        }
        f_color = vec4(result, 1.0);
    } else {
        // --- 2. Composite and Tonemap Pass ---
        vec3 hdrColor = texture(SceneTex, v_texcoord).rgb;
        vec3 bloomColor = texture(BloomTex, v_texcoord).rgb;
        
        // Additive blending of bloom glow scaled by intensity
        vec3 composite = hdrColor + bloomColor * BloomIntensity;
        
        // Exposure tone mapping (Reinhard or ACES style)
        // ACES approximation:
        // vec3 mapped = (composite * (2.51 * composite + 0.03)) / (composite * (2.43 * composite + 0.59) + 0.14);
        // Reinhard tone mapping:
        vec3 mapped = vec3(1.0) - exp(-composite * Exposure);
        
        // Gamma correction
        mapped = pow(mapped, vec3(1.0 / Gamma));
        
        f_color = vec4(mapped, 1.0);
    }
}
