#version 330 core

layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

in vec3 v_pos;
in vec3 v_normal;
in vec2 v_texcoord;

uniform vec3 CameraPos;
uniform vec3 ObjectColor;
uniform vec3 EmissiveColor;
uniform bool UseTexture;
uniform sampler2D Texture;

// Global Lighting (Overhead directional light)
uniform vec3 DirLightDir;
uniform vec3 DirLightColor;

// Dynamic Hand Point Light (Positioned at palm center, moves with hand)
uniform vec3 PointLightPos;
uniform vec3 PointLightColor;
uniform float PointLightAttenuation; // Constant, linear, quadratic components

uniform float BloomThreshold;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 viewDir = normalize(CameraPos - v_pos);
    
    // Sample texture or use flat object color
    vec3 baseColor = UseTexture ? texture(Texture, v_texcoord).rgb : ObjectColor;
    
    // 1. Ambient lighting
    vec3 ambient = 0.15 * baseColor;
    
    // 2. Directional Light (Overhead sunlight style)
    vec3 dirLightDirNorm = normalize(-DirLightDir);
    float diffDir = max(dot(normal, dirLightDirNorm), 0.0);
    vec3 diffuseDir = diffDir * DirLightColor * baseColor;
    
    vec3 halfDir = normalize(dirLightDirNorm + viewDir);
    float specDir = pow(max(dot(normal, halfDir), 0.0), 32.0); // specular shininess = 32
    vec3 specularDir = specDir * DirLightColor * vec3(0.5);
    
    // 3. Point Light (Glow from hand palm)
    vec3 pointLightDir = PointLightPos - v_pos;
    float dist = length(pointLightDir);
    pointLightDir = normalize(pointLightDir);
    
    float diffPoint = max(dot(normal, pointLightDir), 0.0);
    vec3 diffusePoint = diffPoint * PointLightColor * baseColor;
    
    vec3 halfPoint = normalize(pointLightDir + viewDir);
    float specPoint = pow(max(dot(normal, halfPoint), 0.0), 32.0);
    vec3 specularPoint = specPoint * PointLightColor * vec3(0.5);
    
    // Point light attenuation: 1.0 / (1.0 + linear * d + quadratic * d^2)
    float attenuation = 1.0 / (1.0 + 0.7 * dist + 1.8 * dist * dist);
    vec3 pointLighting = (diffusePoint + specularPoint) * attenuation;
    
    // 4. Emissive glow (for self-illuminating objects like particles or glowing parts)
    vec3 emissive = EmissiveColor;
    
    // Combine all lighting components
    vec3 finalColor = ambient + diffuseDir + specularDir + pointLighting + emissive;
    FragColor = vec4(finalColor, 1.0);
    
    // Extract bright regions for Bloom (based on luminance)
    float brightness = dot(finalColor, vec3(0.2126, 0.7152, 0.0722));
    if (brightness > BloomThreshold) {
        BrightColor = vec4(finalColor, 1.0);
    } else {
        BrightColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
