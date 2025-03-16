#version 450

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;

layout(set = 0, binding = 2) uniform Directional_Light_Data {
    vec4 position;  // w component is unused but kept for alignment
    vec3 color;
} directional;

layout(location = 0) out vec4 f_color;

void main() {
    // Load normal once and reuse
    vec3 normal = normalize(subpassLoad(u_normals).rgb);
    
    // Calculate light direction (position is a direction vector for directional lights)
    vec3 light_direction = normalize(directional.position.xyz);
    
    // Calculate diffuse lighting
    float directional_intensity = max(dot(normal, light_direction), 0.0);
    
    // Calculate final lighting contribution
    vec3 directional_color = directional.color.rgb * directional_intensity;
    
    // Load base color and compute final color
    vec3 base_color = subpassLoad(u_color).rgb;
    vec3 combined_color = base_color * directional_color;
    
    f_color = vec4(combined_color, 0.2);
}
