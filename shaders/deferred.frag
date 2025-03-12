#version 450

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_normal;

void main() {
    f_color = vec4(in_color, 0.6);
    f_normal = vec4(in_normal,1.0);
}
