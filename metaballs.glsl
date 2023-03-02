#version 330

#if defined COMPUTE_SHADER

// Model geometry
// in vec3 in_position;

layout (std430, binding = 0) readonly buffer in0
    {
        vec4 points[];
    };

out vec3 pos;
// out vec3 normal;
// out vec3 col;or

void main() {
    
}
dis

#elif defined FRAGMENT_SHADER

out vec4 fragColor;

// in vec3 pos;
// in vec3 normal;
// in vec3 color;

void main() {
    // float l = dot(normalize(-pos), normalize(normal));
    // fragColor = vec4(color * (0.25 + abs(l) * 0.75), 1.0);
    fragColor = vec4(1.0,0.5,0.5,1.0);
}
#endif