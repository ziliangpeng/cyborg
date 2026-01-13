#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;
uniform float u_segments;
uniform float u_rotationSpeed;
uniform float u_time;

const float TWO_PI = 6.28318530718;

void main() {
    vec2 uv = v_texCoord;

    vec2 center = vec2(0.5, 0.5);
    vec2 delta = uv - center;

    float radius = length(delta);
    float angle = atan(delta.y, delta.x);

    float segmentAngle = TWO_PI / u_segments;

    angle = abs(mod(angle, segmentAngle * 2.0) - segmentAngle);

    if (u_rotationSpeed > 0.0) {
        angle += u_time * u_rotationSpeed * 0.5;
    }

    vec2 newDelta = vec2(cos(angle), sin(angle)) * radius;

    vec2 newUV = center + newDelta;

    if (newUV.x < 0.0 || newUV.x > 1.0 || newUV.y < 0.0 || newUV.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        fragColor = texture(u_video, newUV);
    }
}
