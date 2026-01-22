#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;
uniform vec2 u_center;
uniform float u_radius;
uniform float u_strength;

void main() {
  vec2 uv = v_texCoord;
  vec2 offset = uv - u_center;
  float dist = length(offset);

  if (u_radius > 0.0 && dist < u_radius) {
    float percent = (u_radius - dist) / u_radius;
    float angle = u_strength * percent;
    float s = sin(angle);
    float c = cos(angle);
    vec2 rotated = vec2(
      offset.x * c - offset.y * s,
      offset.x * s + offset.y * c
    );
    uv = u_center + rotated;
  }

  uv = clamp(uv, vec2(0.0), vec2(1.0));
  vec3 color = texture(u_video, uv).rgb;
  fragColor = vec4(color, 1.0);
}
