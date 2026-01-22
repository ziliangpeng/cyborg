#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;
uniform vec2 u_resolution;
uniform float u_vignette;
uniform float u_grain;
uniform float u_time;

float rand(vec2 co) {
  return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  vec3 color = texture(u_video, v_texCoord).rgb;

  vec2 centered = v_texCoord - 0.5;
  float aspect = u_resolution.x / u_resolution.y;
  centered.x *= aspect;
  float dist = length(centered);

  float vignette = smoothstep(0.35, 0.9, dist);
  float vignetteFactor = 1.0 - vignette * u_vignette;

  float grain = (rand(v_texCoord * u_resolution + u_time) - 0.5) * u_grain;

  color *= vignetteFactor;
  color += grain;
  color = clamp(color, 0.0, 1.0);

  fragColor = vec4(color, 1.0);
}
