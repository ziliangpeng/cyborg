#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;
uniform float u_levels;

float luminance(vec3 color) {
  return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

void main() {
  vec3 color = texture(u_video, v_texCoord).rgb;
  float lum = luminance(color);
  float levels = max(u_levels, 2.0);
  float quant = floor(lum * (levels - 1.0) + 0.5) / (levels - 1.0);

  if (lum > 0.0) {
    color *= quant / lum;
  } else {
    color = vec3(0.0);
  }

  color = clamp(color, 0.0, 1.0);
  fragColor = vec4(color, 1.0);
}
