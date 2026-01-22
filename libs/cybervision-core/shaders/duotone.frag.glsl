#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;
uniform vec3 u_shadowColor;
uniform vec3 u_highlightColor;

float luminance(vec3 color) {
  return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

void main() {
  vec3 color = texture(u_video, v_texCoord).rgb;
  float lum = luminance(color);
  vec3 duo = mix(u_shadowColor, u_highlightColor, lum);
  fragColor = vec4(duo, 1.0);
}
