#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;
uniform vec2 u_resolution;
uniform float u_scale;
uniform float u_levels;

const float bayer[16] = float[16](
  0.0, 8.0, 2.0, 10.0,
  12.0, 4.0, 14.0, 6.0,
  3.0, 11.0, 1.0, 9.0,
  15.0, 7.0, 13.0, 5.0
);

float luminance(vec3 color) {
  return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

float bayerThreshold(int x, int y) {
  int idx = x + y * 4;
  return (bayer[idx] + 0.5) / 16.0;
}

void main() {
  vec3 color = texture(u_video, v_texCoord).rgb;
  float lum = luminance(color);

  float scale = max(u_scale, 1.0);
  float levels = max(u_levels, 2.0);
  float maxLevel = levels - 1.0;

  vec2 pixel = v_texCoord * u_resolution;
  vec2 cell = floor(pixel / scale);
  int x = int(mod(cell.x, 4.0));
  int y = int(mod(cell.y, 4.0));
  float threshold = bayerThreshold(x, y);

  float scaled = lum * maxLevel;
  float base = floor(scaled);
  float frac = scaled - base;
  if (frac > threshold) {
    base = min(base + 1.0, maxLevel);
  }

  float quant = base / maxLevel;
  fragColor = vec4(vec3(quant), 1.0);
}
