#version 300 es
precision highp float;
precision highp int;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;
uniform vec2 u_resolution;
uniform float u_cellSize;
uniform float u_colorize;
uniform float u_useGlyphs;

const int ASCII_SIZE = 5;
const int DOT_LEVELS = 8;
const int GLYPH_LEVELS = 10;
const int DOT_MASKS[DOT_LEVELS] = int[DOT_LEVELS](
  0,
  4096,
  131200,
  31744,
  1016800,
  4357252,
  11512810,
  33554431
);
const int GLYPH_MASKS[GLYPH_LEVELS] = int[GLYPH_LEVELS](
  0,
  4194304,
  131200,
  31744,
  4357252,
  1016800,
  10648714,
  27070835,
  23058421,
  33488831
);

float luminance(vec3 color) {
  return dot(color, vec3(0.299, 0.587, 0.114));
}

void main() {
  vec2 pixel = v_texCoord * u_resolution;
  float cellSize = max(u_cellSize, 1.0);
  vec2 cell = floor(pixel / cellSize);
  vec2 cellOrigin = cell * cellSize;

  vec2 centerUV = (cellOrigin + vec2(cellSize * 0.5)) / u_resolution;
  vec3 sampleColor = texture(u_video, centerUV).rgb;
  float lum = luminance(sampleColor);
  int dotLevel = int(floor(clamp(lum, 0.0, 1.0) * float(DOT_LEVELS - 1) + 0.5));
  int glyphLevel = int(floor(clamp(lum, 0.0, 1.0) * float(GLYPH_LEVELS - 1) + 0.5));

  vec2 local = (pixel - cellOrigin) / cellSize;
  int gx = int(floor(clamp(local.x, 0.0, 0.999) * float(ASCII_SIZE)));
  int gy = int(floor(clamp(local.y, 0.0, 0.999) * float(ASCII_SIZE)));
  int idx = gx + gy * ASCII_SIZE;

  int mask = (u_useGlyphs > 0.5) ? GLYPH_MASKS[glyphLevel] : DOT_MASKS[dotLevel];
  float on = float((mask >> idx) & 1);

  vec3 glyphColor = (u_colorize > 0.5) ? sampleColor : vec3(1.0);
  vec3 outColor = glyphColor * on;

  fragColor = vec4(outColor, 1.0);
}
