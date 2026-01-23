#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;
uniform vec2 u_resolution;
uniform float u_radius;
uniform float u_levels;

const int MAX_RADIUS = 6;
const int MAX_LEVELS = 8;

float luminance(vec3 color) {
  return dot(color, vec3(0.299, 0.587, 0.114));
}

void main() {
  int radius = int(clamp(u_radius, 1.0, float(MAX_RADIUS)));
  int levels = int(clamp(u_levels, 1.0, float(MAX_LEVELS)));

  int counts[MAX_LEVELS];
  vec3 sums[MAX_LEVELS];
  for (int i = 0; i < MAX_LEVELS; i++) {
    counts[i] = 0;
    sums[i] = vec3(0.0);
  }

  vec2 pixel = v_texCoord * u_resolution;

  for (int dy = -MAX_RADIUS; dy <= MAX_RADIUS; dy++) {
    for (int dx = -MAX_RADIUS; dx <= MAX_RADIUS; dx++) {
      if (abs(dx) > radius || abs(dy) > radius) {
        continue;
      }

      vec2 sampleUV = (pixel + vec2(float(dx), float(dy))) / u_resolution;
      sampleUV = clamp(sampleUV, vec2(0.0), vec2(1.0));
      vec3 color = texture(u_video, sampleUV).rgb;
      float lum = luminance(color);
      int bucket = int(floor(clamp(lum, 0.0, 1.0) * float(levels - 1) + 0.5));

      counts[bucket] += 1;
      sums[bucket] += color;
    }
  }

  int maxCount = 0;
  int maxIndex = 0;
  for (int i = 0; i < MAX_LEVELS; i++) {
    if (i >= levels) {
      break;
    }
    if (counts[i] > maxCount) {
      maxCount = counts[i];
      maxIndex = i;
    }
  }

  vec3 outColor = sums[maxIndex] / float(max(maxCount, 1));
  fragColor = vec4(outColor, 1.0);
}
