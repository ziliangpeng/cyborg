#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;
uniform vec2 u_resolution;
uniform float u_mode;
uniform float u_intensity;
uniform float u_blockSize;
uniform float u_colorShift;
uniform float u_noiseAmount;
uniform float u_scanlineStrength;
uniform float u_time;

float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
  float p3_dot = dot(p3, vec3(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
  vec3 p3_result = p3 + vec3(p3_dot);
  return fract((p3_result.x + p3_result.y) * p3_result.z);
}

vec3 samplePixel(vec2 pixel) {
  if (pixel.x < 0.0 || pixel.x >= u_resolution.x || pixel.y < 0.0 || pixel.y >= u_resolution.y) {
    return vec3(0.0);
  }
  vec2 uv = pixel / u_resolution;
  return texture(u_video, uv).rgb;
}

void main() {
  vec2 pos = v_texCoord * u_resolution;
  float blockSize = max(u_blockSize, 1.0);
  int mode = int(u_mode);

  vec2 shiftedPos = pos;

  if (mode == 0) {
    float band = floor(pos.y / blockSize);
    float offset = (hash(vec2(band, u_time)) - 0.5) * 2.0 * u_intensity;
    shiftedPos = pos + vec2(offset, 0.0);
  } else if (mode == 1) {
    vec2 block = floor(pos / blockSize);
    float offsetX = (hash(block + vec2(u_time, 13.7)) - 0.5) * 2.0 * u_intensity;
    float offsetY = (hash(block + vec2(17.3, u_time)) - 0.5) * 2.0 * u_intensity;
    shiftedPos = pos + vec2(offsetX, offsetY);
  } else {
    float offset = (hash(vec2(pos.y, u_time)) - 0.5) * 2.0 * (u_intensity * 0.5);
    shiftedPos = pos + vec2(offset, 0.0);
  }

  float shift = u_colorShift;
  float r = samplePixel(shiftedPos + vec2(shift, 0.0)).r;
  float g = samplePixel(shiftedPos).g;
  float b = samplePixel(shiftedPos - vec2(shift, 0.0)).b;

  vec3 color = vec3(r, g, b);

  if (u_scanlineStrength > 0.0) {
    float line = 0.5 + 0.5 * sin((pos.y + u_time * 60.0) * 3.14159);
    color *= 1.0 - u_scanlineStrength * line;
  }

  if (u_noiseAmount > 0.0) {
    float noise = (hash(pos + vec2(u_time * 10.0, u_time * 37.0)) - 0.5) * 2.0;
    color += noise * u_noiseAmount;
  }

  color = clamp(color, 0.0, 1.0);
  fragColor = vec4(color, 1.0);
}
