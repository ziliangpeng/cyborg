#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;
uniform vec2 u_resolution;
uniform float u_intensity;
uniform float u_mode;
uniform float u_centerX;
uniform float u_centerY;

// Sample with bounds checking
float sampleChannelVec(vec2 uv, int channel) {
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    return 0.0;
  }

  return texture(u_video, uv)[channel];
}

void main() {
  vec2 uv = v_texCoord;

  // Calculate offset direction based on mode
  vec2 offsetDir;
  int mode = int(u_mode);

  if (mode == 0) {
    // Radial mode - direction from center to pixel
    vec2 center = vec2(u_centerX, u_centerY);
    vec2 dir = uv - center;
    float dist = length(dir);
    if (dist > 0.0) {
      offsetDir = normalize(dir);
    } else {
      offsetDir = vec2(0.0);
    }
  } else if (mode == 1) {
    // Horizontal mode
    offsetDir = vec2(1.0, 0.0);
  } else {
    // Vertical mode
    offsetDir = vec2(0.0, 1.0);
  }

  // Calculate offset in UV space
  float pixelOffset = u_intensity;
  vec2 uvOffset = offsetDir * pixelOffset / u_resolution;

  // Sample each channel with offset
  vec2 rUV = uv + uvOffset;
  vec2 gUV = uv;
  vec2 bUV = uv - uvOffset;

  float r = sampleChannelVec(rUV, 0);
  float g = sampleChannelVec(gUV, 1);
  float b = sampleChannelVec(bUV, 2);

  fragColor = vec4(r, g, b, 1.0);
}
