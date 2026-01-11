#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;
uniform vec2 u_resolution;
uniform float u_dotSize;
uniform float u_useRandomColors;
uniform float u_time;

// Simple hash function for pseudo-random number generation
float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
  float p3_dot = dot(p3, vec3(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
  vec3 p3_result = p3 + vec3(p3_dot);
  return fract((p3_result.x + p3_result.y) * p3_result.z);
}

// Generate random color based on cell position
vec3 randomColor(vec2 cellIndex, float seed) {
  float seedOffset = seed * 127.1;  // Large multiplier breaks linear correlation
  float r = hash(cellIndex + vec2(seedOffset, seedOffset * 0.7));
  float g = hash(cellIndex + vec2(seedOffset * 0.3, seedOffset));
  float b = hash(cellIndex + vec2(seedOffset * 0.5, seedOffset * 0.9));
  return vec3(r, g, b);
}

void main() {
  vec2 pixelPos = v_texCoord * u_resolution;

  // Calculate which cell this pixel belongs to
  vec2 cellIndex = floor(pixelPos / u_dotSize);
  vec2 cellCenter = (cellIndex + 0.5) * u_dotSize;

  // Sample brightness at cell center
  vec2 centerUV = cellCenter / u_resolution;
  vec4 color = texture(u_video, centerUV);
  float brightness = dot(color.rgb, vec3(0.299, 0.587, 0.114));

  // Calculate dot radius based on darkness
  float maxRadius = u_dotSize * 0.5;
  float radius = maxRadius * (1.0 - brightness);

  // Distance from pixel to cell center
  float dist = length(pixelPos - cellCenter);

  // Draw circle
  float inside = step(dist, radius);

  // Determine dot color based on useRandomColors flag
  vec3 dotColor = vec3(0.0);  // Default black

  if (u_useRandomColors > 0.5) {
    // Use random color that changes every second
    dotColor = randomColor(cellIndex, u_time);
  }

  // Colored/black dot on white background
  vec3 outputColor = mix(vec3(1.0), dotColor, inside);
  fragColor = vec4(outputColor, 1.0);
}
