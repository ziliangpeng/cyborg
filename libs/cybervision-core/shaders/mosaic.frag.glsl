#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;
uniform vec2 u_resolution;
uniform float u_blockSize;
uniform float u_mode;
uniform float u_time;

// Hash function for random sampling
float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
  float p3_dot = dot(p3, vec3(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
  vec3 p3_result = p3 + vec3(p3_dot);
  return fract((p3_result.x + p3_result.y) * p3_result.z);
}

// Convert RGB to grayscale for luminance comparison
float luminance(vec3 color) {
  return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

// Sample pixel with bounds checking
vec3 samplePixel(vec2 uv) {
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    return vec3(0.0);
  }
  return texture(u_video, uv).rgb;
}

// Center mode: sample center pixel of block
vec3 centerSample(vec2 blockTopLeftUV, float blockSizeUV) {
  vec2 centerUV = blockTopLeftUV + vec2(blockSizeUV * 0.5);
  return samplePixel(centerUV);
}

// Average mode: average all pixels in block
vec3 averageSample(vec2 blockTopLeftUV, float blockSizeUV) {
  vec3 sum = vec3(0.0);
  float count = 0.0;

  int samples = int(u_blockSize);
  samples = min(samples, 32); // Limit for performance

  for (int dy = 0; dy < 32; dy++) {
    if (dy >= samples) break;
    for (int dx = 0; dx < 32; dx++) {
      if (dx >= samples) break;

      vec2 offset = vec2(float(dx), float(dy)) / u_resolution;
      vec2 sampleUV = blockTopLeftUV + offset;
      vec3 color = samplePixel(sampleUV);
      sum += color;
      count += 1.0;
    }
  }

  return sum / max(count, 1.0);
}

// Min mode: darkest pixel in block
vec3 minSample(vec2 blockTopLeftUV, float blockSizeUV) {
  vec3 minColor = vec3(1.0);
  float minLuminance = 1.0;

  int samples = int(u_blockSize);
  samples = min(samples, 32); // Limit for performance

  for (int dy = 0; dy < 32; dy++) {
    if (dy >= samples) break;
    for (int dx = 0; dx < 32; dx++) {
      if (dx >= samples) break;

      vec2 offset = vec2(float(dx), float(dy)) / u_resolution;
      vec2 sampleUV = blockTopLeftUV + offset;
      vec3 color = samplePixel(sampleUV);
      float lum = luminance(color);

      if (lum < minLuminance) {
        minLuminance = lum;
        minColor = color;
      }
    }
  }

  return minColor;
}

// Max mode: brightest pixel in block
vec3 maxSample(vec2 blockTopLeftUV, float blockSizeUV) {
  vec3 maxColor = vec3(0.0);
  float maxLuminance = 0.0;

  int samples = int(u_blockSize);
  samples = min(samples, 32); // Limit for performance

  for (int dy = 0; dy < 32; dy++) {
    if (dy >= samples) break;
    for (int dx = 0; dx < 32; dx++) {
      if (dx >= samples) break;

      vec2 offset = vec2(float(dx), float(dy)) / u_resolution;
      vec2 sampleUV = blockTopLeftUV + offset;
      vec3 color = samplePixel(sampleUV);
      float lum = luminance(color);

      if (lum > maxLuminance) {
        maxLuminance = lum;
        maxColor = color;
      }
    }
  }

  return maxColor;
}

// Dominant mode: most common quantized color in block
vec3 dominantSample(vec2 blockTopLeftUV, float blockSizeUV) {
  // Simplified approach: just use center sample for now
  // Full histogram is too expensive and causes issues
  return centerSample(blockTopLeftUV, blockSizeUV);
}

// Random mode: random pixel from block (animated with time)
vec3 randomSample(vec2 blockTopLeftUV, float blockSizeUV) {
  // Use block position and time to generate random offset
  vec2 blockPos = blockTopLeftUV * u_resolution;
  vec2 seed = blockPos + vec2(u_time, u_time * 0.7);

  float randX = hash(seed) * u_blockSize;
  float randY = hash(seed + vec2(100.0, 200.0)) * u_blockSize;

  vec2 randomOffset = vec2(randX, randY) / u_resolution;
  vec2 randomUV = blockTopLeftUV + randomOffset;

  return samplePixel(randomUV);
}

void main() {
  vec2 pixelPos = v_texCoord * u_resolution;

  // Determine which block this pixel belongs to
  vec2 blockIndex = floor(pixelPos / u_blockSize);
  vec2 blockTopLeftPixel = blockIndex * u_blockSize;
  vec2 blockTopLeftUV = blockTopLeftPixel / u_resolution;
  float blockSizeUV = u_blockSize / u_resolution.x; // Assume square-ish aspect

  // Sample based on mode
  vec3 color;
  int mode = int(u_mode);

  if (mode == 0) {
    color = centerSample(blockTopLeftUV, blockSizeUV);
  } else if (mode == 1) {
    color = averageSample(blockTopLeftUV, blockSizeUV);
  } else if (mode == 2) {
    color = minSample(blockTopLeftUV, blockSizeUV);
  } else if (mode == 3) {
    color = maxSample(blockTopLeftUV, blockSizeUV);
  } else if (mode == 4) {
    color = dominantSample(blockTopLeftUV, blockSizeUV);
  } else {
    color = randomSample(blockTopLeftUV, blockSizeUV);
  }

  fragColor = vec4(color, 1.0);
}
