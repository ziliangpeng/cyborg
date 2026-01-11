#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;
uniform vec2 u_resolution;
uniform float u_algorithm;
uniform float u_colorCount;
uniform float u_threshold;

vec3 quantize(vec3 color, float levels) {
  float l = max(levels, 2.0);
  return floor(color * l) / (l - 1.0);
}

vec3 generateCentroid(int index, int k) {
  float kf = float(k);
  float idx = float(index);
  float divisions = ceil(pow(kf, 1.0/3.0));
  float r_idx = mod(idx, divisions);
  float g_idx = mod(floor(idx / divisions), divisions);
  float b_idx = floor(idx / (divisions * divisions));
  return vec3(
    (r_idx + 0.5) / divisions,
    (g_idx + 0.5) / divisions,
    (b_idx + 0.5) / divisions
  );
}

vec3 kmeans(vec3 color, int k) {
  float minDist = 999999.0;
  vec3 bestCentroid = vec3(0.0);
  for (int i = 0; i < 32; i++) {
    if (i >= k) break;
    vec3 center = generateCentroid(i, k);
    float dist = length(color - center);
    if (dist < minDist) {
      minDist = dist;
      bestCentroid = center;
    }
  }
  return bestCentroid;
}

vec3 meanshiftSmoothing(vec2 uv, vec3 color, float bandwidth) {
  vec3 currentColor = color;
  vec2 texelSize = 1.0 / u_resolution;
  vec3 weightedSum = vec3(0.0);
  float totalWeight = 0.0;

  for (int dy = -3; dy <= 3; dy++) {
    for (int dx = -3; dx <= 3; dx++) {
      vec2 offset = vec2(float(dx), float(dy)) * texelSize;
      vec3 neighbor = texture(u_video, uv + offset).rgb;
      float colorDist = length(currentColor - neighbor);
      float weight = exp(-(colorDist * colorDist) / (2.0 * bandwidth * bandwidth));
      weightedSum += neighbor * weight;
      totalWeight += weight;
    }
  }

  if (totalWeight > 0.0) {
    currentColor = weightedSum / totalWeight;
  }
  return currentColor;
}

vec3 posterizeEdgeAware(vec2 uv, vec3 color, float levels, float threshold) {
  vec3 quantized = quantize(color, levels);
  vec2 texelSize = 1.0 / u_resolution;
  vec3 weightedSum = vec3(0.0);
  float totalWeight = 0.0;

  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      vec2 offset = vec2(float(dx), float(dy)) * texelSize;
      vec3 neighbor = texture(u_video, uv + offset).rgb;
      vec3 neighborQuantized = quantize(neighbor, levels);

      float spatialDist = length(vec2(float(dx), float(dy)));
      float spatialWeight = exp(-spatialDist * spatialDist / 4.0);
      float colorDist = length(quantized - neighborQuantized);
      float colorWeight = 1.0 - smoothstep(0.0, threshold, colorDist);

      float weight = spatialWeight * colorWeight;
      weightedSum += neighborQuantized * weight;
      totalWeight += weight;
    }
  }

  if (totalWeight > 0.0) {
    return weightedSum / totalWeight;
  }
  return quantized;
}

void main() {
  vec3 color = texture(u_video, v_texCoord).rgb;
  vec3 result;

  int algo = int(u_algorithm);
  int k = int(u_colorCount);

  if (algo == 0) {
    // Quantization (per-channel)
    result = quantize(color, u_colorCount);
  } else if (algo == 1) {
    // Quantization True (true colors)
    result = kmeans(color, k);
  } else if (algo == 2) {
    // K-means (per-channel)
    result = quantize(color, max(u_colorCount, 2.0));
  } else if (algo == 3) {
    // K-means True (true colors)
    result = kmeans(color, k);
  } else if (algo == 4) {
    // Mean shift (per-channel)
    vec3 smoothed = meanshiftSmoothing(v_texCoord, color, u_threshold);
    result = quantize(smoothed, max(u_colorCount, 2.0));
  } else if (algo == 5) {
    // Mean shift True (true colors)
    vec3 smoothed = meanshiftSmoothing(v_texCoord, color, u_threshold);
    result = kmeans(smoothed, k);
  } else if (algo == 6) {
    // Posterize (per-channel)
    result = posterizeEdgeAware(v_texCoord, color, u_colorCount, u_threshold);
  } else {
    // Posterize True (true colors)
    vec3 posterized = posterizeEdgeAware(v_texCoord, color, u_colorCount, u_threshold);
    result = kmeans(posterized, k);
  }

  fragColor = vec4(result, 1.0);
}
