// Clustering compute shader
// Implements multiple color clustering algorithms

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct ClusteringParams {
  algorithm: f32,    // 0=quantization, 1=kmeans, 2=meanshift, 3=posterize
  colorCount: f32,   // Number of colors (K for k-means, levels for quantization)
  threshold: f32,    // Color similarity threshold
  _pad0: f32,
  width: f32,
  height: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(2) var<uniform> params: ClusteringParams;

// ============ QUANTIZATION ============
fn quantize(color: vec3f, levels: f32) -> vec3f {
  let l = max(levels, 2.0);
  return floor(color * l) / (l - 1.0);
}

// ============ K-MEANS (pre-seeded centroids) ============
fn generateCentroid(index: u32, k: u32) -> vec3f {
  // Generate K centroids evenly distributed in RGB cube
  let kf = f32(k);
  let idx = f32(index);

  // Simple linear distribution in color space
  let divisions = ceil(pow(kf, 1.0/3.0));
  let r_idx = idx % divisions;
  let g_idx = floor(idx / divisions) % divisions;
  let b_idx = floor(idx / (divisions * divisions));

  return vec3f(
    (r_idx + 0.5) / divisions,
    (g_idx + 0.5) / divisions,
    (b_idx + 0.5) / divisions
  );
}

fn kmeans(color: vec3f, k: u32) -> vec3f {
  var minDist = 999999.0;
  var bestCentroid = vec3f(0.0);

  for (var i: u32 = 0u; i < k; i++) {
    let centroid = generateCentroid(i, k);
    let dist = length(color - centroid);
    if (dist < minDist) {
      minDist = dist;
      bestCentroid = centroid;
    }
  }

  return bestCentroid;
}

// ============ MEAN SHIFT (approximation) ============
fn meanshiftSmoothing(pos: vec2i, color: vec3f, bandwidth: f32) -> vec3f {
  let dims = textureDimensions(inputTex);
  var currentColor = color;

  // Single iteration mean shift
  let radius = 3;  // Spatial radius
  var weightedSum = vec3f(0.0);
  var totalWeight = 0.0;

  for (var dy = -radius; dy <= radius; dy++) {
    for (var dx = -radius; dx <= radius; dx++) {
      let samplePos = pos + vec2i(dx, dy);
      if (samplePos.x >= 0 && samplePos.x < i32(dims.x) &&
          samplePos.y >= 0 && samplePos.y < i32(dims.y)) {
        let neighbor = textureLoad(inputTex, samplePos, 0).rgb;
        let colorDist = length(currentColor - neighbor);

        // Gaussian kernel
        let weight = exp(-(colorDist * colorDist) / (2.0 * bandwidth * bandwidth));
        weightedSum += neighbor * weight;
        totalWeight += weight;
      }
    }
  }

  if (totalWeight > 0.0) {
    currentColor = weightedSum / totalWeight;
  }

  return currentColor;
}

// ============ POSTERIZE (edge-aware quantization) ============
fn posterizeEdgeAware(pos: vec2i, color: vec3f, levels: f32, threshold: f32) -> vec3f {
  let dims = textureDimensions(inputTex);
  let quantized = quantize(color, levels);

  // Edge-aware smoothing
  let radius = 2;
  var weightedSum = vec3f(0.0);
  var totalWeight = 0.0;

  for (var dy = -radius; dy <= radius; dy++) {
    for (var dx = -radius; dx <= radius; dx++) {
      let samplePos = pos + vec2i(dx, dy);
      if (samplePos.x >= 0 && samplePos.x < i32(dims.x) &&
          samplePos.y >= 0 && samplePos.y < i32(dims.y)) {
        let neighbor = textureLoad(inputTex, samplePos, 0).rgb;
        let neighborQuantized = quantize(neighbor, levels);

        // Spatial weight (Gaussian)
        let spatialDist = length(vec2f(f32(dx), f32(dy)));
        let spatialWeight = exp(-spatialDist * spatialDist / 4.0);

        // Color similarity weight
        let colorDist = length(quantized - neighborQuantized);
        let colorWeight = 1.0 - smoothstep(0.0, threshold, colorDist);

        let weight = spatialWeight * colorWeight;
        weightedSum += neighborQuantized * weight;
        totalWeight += weight;
      }
    }
  }

  if (totalWeight > 0.0) {
    return weightedSum / totalWeight;
  }
  return quantized;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outputTex);

  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let pos = vec2i(id.xy);
  let color = textureLoad(inputTex, pos, 0).rgb;

  var outputColor: vec3f;

  let algo = u32(params.algorithm);
  let k = u32(params.colorCount);

  if (algo == 0u) {
    // Quantization / K-means (per-channel)
    outputColor = quantize(color, params.colorCount);
  } else if (algo == 1u) {
    // Quantization / K-means (true colors)
    outputColor = kmeans(color, k);
  } else if (algo == 2u) {
    // Mean shift (per-channel)
    let smoothed = meanshiftSmoothing(pos, color, params.threshold);
    outputColor = quantize(smoothed, params.colorCount);
  } else if (algo == 3u) {
    // Mean shift (true colors)
    let smoothed = meanshiftSmoothing(pos, color, params.threshold);
    outputColor = kmeans(smoothed, k);
  } else if (algo == 4u) {
    // Posterize (per-channel)
    outputColor = posterizeEdgeAware(pos, color, params.colorCount, params.threshold);
  } else {
    // Posterize (true colors)
    let posterized = posterizeEdgeAware(pos, color, params.colorCount, params.threshold);
    outputColor = kmeans(posterized, k);
  }

  textureStore(outputTex, pos, vec4f(outputColor, 1.0));
}
