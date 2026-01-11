// Halftone compute shader
// Converts video input to black dots on white background halftone pattern

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct HalftoneParams {
  sampleSize: f32,
  width: f32,
  height: f32,
  coloredDots: f32,  // Number of colored dots (N)
  time: f32,  // Floor of seconds, changes every second
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
}

@group(0) @binding(2) var<uniform> params: HalftoneParams;

// Sample average brightness in a region
fn sampleBrightness(tex: texture_2d<f32>, center: vec2f, sampleSize: f32) -> f32 {
  let dims = textureDimensions(tex);
  let halfSize = sampleSize * 0.5;

  var sum = 0.0;
  var count = 0.0;

  // Sample a 3x3 grid within the cell for average brightness
  for (var dy = -1.0; dy <= 1.0; dy += 1.0) {
    for (var dx = -1.0; dx <= 1.0; dx += 1.0) {
      let offset = vec2f(dx, dy) * (sampleSize * 0.25);
      let samplePos = center + offset;

      if (samplePos.x >= 0.0 && samplePos.x < f32(dims.x) &&
          samplePos.y >= 0.0 && samplePos.y < f32(dims.y)) {
        let color = textureLoad(tex, vec2i(samplePos), 0);
        // Convert to grayscale (luminance)
        let brightness = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
        sum += brightness;
        count += 1.0;
      }
    }
  }

  return sum / max(count, 1.0);
}

// Simple hash function for pseudo-random number generation
fn hash(p: vec2f) -> f32 {
  let p3 = fract(vec3f(p.x, p.y, p.x) * 0.1031);
  let p3_dot = dot(p3, vec3f(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
  let p3_result = p3 + vec3f(p3_dot, p3_dot, p3_dot);
  return fract((p3_result.x + p3_result.y) * p3_result.z);
}

// Generate random color based on cell position
fn randomColor(cellIndex: vec2f, seed: f32) -> vec3f {
  let r = hash(cellIndex + vec2f(seed, 0.0));
  let g = hash(cellIndex + vec2f(0.0, seed));
  let b = hash(cellIndex + vec2f(seed, seed));
  return vec3f(r, g, b);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outputTex);
  let pos = vec2f(f32(id.x), f32(id.y));

  // Check bounds
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let sampleSize = params.sampleSize;

  // Determine which cell this pixel belongs to
  let cellX = floor(pos.x / sampleSize);
  let cellY = floor(pos.y / sampleSize);
  let cellCenter = vec2f((cellX + 0.5) * sampleSize, (cellY + 0.5) * sampleSize);

  // Sample brightness at cell center
  let brightness = sampleBrightness(inputTex, cellCenter, sampleSize);

  // Calculate dot radius based on darkness (darker = larger dot)
  let maxRadius = sampleSize * 0.5;
  let radius = maxRadius * (1.0 - brightness);

  // Distance from pixel to cell center
  let dist = length(pos - cellCenter);

  // Draw circle using step function
  let inside = step(dist, radius);

  // Determine if this dot should be colored
  let cellIndex = vec2f(cellX, cellY);
  // Include time in hash so selection changes every second
  let cellHash = hash(cellIndex + vec2f(params.time * 1000.0, params.time * 1000.0));
  let totalCells = (params.width / sampleSize) * (params.height / sampleSize);
  let colorProbability = params.coloredDots / totalCells;
  let isColored = cellHash < colorProbability;

  // Choose color: random color if selected, black otherwise
  var dotColor = vec3f(0.0);  // Default black
  if (isColored) {
    dotColor = randomColor(cellIndex, 12.345);
  }

  // Black/colored dot on white background
  let color = mix(vec3f(1.0), dotColor, inside);

  textureStore(outputTex, vec2i(id.xy), vec4f(color, 1.0));
}
