// Edge detection compute shader
// Supports Sobel, Prewitt, Laplacian, and Canny-style edge detection

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct EdgeParams {
  algorithm: f32,      // 0=Sobel, 1=Prewitt, 2=Laplacian, 3=Canny
  threshold: f32,      // Edge threshold
  showOverlay: f32,    // 0=black bg, 1=overlay on original
  invert: f32,         // Invert colors
  edgeColorR: f32,     // RGB edge color
  edgeColorG: f32,
  edgeColorB: f32,
  _pad1: f32,
  width: f32,
  height: f32,
  thickness: f32,      // 1-3 for edge dilation
  _pad2: f32,
}

@group(0) @binding(2) var<uniform> params: EdgeParams;

// Convert RGB to grayscale using luminance formula
fn toGrayscale(color: vec3f) -> f32 {
  return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

// Sample pixel with bounds checking
fn samplePixel(tex: texture_2d<f32>, pos: vec2i) -> f32 {
  let dims = textureDimensions(tex);
  if (pos.x < 0 || pos.x >= i32(dims.x) || pos.y < 0 || pos.y >= i32(dims.y)) {
    return 0.0;
  }
  let color = textureLoad(tex, pos, 0);
  return toGrayscale(color.rgb);
}

// Sobel edge detection
fn sobelEdge(tex: texture_2d<f32>, pos: vec2i) -> f32 {
  // Sobel kernels
  // Gx:             Gy:
  // [-1  0  1]      [-1 -2 -1]
  // [-2  0  2]      [ 0  0  0]
  // [-1  0  1]      [ 1  2  1]

  let tl = samplePixel(tex, pos + vec2i(-1, -1));
  let tc = samplePixel(tex, pos + vec2i(0, -1));
  let tr = samplePixel(tex, pos + vec2i(1, -1));
  let ml = samplePixel(tex, pos + vec2i(-1, 0));
  let mr = samplePixel(tex, pos + vec2i(1, 0));
  let bl = samplePixel(tex, pos + vec2i(-1, 1));
  let bc = samplePixel(tex, pos + vec2i(0, 1));
  let br = samplePixel(tex, pos + vec2i(1, 1));

  let gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
  let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;

  return sqrt(gx * gx + gy * gy);
}

// Prewitt edge detection
fn prewittEdge(tex: texture_2d<f32>, pos: vec2i) -> f32 {
  // Prewitt kernels (uniform weights)
  // Gx:             Gy:
  // [-1  0  1]      [-1 -1 -1]
  // [-1  0  1]      [ 0  0  0]
  // [-1  0  1]      [ 1  1  1]

  let tl = samplePixel(tex, pos + vec2i(-1, -1));
  let tc = samplePixel(tex, pos + vec2i(0, -1));
  let tr = samplePixel(tex, pos + vec2i(1, -1));
  let ml = samplePixel(tex, pos + vec2i(-1, 0));
  let mr = samplePixel(tex, pos + vec2i(1, 0));
  let bl = samplePixel(tex, pos + vec2i(-1, 1));
  let bc = samplePixel(tex, pos + vec2i(0, 1));
  let br = samplePixel(tex, pos + vec2i(1, 1));

  let gx = -tl - ml - bl + tr + mr + br;
  let gy = -tl - tc - tr + bl + bc + br;

  return sqrt(gx * gx + gy * gy);
}

// Laplacian edge detection
fn laplacianEdge(tex: texture_2d<f32>, pos: vec2i) -> f32 {
  // Laplacian kernel (detects edges in all directions)
  // [ 0 -1  0]
  // [-1  4 -1]
  // [ 0 -1  0]

  let center = samplePixel(tex, pos);
  let top = samplePixel(tex, pos + vec2i(0, -1));
  let bottom = samplePixel(tex, pos + vec2i(0, 1));
  let left = samplePixel(tex, pos + vec2i(-1, 0));
  let right = samplePixel(tex, pos + vec2i(1, 0));

  let laplacian = 4.0 * center - top - bottom - left - right;
  return abs(laplacian);
}

// Canny-style edge detection (Sobel + non-maximum suppression)
fn cannyEdge(tex: texture_2d<f32>, pos: vec2i) -> f32 {
  // First, calculate Sobel gradients
  let tl = samplePixel(tex, pos + vec2i(-1, -1));
  let tc = samplePixel(tex, pos + vec2i(0, -1));
  let tr = samplePixel(tex, pos + vec2i(1, -1));
  let ml = samplePixel(tex, pos + vec2i(-1, 0));
  let mr = samplePixel(tex, pos + vec2i(1, 0));
  let bl = samplePixel(tex, pos + vec2i(-1, 1));
  let bc = samplePixel(tex, pos + vec2i(0, 1));
  let br = samplePixel(tex, pos + vec2i(1, 1));

  let gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
  let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;

  let magnitude = sqrt(gx * gx + gy * gy);

  // Simple non-maximum suppression approximation
  // Check perpendicular to gradient direction
  let angle = atan2(gy, gx);
  let absAngle = abs(angle);

  var neighbor1: f32;
  var neighbor2: f32;

  // Quantize angle to 4 directions: 0, 45, 90, 135 degrees
  if (absAngle < 0.3927) { // ~22.5 degrees - horizontal
    neighbor1 = samplePixel(tex, pos + vec2i(-1, 0));
    neighbor2 = samplePixel(tex, pos + vec2i(1, 0));
  } else if (absAngle < 1.178) { // ~67.5 degrees - diagonal
    if (angle > 0.0) {
      neighbor1 = samplePixel(tex, pos + vec2i(-1, -1));
      neighbor2 = samplePixel(tex, pos + vec2i(1, 1));
    } else {
      neighbor1 = samplePixel(tex, pos + vec2i(-1, 1));
      neighbor2 = samplePixel(tex, pos + vec2i(1, -1));
    }
  } else if (absAngle < 1.963) { // ~112.5 degrees - vertical
    neighbor1 = samplePixel(tex, pos + vec2i(0, -1));
    neighbor2 = samplePixel(tex, pos + vec2i(0, 1));
  } else { // ~157.5 degrees - diagonal
    if (angle > 0.0) {
      neighbor1 = samplePixel(tex, pos + vec2i(-1, 1));
      neighbor2 = samplePixel(tex, pos + vec2i(1, -1));
    } else {
      neighbor1 = samplePixel(tex, pos + vec2i(-1, -1));
      neighbor2 = samplePixel(tex, pos + vec2i(1, 1));
    }
  }

  // Non-maximum suppression: suppress if not local maximum
  let center = samplePixel(tex, pos);
  if (center < neighbor1 || center < neighbor2) {
    return 0.0;
  }

  return magnitude;
}

// Apply edge thickness via dilation
fn applyThickness(tex: texture_2d<f32>, pos: vec2i, edgeValue: f32, thickness: f32) -> f32 {
  if (thickness <= 1.0) {
    return edgeValue;
  }

  var maxEdge = edgeValue;
  let radius = i32(thickness);

  for (var dy = -radius; dy <= radius; dy++) {
    for (var dx = -radius; dx <= radius; dx++) {
      if (dx == 0 && dy == 0) {
        continue;
      }

      let neighborPos = pos + vec2i(dx, dy);
      let dims = textureDimensions(tex);
      if (neighborPos.x >= 0 && neighborPos.x < i32(dims.x) &&
          neighborPos.y >= 0 && neighborPos.y < i32(dims.y)) {

        var neighborEdge: f32;
        let algo = i32(params.algorithm);
        if (algo == 0) {
          neighborEdge = sobelEdge(tex, neighborPos);
        } else if (algo == 1) {
          neighborEdge = prewittEdge(tex, neighborPos);
        } else if (algo == 2) {
          neighborEdge = laplacianEdge(tex, neighborPos);
        } else {
          neighborEdge = cannyEdge(tex, neighborPos);
        }

        maxEdge = max(maxEdge, neighborEdge);
      }
    }
  }

  return maxEdge;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outputTex);
  let pos = vec2i(i32(id.x), i32(id.y));

  // Check bounds
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  // Run selected edge detection algorithm
  var edgeMagnitude: f32;
  let algo = i32(params.algorithm);

  if (algo == 0) {
    edgeMagnitude = sobelEdge(inputTex, pos);
  } else if (algo == 1) {
    edgeMagnitude = prewittEdge(inputTex, pos);
  } else if (algo == 2) {
    edgeMagnitude = laplacianEdge(inputTex, pos);
  } else {
    edgeMagnitude = cannyEdge(inputTex, pos);
  }

  // Apply thickness if needed
  if (params.thickness > 1.0) {
    edgeMagnitude = applyThickness(inputTex, pos, edgeMagnitude, params.thickness);
  }

  // Apply threshold
  let isEdge = step(params.threshold, edgeMagnitude);

  // Get edge color
  let edgeColor = vec3f(params.edgeColorR, params.edgeColorG, params.edgeColorB);

  // Determine output color based on overlay mode
  var outputColor: vec3f;

  if (params.showOverlay > 0.5) {
    // Overlay mode: show edges on original image
    let originalColor = textureLoad(inputTex, pos, 0).rgb;
    outputColor = mix(originalColor, edgeColor, isEdge);
  } else {
    // Black background mode
    let backgroundColor = vec3f(0.0);
    outputColor = mix(backgroundColor, edgeColor, isEdge);
  }

  // Apply invert if needed
  if (params.invert > 0.5) {
    outputColor = vec3f(1.0) - outputColor;
  }

  textureStore(outputTex, pos, vec4f(outputColor, 1.0));
}
