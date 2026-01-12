// Pixel Sort - Main Sorting Shader
// Performs bitonic or bubble sort on horizontal pixel segments

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct SortParams {
  thresholdLow: f32,
  thresholdHigh: f32,
  width: f32,
  height: f32,
  sortKey: f32,        // 0=luminance, 1=hue, 2=saturation, 3=red, 4=green, 5=blue
  sortOrder: f32,      // 0=ascending, 1=descending
  direction: f32,      // 0=horizontal, 1=vertical, 2=diagonal_right, 3=diagonal_left
  algorithm: f32,      // 0=bitonic, 1=bubble
  stage: f32,          // For bitonic sort: current stage
  step: f32,           // For bitonic sort: current step
  iteration: f32,      // For bubble sort: current iteration (0-99)
  _pad: f32,
}

@group(0) @binding(2) var<uniform> params: SortParams;

fn luminance(color: vec3f) -> f32 {
  return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

fn rgbToHsv(rgb: vec3f) -> vec3f {
  let cmax = max(max(rgb.r, rgb.g), rgb.b);
  let cmin = min(min(rgb.r, rgb.g), rgb.b);
  let delta = cmax - cmin;

  var h: f32 = 0.0;
  if (delta > 0.0) {
    if (cmax == rgb.r) {
      h = 60.0 * (((rgb.g - rgb.b) / delta) % 6.0);
    } else if (cmax == rgb.g) {
      h = 60.0 * (((rgb.b - rgb.r) / delta) + 2.0);
    } else {
      h = 60.0 * (((rgb.r - rgb.g) / delta) + 4.0);
    }
  }
  if (h < 0.0) {
    h += 360.0;
  }

  let s = select(0.0, delta / cmax, cmax > 0.0);
  let v = cmax;

  return vec3f(h / 360.0, s, v);
}

fn getSortKey(color: vec4f) -> f32 {
  let mode = i32(params.sortKey);

  if (mode == 0) {
    // Luminance
    return luminance(color.rgb);
  } else if (mode == 1) {
    // Hue
    let hsv = rgbToHsv(color.rgb);
    return hsv.x;
  } else if (mode == 2) {
    // Saturation
    let hsv = rgbToHsv(color.rgb);
    return hsv.y;
  } else if (mode == 3) {
    // Red
    return color.r;
  } else if (mode == 4) {
    // Green
    return color.g;
  } else {
    // Blue (mode == 5)
    return color.b;
  }
}

fn shouldSwap(key1: f32, key2: f32, ascending: bool) -> bool {
  if (ascending) {
    return key1 > key2;
  } else {
    return key1 < key2;
  }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outputTex);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let coord = vec2i(id.xy);
  let color = textureLoad(inputTex, coord, 0);

  // Check if this pixel is in a sortable segment (alpha == 1.0)
  if (color.a < 0.5) {
    // Not in segment, pass through
    textureStore(outputTex, coord, color);
    return;
  }

  // Determine sort direction
  let dir = i32(params.direction);
  let algo = i32(params.algorithm);
  let x = i32(id.x);
  let y = i32(id.y);

  // Determine primary coordinate and bounds based on direction
  // dir: 0=horizontal, 1=vertical, 2=diagonal_right, 3=diagonal_left
  var primaryCoord: i32;
  var maxPrimary: i32;
  var partnerCoord: vec2i;

  if (dir == 0) {
    // Horizontal: sort along x-axis
    primaryCoord = x;
    maxPrimary = i32(dims.x);
  } else if (dir == 1) {
    // Vertical: sort along y-axis
    primaryCoord = y;
    maxPrimary = i32(dims.y);
  } else if (dir == 2) {
    // Diagonal right (top-left to bottom-right)
    primaryCoord = x + y;
    maxPrimary = i32(dims.x + dims.y);
  } else {
    // Diagonal left (top-right to bottom-left)
    primaryCoord = x - y + i32(dims.y);
    maxPrimary = i32(dims.x + dims.y);
  }

  if (algo == 0) {
    // Bitonic sort
    let stage = i32(params.stage);
    let step = i32(params.step);

    let pairDistance = 1 << u32(step);
    let blockSize = 1 << u32(stage + 1);

    let pairIndex = primaryCoord % blockSize;
    let ascending = params.sortOrder < 0.5;

    // Determine if this thread should be ascending or descending based on block
    let blockAscending = ((primaryCoord / blockSize) % 2 == 0) == ascending;

    // Only process if within pair distance
    if (pairIndex < blockSize / 2) {
      // Calculate partner position based on direction
      if (dir == 0) {
        // Horizontal
        let partnerX = x + pairDistance;
        if (partnerX < i32(dims.x)) {
          partnerCoord = vec2i(partnerX, y);
        } else {
          textureStore(outputTex, coord, color);
          return;
        }
      } else if (dir == 1) {
        // Vertical
        let partnerY = y + pairDistance;
        if (partnerY < i32(dims.y)) {
          partnerCoord = vec2i(x, partnerY);
        } else {
          textureStore(outputTex, coord, color);
          return;
        }
      } else if (dir == 2) {
        // Diagonal right
        let partnerX = x + pairDistance;
        let partnerY = y + pairDistance;
        if (partnerX < i32(dims.x) && partnerY < i32(dims.y)) {
          partnerCoord = vec2i(partnerX, partnerY);
        } else {
          textureStore(outputTex, coord, color);
          return;
        }
      } else {
        // Diagonal left
        let partnerX = x + pairDistance;
        let partnerY = y - pairDistance;
        if (partnerX < i32(dims.x) && partnerY >= 0) {
          partnerCoord = vec2i(partnerX, partnerY);
        } else {
          textureStore(outputTex, coord, color);
          return;
        }
      }

      let partnerColor = textureLoad(inputTex, partnerCoord, 0);

      // Only sort if both pixels are in segment
      if (partnerColor.a >= 0.5) {
        let key1 = getSortKey(color);
        let key2 = getSortKey(partnerColor);

        if (shouldSwap(key1, key2, blockAscending)) {
          // Swap: write partner's color to current position
          textureStore(outputTex, coord, partnerColor);
          return;
        }
      }
    }
  } else {
    // Bubble sort (algo == 1)
    let iter = i32(params.iteration);
    let isEvenPass = iter % 2 == 0;

    // Even pass: compare (0,1), (2,3), (4,5)...
    // Odd pass: compare (1,2), (3,4), (5,6)...
    let shouldCompare = (isEvenPass && primaryCoord % 2 == 0) || (!isEvenPass && primaryCoord % 2 == 1);

    if (shouldCompare) {
      // Calculate partner position based on direction
      if (dir == 0) {
        // Horizontal
        let partnerX = x + 1;
        if (partnerX < i32(dims.x)) {
          partnerCoord = vec2i(partnerX, y);
        } else {
          textureStore(outputTex, coord, color);
          return;
        }
      } else if (dir == 1) {
        // Vertical
        let partnerY = y + 1;
        if (partnerY < i32(dims.y)) {
          partnerCoord = vec2i(x, partnerY);
        } else {
          textureStore(outputTex, coord, color);
          return;
        }
      } else if (dir == 2) {
        // Diagonal right
        let partnerX = x + 1;
        let partnerY = y + 1;
        if (partnerX < i32(dims.x) && partnerY < i32(dims.y)) {
          partnerCoord = vec2i(partnerX, partnerY);
        } else {
          textureStore(outputTex, coord, color);
          return;
        }
      } else {
        // Diagonal left
        let partnerX = x + 1;
        let partnerY = y - 1;
        if (partnerX < i32(dims.x) && partnerY >= 0) {
          partnerCoord = vec2i(partnerX, partnerY);
        } else {
          textureStore(outputTex, coord, color);
          return;
        }
      }

      let partnerColor = textureLoad(inputTex, partnerCoord, 0);

      // Only sort if both pixels are in segment
      if (partnerColor.a >= 0.5) {
        let key1 = getSortKey(color);
        let key2 = getSortKey(partnerColor);
        let ascending = params.sortOrder < 0.5;

        if (shouldSwap(key1, key2, ascending)) {
          // Swap: write partner's color to current position
          textureStore(outputTex, coord, partnerColor);
          return;
        }
      }
    }
  }

  // Default: pass through
  textureStore(outputTex, coord, color);
}
