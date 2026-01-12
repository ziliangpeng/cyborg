// Pixel Sort - Segment Identification Pass
// Identifies which pixels belong to sortable segments based on threshold

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct SegmentParams {
  thresholdLow: f32,
  thresholdHigh: f32,
  width: f32,
  height: f32,
  thresholdMode: f32,  // 0=brightness, 1=saturation, 2=hue, 3=edge
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
}

@group(0) @binding(2) var<uniform> params: SegmentParams;

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
      h = (rgb.g - rgb.b) / delta;
    } else if (cmax == rgb.g) {
      h = (rgb.b - rgb.r) / delta + 2.0;
    } else {
      h = (rgb.r - rgb.g) / delta + 4.0;
    }
    h = h * 60.0;
    if (h < 0.0) {
      h += 360.0;
    }
  }

  let s = select(0.0, delta / cmax, cmax > 0.0);
  let v = cmax;

  return vec3f(h / 360.0, s, v);  // Normalize hue to 0-1
}

fn sobel(coord: vec2i) -> f32 {
  let dims = vec2i(textureDimensions(inputTex));

  // Sample 3x3 neighborhood
  var lum: array<f32, 9>;
  var idx = 0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let sampleCoord = clamp(coord + vec2i(dx, dy), vec2i(0), dims - vec2i(1));
      let color = textureLoad(inputTex, sampleCoord, 0).rgb;
      lum[idx] = luminance(color);
      idx++;
    }
  }

  // Sobel operators
  let gx = -lum[0] + lum[2] - 2.0*lum[3] + 2.0*lum[5] - lum[6] + lum[8];
  let gy = -lum[0] - 2.0*lum[1] - lum[2] + lum[6] + 2.0*lum[7] + lum[8];

  return sqrt(gx*gx + gy*gy);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outputTex);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let coord = vec2i(id.xy);
  let color = textureLoad(inputTex, coord, 0);

  // Compute threshold value based on mode
  var thresholdValue: f32;
  let mode = i32(params.thresholdMode);

  if (mode == 0) {
    // Brightness
    thresholdValue = luminance(color.rgb);
  } else if (mode == 1) {
    // Saturation
    let hsv = rgbToHsv(color.rgb);
    thresholdValue = hsv.y;
  } else if (mode == 2) {
    // Hue
    let hsv = rgbToHsv(color.rgb);
    thresholdValue = hsv.x;
  } else {
    // Edge (mode == 3)
    thresholdValue = 1.0 - sobel(coord);  // Invert so high values = smooth areas
  }

  // Check if pixel is in sortable segment
  let inSegment = thresholdValue >= params.thresholdLow && thresholdValue <= params.thresholdHigh;

  // Store result: output original color, alpha channel indicates segment membership
  // alpha = 1.0 means in segment, alpha = 0.0 means not in segment
  textureStore(outputTex, coord, vec4f(color.rgb, select(0.0, 1.0, inSegment)));
}
