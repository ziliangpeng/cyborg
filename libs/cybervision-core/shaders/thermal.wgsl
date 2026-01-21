// Thermal/Heat Map compute shader
// Maps luminance to thermal color gradients

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct ThermalParams {
  palette: f32,    // 0=Classic, 1=Infrared, 2=Fire
  contrast: f32,   // 0.5-2.0
  invert: f32,     // 0=normal, 1=inverted
  _pad0: f32,
  width: f32,
  height: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(2) var<uniform> params: ThermalParams;

// Convert RGB to grayscale luminance
fn luminance(color: vec3f) -> f32 {
  return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

// Apply contrast adjustment
fn applyContrast(lum: f32, contrast: f32) -> f32 {
  let centered = lum - 0.5;
  let adjusted = centered * contrast + 0.5;
  return clamp(adjusted, 0.0, 1.0);
}

// Classic Thermal: Black -> Blue -> Cyan -> Green -> Yellow -> Orange -> Red -> White
fn classicThermal(t: f32) -> vec3f {
  if (t < 0.143) {
    return mix(vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), t / 0.143);
  } else if (t < 0.286) {
    return mix(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 1.0), (t - 0.143) / 0.143);
  } else if (t < 0.429) {
    return mix(vec3f(0.0, 1.0, 1.0), vec3f(0.0, 1.0, 0.0), (t - 0.286) / 0.143);
  } else if (t < 0.571) {
    return mix(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 1.0, 0.0), (t - 0.429) / 0.143);
  } else if (t < 0.714) {
    return mix(vec3f(1.0, 1.0, 0.0), vec3f(1.0, 0.5, 0.0), (t - 0.571) / 0.143);
  } else if (t < 0.857) {
    return mix(vec3f(1.0, 0.5, 0.0), vec3f(1.0, 0.0, 0.0), (t - 0.714) / 0.143);
  } else {
    return mix(vec3f(1.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0), (t - 0.857) / 0.143);
  }
}

// Infrared: Black -> Purple -> Blue -> Magenta -> Red -> Orange -> Yellow -> White
fn infraredPalette(t: f32) -> vec3f {
  if (t < 0.143) {
    return mix(vec3f(0.0, 0.0, 0.0), vec3f(0.3, 0.0, 0.5), t / 0.143);
  } else if (t < 0.286) {
    return mix(vec3f(0.3, 0.0, 0.5), vec3f(0.0, 0.0, 1.0), (t - 0.143) / 0.143);
  } else if (t < 0.429) {
    return mix(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 1.0), (t - 0.286) / 0.143);
  } else if (t < 0.571) {
    return mix(vec3f(1.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), (t - 0.429) / 0.143);
  } else if (t < 0.714) {
    return mix(vec3f(1.0, 0.0, 0.0), vec3f(1.0, 0.5, 0.0), (t - 0.571) / 0.143);
  } else if (t < 0.857) {
    return mix(vec3f(1.0, 0.5, 0.0), vec3f(1.0, 1.0, 0.0), (t - 0.714) / 0.143);
  } else {
    return mix(vec3f(1.0, 1.0, 0.0), vec3f(1.0, 1.0, 1.0), (t - 0.857) / 0.143);
  }
}

// Fire: Black -> Dark Red -> Red -> Orange -> Yellow -> White
fn firePalette(t: f32) -> vec3f {
  if (t < 0.2) {
    return mix(vec3f(0.0, 0.0, 0.0), vec3f(0.5, 0.0, 0.0), t / 0.2);
  } else if (t < 0.4) {
    return mix(vec3f(0.5, 0.0, 0.0), vec3f(1.0, 0.0, 0.0), (t - 0.2) / 0.2);
  } else if (t < 0.6) {
    return mix(vec3f(1.0, 0.0, 0.0), vec3f(1.0, 0.5, 0.0), (t - 0.4) / 0.2);
  } else if (t < 0.8) {
    return mix(vec3f(1.0, 0.5, 0.0), vec3f(1.0, 1.0, 0.0), (t - 0.6) / 0.2);
  } else {
    return mix(vec3f(1.0, 1.0, 0.0), vec3f(1.0, 1.0, 1.0), (t - 0.8) / 0.2);
  }
}

// Map luminance to thermal color based on palette
fn thermalColor(lum: f32, palette: i32) -> vec3f {
  if (palette == 0) {
    return classicThermal(lum);
  } else if (palette == 1) {
    return infraredPalette(lum);
  } else {
    return firePalette(lum);
  }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outputTex);
  let pos = vec2i(i32(id.x), i32(id.y));

  // Check bounds
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  // Sample input color
  let color = textureLoad(inputTex, pos, 0).rgb;

  // Calculate luminance
  var lum = luminance(color);

  // Apply contrast adjustment
  lum = applyContrast(lum, params.contrast);

  // Apply inversion if enabled
  if (params.invert > 0.5) {
    lum = 1.0 - lum;
  }

  // Map to thermal color
  let thermalCol = thermalColor(lum, i32(params.palette));

  textureStore(outputTex, pos, vec4f(thermalCol, 1.0));
}
