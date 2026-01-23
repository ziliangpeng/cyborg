// ASCII effect compute shader

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct AsciiParams {
  cellSize: f32,
  colorize: f32,
  useGlyphs: f32,
  _pad0: f32,
}

@group(0) @binding(2) var<uniform> params: AsciiParams;

const ASCII_SIZE: i32 = 5;
const DOT_LEVELS: i32 = 8;
const GLYPH_LEVELS: i32 = 10;
const DOT_MASKS: array<u32, 8> = array<u32, 8>(
  0u,
  4096u,
  131200u,
  31744u,
  1016800u,
  4357252u,
  11512810u,
  33554431u
);
const GLYPH_MASKS: array<u32, 10> = array<u32, 10>(
  0u,
  4194304u,
  131200u,
  31744u,
  4357252u,
  1016800u,
  10648714u,
  27070835u,
  23058421u,
  33488831u
);

fn luminance(color: vec3f) -> f32 {
  return dot(color, vec3f(0.299, 0.587, 0.114));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outputTex);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let pos = vec2i(i32(id.x), i32(id.y));
  let cellSize = max(1, i32(params.cellSize));
  let cellX = (pos.x / cellSize) * cellSize;
  let cellY = (pos.y / cellSize) * cellSize;
  let centerX = clamp(cellX + cellSize / 2, 0, i32(dims.x) - 1);
  let centerY = clamp(cellY + cellSize / 2, 0, i32(dims.y) - 1);

  let sampleColor = textureLoad(inputTex, vec2i(centerX, centerY), 0).rgb;
  let lum = luminance(sampleColor);
  let dotLevel = i32(clamp(floor(lum * f32(DOT_LEVELS - 1) + 0.5), 0.0, f32(DOT_LEVELS - 1)));
  let glyphLevel = i32(clamp(floor(lum * f32(GLYPH_LEVELS - 1) + 0.5), 0.0, f32(GLYPH_LEVELS - 1)));

  let localX = f32(pos.x - cellX) / f32(cellSize);
  let localY = f32(pos.y - cellY) / f32(cellSize);
  let gx = i32(clamp(floor(localX * f32(ASCII_SIZE)), 0.0, f32(ASCII_SIZE - 1)));
  let gy = i32(clamp(floor(localY * f32(ASCII_SIZE)), 0.0, f32(ASCII_SIZE - 1)));
  let idx = gx + gy * ASCII_SIZE;

  let useGlyphs = params.useGlyphs > 0.5;
  let mask = select(DOT_MASKS[dotLevel], GLYPH_MASKS[glyphLevel], useGlyphs);
  let bit = (mask >> u32(idx)) & 1u;

  let glyphColor = select(vec3f(1.0), sampleColor, params.colorize > 0.5);
  let outColor = glyphColor * f32(bit);

  textureStore(outputTex, pos, vec4f(outColor, 1.0));
}
