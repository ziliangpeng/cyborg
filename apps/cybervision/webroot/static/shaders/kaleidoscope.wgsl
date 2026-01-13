struct Uniforms {
    segments: f32,
    rotationSpeed: f32,
    time: f32,
    padding: f32,
    videoWidth: f32,
    videoHeight: f32,
    padding2: vec2<f32>,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var dstTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(dstTex);
    let x = global_id.x;
    let y = global_id.y;

    if (x >= dims.x || y >= dims.y) {
        return;
    }

    let uv = vec2<f32>(f32(x), f32(y)) / vec2<f32>(f32(dims.x), f32(dims.y));

    let center = vec2<f32>(0.5, 0.5);
    let delta = uv - center;

    let radius = length(delta);
    var angle = atan2(delta.y, delta.x);

    let segmentAngle = 6.28318530718 / uniforms.segments;

    let twiceSegment = segmentAngle * 2.0;
    // Use % operator instead of mod() function
    angle = abs((angle % twiceSegment) - segmentAngle);

    if (uniforms.rotationSpeed > 0.0) {
        angle = angle + uniforms.time * uniforms.rotationSpeed * 0.5;
    }

    let newDelta = vec2<f32>(cos(angle), sin(angle)) * radius;
    var newUV = center + newDelta;

    var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    if (newUV.x >= 0.0 && newUV.x <= 1.0 && newUV.y >= 0.0 && newUV.y <= 1.0) {
        // Use textureLoad for compute shader instead of textureSampleLevel
        let texCoord = vec2<i32>(i32(newUV.x * f32(dims.x)), i32(newUV.y * f32(dims.y)));
        color = textureLoad(srcTex, texCoord, 0);
    }

    textureStore(dstTex, vec2<i32>(i32(x), i32(y)), color);
}
