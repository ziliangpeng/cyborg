#include <metal_stdlib>
using namespace metal;

struct Star {
    float3 position [[attribute(0)]];
    float brightness [[attribute(1)]];
};

struct StarUniforms {
    float4x4 view_projection;
};

struct VertexOut {
    float4 position [[position]];
    float point_size [[point_size]];
    float brightness;
};

vertex VertexOut vertex_main(
    Star star [[stage_in]],
    constant StarUniforms& uniforms [[buffer(1)]]
) {
    VertexOut out;
    out.position = uniforms.view_projection * float4(star.position, 1.0);
    out.point_size = 2.0;
    out.brightness = star.brightness;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return float4(in.brightness, in.brightness, in.brightness, 1.0);
}
