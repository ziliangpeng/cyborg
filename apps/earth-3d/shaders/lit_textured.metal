#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 uv [[attribute(2)]];
};

struct Uniforms {
    float4x4 mvp_matrix;
    float4x4 model_matrix;
    float3 light_direction;
    float ambient_strength;
};

struct VertexOut {
    float4 position [[position]];
    float3 world_normal;
    float2 uv;
};

vertex VertexOut vertex_main(
    Vertex in [[stage_in]],
    constant Uniforms& uniforms [[buffer(1)]]
) {
    VertexOut out;
    out.position = uniforms.mvp_matrix * float4(in.position, 1.0);

    // Transform normal to world space
    out.world_normal = (uniforms.model_matrix * float4(in.normal, 0.0)).xyz;
    out.world_normal = normalize(out.world_normal);

    out.uv = in.uv;
    return out;
}

fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    texture2d<float> colorTexture [[texture(0)]],
    sampler textureSampler [[sampler(0)]],
    constant Uniforms& uniforms [[buffer(0)]]
) {
    // Sample the Earth texture
    float4 texture_color = colorTexture.sample(textureSampler, in.uv);

    // Normalize the interpolated normal
    float3 normal = normalize(in.world_normal);

    // Calculate diffuse lighting
    float3 light_dir = normalize(uniforms.light_direction);
    float diffuse = max(dot(normal, light_dir), 0.0);

    // Boost the diffuse component for brighter lit areas
    float boosted_diffuse = diffuse * 1.3;  // 30% brighter on lit side

    // Combine ambient and diffuse
    float light_intensity = uniforms.ambient_strength + (1.0 - uniforms.ambient_strength) * boosted_diffuse;

    // Apply lighting to texture color
    float3 lit_color = texture_color.rgb * light_intensity;

    return float4(lit_color, texture_color.a);
}
