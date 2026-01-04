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
    texture2d<float> dayTexture [[texture(0)]],
    texture2d<float> nightTexture [[texture(1)]],
    sampler textureSampler [[sampler(0)]],
    constant Uniforms& uniforms [[buffer(0)]]
) {
    // Sample both day and night textures
    float4 day_color = dayTexture.sample(textureSampler, in.uv);
    float4 night_color = nightTexture.sample(textureSampler, in.uv);

    // Normalize the interpolated normal
    float3 normal = normalize(in.world_normal);

    // Calculate diffuse lighting
    float3 light_dir = normalize(uniforms.light_direction);
    float diffuse = max(dot(normal, light_dir), 0.0);

    // Boost the diffuse component for brighter lit areas
    float boosted_diffuse = diffuse * 1.3;  // 30% brighter on lit side

    // Combine ambient and diffuse for day side
    float light_intensity = uniforms.ambient_strength + (1.0 - uniforms.ambient_strength) * boosted_diffuse;

    // Apply lighting to day texture
    float3 lit_day_color = day_color.rgb * light_intensity;

    // Blend between day and night based on lighting (not UV coordinates)
    // This ensures day/night boundary follows the lighting terminator
    float night_mix = 1.0 - smoothstep(0.0, 0.15, diffuse);

    // City lights should only appear on the dark side
    float city_lights_threshold = smoothstep(0.1, 0.0, diffuse);

    // On night side: show city lights (dimmer than before)
    // On day side: show lit day texture
    float3 night_side = night_color.rgb * 0.8 * city_lights_threshold;
    float3 blended_color = mix(lit_day_color, night_side, night_mix);

    // Atmospheric glow (Fresnel effect) - only on the lit side
    // Calculate view direction (pointing toward camera)
    float3 view_dir = normalize(-in.position.xyz);
    float fresnel = 1.0 - max(dot(normal, view_dir), 0.0);
    fresnel = pow(fresnel, 3.0); // Sharpen the edge effect

    // Blue atmospheric glow - only visible on lit edges
    float lit_edge = smoothstep(0.0, 0.3, diffuse); // Only on lit side
    float3 atmosphere_color = float3(0.3, 0.5, 1.0); // Light blue
    float3 glow = atmosphere_color * fresnel * 0.25 * lit_edge; // Reduced intensity, only on lit side

    // Combine blended color with atmospheric glow
    float3 final_color = blended_color + glow;

    return float4(final_color, 1.0);
}
