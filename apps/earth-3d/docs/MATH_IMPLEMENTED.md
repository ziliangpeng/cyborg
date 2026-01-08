# Mathematical Concepts Implemented in Earth-3D

This document explains all mathematical formulas that we explicitly wrote in code for the Earth-3D visualization application.

## 1. Spherical Coordinates and UV Sphere Generation

**Location**: `src/geometry/sphere.rs`

### Spherical to Cartesian Conversion

We use spherical coordinates to generate points on a sphere surface, then convert them to Cartesian (x, y, z) coordinates.

**Formula**:
```
φ (phi)   = π × (ring / rings)           // Latitude angle [0, π]
θ (theta) = 2π × (segment / segments)    // Longitude angle [0, 2π]

x = r × sin(φ) × cos(θ)
y = r × cos(φ)
z = r × sin(φ) × sin(θ)
```

**Code**:
```rust
for ring in 0..=rings {
    let phi = std::f32::consts::PI * (ring as f32) / (rings as f32);
    let sin_phi = phi.sin();
    let cos_phi = phi.cos();

    for segment in 0..=segments {
        let theta = 2.0 * std::f32::consts::PI * (segment as f32) / (segments as f32);
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        let x = sin_phi * cos_theta;
        let y = cos_phi;
        let z = sin_phi * sin_theta;

        let position = Vec3::new(x * radius, y * radius, z * radius);
    }
}
```

**Explanation**:
- **φ (phi)** represents the polar angle from north pole (0) to south pole (π)
- **θ (theta)** represents the azimuthal angle around the equator (0 to 2π)
- This parametric representation naturally describes all points on a sphere
- Each (φ, θ) pair uniquely maps to exactly one point on the sphere surface

**Why it works**: Spherical coordinates are the natural way to describe points on a sphere, just like longitude and latitude on Earth.

---

### UV Texture Mapping

**Formula**:
```
u = segment / segments     // Horizontal [0, 1]
v = ring / rings           // Vertical [0, 1]
```

**Code**:
```rust
let u = (segment as f32) / (segments as f32);
let v = (ring as f32) / (rings as f32);
let uv = Vec2::new(u, v);
```

**Explanation**:
- Maps sphere surface to a rectangular texture (equirectangular projection)
- Latitude maps to V coordinate (vertical)
- Longitude maps to U coordinate (horizontal)
- This allows a 2D Earth image to wrap seamlessly around the 3D sphere

---

## 2. Lighting Calculations

**Location**: `shaders/lit_textured.metal`

### Lambertian (Diffuse) Reflection

The Lambertian reflection model describes how matte (non-shiny) surfaces reflect light.

**Formula**:
```
diffuse = max(N · L, 0)
```

Where:
- **N** = surface normal (perpendicular to surface)
- **L** = light direction (from surface toward light)
- **·** = dot product

**Code**:
```metal
float3 light_dir = normalize(uniforms.light_direction);
float diffuse = max(dot(normal, light_dir), 0.0);
```

**Physical meaning**:
- When `N · L = 1`: Light hits surface perpendicularly → maximum brightness
- When `N · L = 0`: Light hits surface at grazing angle → no direct light
- When `N · L < 0`: Light comes from behind → clamped to 0 (no negative light)

**Why `max(..., 0.0)`**: Prevents back-facing surfaces from being "negatively lit"

---

### Ambient + Diffuse Combination

**Formula**:
```
light_intensity = ambient_strength + (1 - ambient_strength) × diffuse × boost
```

**Code**:
```metal
float boosted_diffuse = diffuse * 1.3;  // 30% brighter on lit side
float light_intensity = uniforms.ambient_strength +
                       (1.0 - uniforms.ambient_strength) * boosted_diffuse;
float3 lit_color = texture_color × light_intensity;
```

**Explanation**:
- **Ambient lighting**: Constant base illumination (simulates indirect/scattered light)
- **Diffuse component**: Direction-dependent direct illumination
- **Boost factor (1.3)**: Artistic choice to make the day side brighter
- Blends ambient (0.15) with enhanced diffuse lighting

---

## 3. Fresnel Effect (Atmospheric Glow)

**Location**: `shaders/lit_textured.metal`

### View Direction Calculation (World Space)

To compute correct lighting, we calculate the view direction in world space coordinates.

**Formula**:
```
view_dir = normalize(camera_position - world_position)
```

**Code**:
```metal
// In vertex shader - calculate world position:
out.world_position = (uniforms.model_matrix * float4(in.position, 1.0)).xyz;

// In fragment shader - calculate view direction:
float3 view_dir = normalize(uniforms.camera_position - in.world_position);
```

**Explanation**:
- `world_position`: Vertex position transformed to world space
- `camera_position`: Camera's world space position
- `view_dir`: Direction from surface point toward camera
- Must be in same coordinate space for correct lighting calculations

---

### Fresnel Approximation

The Fresnel effect makes edges appear to glow when viewed at grazing angles (like atmospheric scattering).

**Formula**:
```
fresnel = (1 - N · V)^power
```

Where:
- **N** = surface normal
- **V** = view direction (toward camera)
- **power** = edge sharpness parameter (we use 3.0)

**Code**:
```metal
float3 view_dir = normalize(uniforms.camera_position - in.world_position);
float fresnel = 1.0 - max(dot(normal, view_dir), 0.0);
fresnel = pow(fresnel, ATMOSPHERE_GLOW_POWER);  // power = 3.0

// Apply to atmospheric glow:
float3 glow = ATMOSPHERE_COLOR * fresnel * ATMOSPHERE_GLOW_INTENSITY * lit_edge;
```

**Physical meaning**:
- When looking straight at surface: `N · V ≈ 1`, so `fresnel ≈ 0` (no glow)
- When looking at edge: `N · V ≈ 0`, so `fresnel ≈ 1` (strong glow)
- Power function sharpens the falloff for more dramatic edge lighting

**Why it works**: Simulates how Earth's atmosphere scatters light more at the edges (limb darkening effect)

---

## 4. Color Blending

**Location**: `shaders/lit_textured.metal`

### Day/Night Texture Blending

We smoothly transition between day and night textures based on lighting.

**Formula**:
```
night_mix = 1 - smoothstep(start, end, diffuse)
```

**Code**:
```metal
const float NIGHT_MIX_START = 0.0;
const float NIGHT_MIX_END = 0.15;

float night_mix = 1.0 - smoothstep(NIGHT_MIX_START, NIGHT_MIX_END, diffuse);
float3 blended_color = mix(lit_day_color, night_side, night_mix);
```

**Explanation**:
- `smoothstep` creates an S-curve transition between start and end values
- When `diffuse < 0.0`: `night_mix = 1.0` (fully night)
- When `diffuse > 0.15`: `night_mix = 0.0` (fully day)
- Between 0.0 and 0.15: smooth transition
- This ensures the day/night boundary follows the lighting terminator

---

### City Lights Appearance Logic

City lights should only appear on the dark side of Earth.

**Formula**:
```
city_lights_threshold = smoothstep(start, end, diffuse)
```

**Code**:
```metal
const float CITY_LIGHTS_FADE_START = 0.1;
const float CITY_LIGHTS_FADE_END = 0.0;

float city_lights_threshold = smoothstep(CITY_LIGHTS_FADE_START,
                                         CITY_LIGHTS_FADE_END, diffuse);
float3 night_side = night_texture.rgb * CITY_LIGHTS_BRIGHTNESS * city_lights_threshold;
```

**Explanation**:
- When `diffuse > 0.1`: `threshold = 0.0` (no city lights on day side)
- When `diffuse < 0.0`: `threshold = 1.0` (full city lights on night side)
- Smooth fade prevents harsh boundary
- Prevents the visual bug of city lights appearing on the bright side

---

### Linear Interpolation (mix)

Blends between two colors based on a factor.

**Formula**:
```
result = a × (1 - t) + b × t
```

**Code**:
```metal
float3 blended_color = mix(lit_day_color, night_side, night_mix);
```

**Explanation**:
- When `t = 0`: Returns `a` (lit day color)
- When `t = 1`: Returns `b` (night side)
- When `t = 0.5`: Returns 50/50 blend
- Used for smooth transitions between day and night textures

---

## 5. Random Star Distribution

**Location**: `src/stars.rs`

### Spherical Coordinate Randomization

Generates random points on a sphere surface for star positions.

**Formula**:
```
θ = random() × 2π     // Random azimuthal angle
φ = random() × π      // Random polar angle

x = r × sin(φ) × cos(θ)
y = r × cos(φ)
z = r × sin(φ) × sin(θ)
```

**Code**:
```rust
for _ in 0..count {
    let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
    let phi = rng.gen::<f32>() * std::f32::consts::PI;

    let x = radius * phi.sin() * theta.cos();
    let y = radius * phi.cos();
    let z = radius * phi.sin() * theta.sin();

    let brightness = rng.gen::<f32>() * 0.5 + 0.5; // [0.5, 1.0]
}
```

**Note**: This approach produces a **non-uniform distribution** - stars bunch more at the poles. For truly uniform distribution, a more sophisticated method (like Marsaglia's method) would be needed, but the visual difference is minimal for our use case.

---

## 6. Triangle Mesh Indexing

**Location**: `src/geometry/sphere.rs`

### Quad to Triangle Tessellation

Converts each quad (4 vertices) in the UV sphere grid into two triangles (6 indices).

**Formula**:
```
current = ring × (segments + 1) + segment
next = current + (segments + 1)

Triangle 1: [current, next, current + 1]
Triangle 2: [current + 1, next, next + 1]
```

**Code**:
```rust
for ring in 0..rings {
    for segment in 0..segments {
        let current = ring * (segments + 1) + segment;
        let next = current + segments + 1;

        // First triangle
        indices.push(current);
        indices.push(next);
        indices.push(current + 1);

        // Second triangle
        indices.push(current + 1);
        indices.push(next);
        indices.push(next + 1);
    }
}
```

**Explanation**:
- Vertices are stored in row-major order (by rings, then segments)
- Each quad is split into two triangles with counter-clockwise winding
- `segments + 1` accounts for the wraparound vertex in each ring
- This creates a continuous mesh without gaps or overlaps

**Visual**:
```
current ------- current+1
   |        ╱      |
   |     ╱         |
   |  ╱            |
next -------- next+1
```

---

## Summary

We implemented **~10 distinct mathematical formulas** explicitly in code:

1. Spherical to Cartesian coordinate conversion (3D parametric surface)
2. UV texture mapping (2D to 3D projection)
3. Lambertian diffuse reflection (dot product-based lighting)
4. Ambient + diffuse combination (lighting model)
5. World space view direction calculation (coordinate transformation)
6. Fresnel approximation (view-dependent edge effect)
7. Day/night blending (smoothstep-based mixing)
8. City lights logic (threshold-based visibility)
9. Random star distribution (spherical randomization)
10. Triangle mesh indexing (quad tessellation)

These formulas represent the high-level graphics algorithms and visual effects that define the appearance and behavior of the Earth-3D visualization.
