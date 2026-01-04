# Earth 3D Visualization

A realistic 3D Earth visualization built with Rust and Metal on macOS, featuring day/night cycles, city lights, atmospheric glow, and a rotating starfield background.

![Earth 3D](../../docs/earth-3d-screenshot.png)

## Features

### Core Rendering
- **High-resolution sphere mesh**: UV sphere with 2,145 vertices and 12,288 triangles
- **NASA Earth textures**:
  - Blue Marble (5400x2700) for daytime
  - Black Marble for nighttime city lights
- **Phong lighting model**: Realistic directional sunlight with ambient and diffuse components
- **Day/night blending**: Smooth transition following the lighting terminator
- **Atmospheric glow**: Fresnel effect creating blue atmospheric edges on the lit side
- **Starfield background**: 2,000 stars with varying brightness

### Camera Controls
- **Arrow keys**: Move camera around the Earth
  - Up/Down: Move forward/backward
  - Left/Right: Strafe left/right
- **Mouse drag**: Rotate camera view
- **Zoom**: +/- keys to zoom in and out

### Animation Controls
- **Spacebar**: Pause/resume Earth rotation
- **[ ]**: Decrease/increase rotation speed

## Technical Stack

- **Language**: Rust 2021
- **Graphics API**: Metal (via metal-rs)
- **Windowing**: winit 0.30
- **Math**: glam 0.29 (SIMD-optimized)
- **Textures**: image crate (JPEG only)
- **Build Systems**:
  - Cargo for development
  - Bazel for production builds

## Building and Running

### With Cargo (Quick Development)

```bash
cd apps/earth-3d
cargo run --release
```

### With Bazel (Production)

```bash
# Build only
bazel build //apps/earth-3d:earth-3d

# Build and run
bazel run //apps/earth-3d:earth-3d
```

## Project Structure

```
apps/earth-3d/
├── src/
│   ├── main.rs           # Entry point and event loop
│   ├── renderer.rs       # Metal rendering setup and frame rendering
│   ├── camera.rs         # Camera matrices and movement
│   ├── texture.rs        # Texture loading with Bazel/Cargo path resolution
│   ├── stars.rs          # Star field generation
│   └── geometry/
│       └── sphere.rs     # UV sphere mesh generation
├── shaders/
│   ├── lit_textured.metal  # Main Earth shader with lighting
│   └── stars.metal         # Star rendering shader
├── assets/
│   ├── earth_texture.jpg   # NASA Blue Marble (daytime)
│   └── earth_night.jpg     # NASA Black Marble (city lights)
├── BUILD.bazel          # Bazel build configuration
├── Cargo.toml           # Rust dependencies
└── build.rs             # Build script for Metal shader compilation
```

## Shaders

### Earth Shader (`lit_textured.metal`)
- Vertex shader: Transforms vertices and normals to world space
- Fragment shader:
  - Samples both day and night textures
  - Calculates diffuse lighting from directional sun
  - Blends textures based on lighting (not UV coordinates)
  - Applies atmospheric glow using Fresnel effect (lit side only)
  - Dims city lights appropriately for night side

### Star Shader (`stars.metal`)
- Renders point sprites for stars
- Varies brightness based on per-star attributes
- Renders behind Earth with depth testing

## Implementation Notes

### Bazel Integration
The project uses standard Bazel + Rust integration with `rules_rust 0.68.1`:
- Dependencies managed via `crate_universe` from Cargo.toml
- Smart path resolution in `texture.rs` works with both Bazel runfiles and Cargo
- No wrapper scripts needed - pure `rust_binary` target

### Texture Path Resolution
Assets are loaded using a multi-path strategy:
1. Relative path (for `cargo run`)
2. `RUNFILES_DIR` environment variable (Bazel)
3. Binary-relative runfiles directory (alternative Bazel location)

This allows seamless operation with both build systems.

### Lighting Model
- **Light direction**: Points FROM surface TOWARD light source
- **Diffuse calculation**: `max(dot(normal, light_dir), 0.0)`
- **Day/night blending**: Based purely on diffuse lighting value
- **Terminator smoothing**: `smoothstep(0.0, 0.15, diffuse)` for gradual transition

## Performance

- **60 FPS** on M4 Pro MacBook
- **GPU**: Apple M4 Pro with Metal 3
- **Resolution**: Dynamic (matches window size)
- **Optimizations**:
  - Release build with `-C opt-level=3`
  - SIMD math operations via glam
  - Efficient sphere mesh with shared vertices

## Future Enhancements

### High Priority
- **Moon**: Add orbiting moon with correct phase rendering
- **Cloud layer**: Animated cloud texture overlay with transparency
- **Better camera controls**: Smooth interpolation and arc-ball rotation
- **Time of day**: Adjust sun position based on real time
- **Atmosphere scattering**: Rayleigh and Mie scattering for realistic sky colors

### Medium Priority
- **Higher resolution textures**: 8K Blue Marble and Black Marble
- **Specular highlights**: Ocean reflections using specular lighting
- **Normal mapping**: Add surface detail without more geometry
- **Bloom effect**: Glow around city lights for better visibility
- **HDR rendering**: Improve color range and brightness

### Lower Priority
- **Multiple planets**: Support for other solar system bodies
- **Skybox**: Replace stars with actual space imagery
- **Country borders**: Optional overlay showing political boundaries
- **Day/night cycle animation**: Automatic sun rotation over time
- **Screenshot/video export**: Save rendered frames

### Polish Ideas
- **Smoother atmospheric transition**: Better gradient at the terminator
- **City lights intensity variation**: Different brightness for different regions
- **Seasonal tilt**: Adjust Earth's axial tilt (23.5°)
- **Real-time shadows**: Self-shadowing for terrain features
- **Performance metrics**: Display FPS, frame time, and memory usage
- **UI overlay**: Info panel showing camera position, sun angle, etc.
- **Multiple light sources**: Add moonlight or artificial lighting

## Credits

- **Textures**: NASA Visible Earth (public domain)
  - Blue Marble: https://visibleearth.nasa.gov/
  - Black Marble: https://earthobservatory.nasa.gov/
- **Metal Framework**: Apple Inc.
- **Rust crates**: See Cargo.toml for full list

## License

[Your License Here]

## Author

Ziliang Peng (ziliangpeng)

---

## Project Story

This project is a modern recreation of a 3D Earth visualization I originally built using OpenGL on Windows back in 2006, nearly 20 years ago. The original version featured basic texture mapping and simple lighting, built with C++ and OpenGL.

Fast forward to 2026, and this new version leverages modern graphics APIs and languages:

**What Changed:**
- OpenGL → Metal (Apple's modern GPU API)
- C++ → Rust (memory safety and modern tooling)
- Windows → macOS (Apple Silicon with unified memory architecture)
- Manual texture loading → NASA's high-resolution satellite imagery
- Basic lighting → Phong model with day/night cycles and atmospheric effects

**Development Process:**
The entire project was built collaboratively with Claude Code over the course of a single session, following a structured approach:

1. **Phase 0-1**: Project setup and Metal initialization
2. **Phase 2**: Sphere mesh generation and basic rendering
3. **Phase 3**: Texture mapping with NASA imagery
4. **Phase 4**: Phong lighting and day/night effects
5. **Phase 5**: Camera controls and interactivity
6. **Phase 6**: Polish - stars, atmospheric glow, and city lights
7. **Bazel Integration**: Added production build system with standard rules_rust

**Key Technical Challenges Solved:**
- Setting up Metal rendering pipeline from scratch in Rust
- Implementing proper day/night texture blending that follows lighting
- Ensuring city lights only appear on the dark side
- Fixing atmospheric glow to avoid making the night side too bright
- Creating a Bazel build that works seamlessly with Cargo
- Handling asset paths for both development (Cargo) and production (Bazel)

**What Makes This Version Special:**
- Uses NASA's actual satellite imagery (Blue Marble for day, Black Marble for night)
- City lights texture shows real human civilization patterns
- Atmospheric glow creates a realistic space-view appearance
- Smooth day/night terminator that follows the sun's angle
- Interactive controls for exploration
- Modern Rust safety guarantees with zero-cost abstractions
- Production-ready Bazel build system

The project demonstrates how far graphics programming has evolved in 20 years, both in terms of hardware capabilities (60 FPS at high resolution on a laptop GPU) and software tooling (Rust's safety, Bazel's reproducibility, Claude Code's AI assistance).

---

Built with [Claude Code](https://claude.com/claude-code)
