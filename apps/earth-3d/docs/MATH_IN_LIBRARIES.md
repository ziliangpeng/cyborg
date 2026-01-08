# Mathematical Formulas Hidden in Libraries

This document details the mathematical operations abstracted by the `glam` library and Metal shader built-ins used in the Earth-3D application. While we don't implement these formulas ourselves, understanding what happens "under the hood" helps explain how the application works.

## Overview

For every line of high-level code we write, there are often hundreds of lines of mathematical primitives running underneath. This document reveals those hidden operations.

---

## 1. Matrix Mathematics (glam)

### Rotation Matrix (Y-axis)

**API**: `Mat4::from_rotation_y(θ)`

**What we write**:
```rust
let model_matrix = Mat4::from_rotation_y(self.rotation_angle);
```

**Hidden formula**:
```
R_y(θ) = ┌ cos(θ)   0   sin(θ)   0 ┐
         │   0      1     0      0 │
         │-sin(θ)   0   cos(θ)   0 │
         └   0      0     0      1 ┘
```

**Explanation**:
- Rotates points around the Y-axis (vertical axis) by angle θ
- X and Z coordinates are rotated, Y stays unchanged
- This creates the spinning Earth effect
- Preserves distances and angles (rigid transformation)

**Hidden complexity**: Computes two trigonometric functions (sin, cos) and constructs a 4×4 matrix

---

### Perspective Projection Matrix

**API**: `Mat4::perspective_rh(fov, aspect, near, far)`

**What we write**:
```rust
let projection_matrix = Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far);
```

**Hidden formula**:
```
f = 1 / tan(fov / 2)

P = ┌ f/aspect    0         0                           0                    ┐
    │    0        f         0                           0                    │
    │    0        0    -(far+near)/(far-near)   -2·far·near/(far-near)      │
    └    0        0        -1                           0                    ┘
```

Where:
- `fov`: Field of view angle (35° in our app)
- `aspect`: Width / height ratio of the screen
- `near`: Near clipping plane (0.1)
- `far`: Far clipping plane (100.0)
- `f`: Focal length factor

**Explanation**:
- Maps 3D view frustum to normalized device coordinates (NDC)
- Creates the perspective effect (distant objects appear smaller)
- Objects closer than `near` or farther than `far` are clipped
- Right-handed coordinate system (`_rh`)

**Hidden complexity**:
- Tangent calculation
- Multiple divisions and arithmetic operations
- Constructs a 4×4 matrix with carefully chosen values

**What it does**: This is why the Earth looks like a realistic 3D object instead of flat!

---

### Look-At Matrix (View Matrix)

**API**: `Mat4::look_at_rh(eye, target, up)`

**What we write**:
```rust
let view_matrix = Mat4::look_at_rh(self.position, self.target, self.up);
```

**Hidden calculations**:
```
forward = normalize(target - eye)
right = normalize(forward × up)
up' = right × forward

V = ┌ right.x    right.y    right.z    -right·eye   ┐
    │ up'.x      up'.y      up'.z      -up'·eye     │
    │-forward.x -forward.y -forward.z  forward·eye  │
    └   0          0          0            1         ┘
```

**Explanation**:
- Positions the virtual camera at `eye`, looking at `target`
- `up` vector defines camera orientation (which way is "up")
- Constructs orthonormal basis vectors (right, up', forward)
- Transforms world coordinates into camera space

**Hidden complexity**:
- Two vector subtractions
- Two cross products (6 multiplications + 6 subtractions each)
- Two vector normalizations (2 square roots + 6 divisions)
- Three dot products (9 multiplications + 6 additions)
- Constructs a 4×4 transformation matrix

**Total operations**: Approximately 50+ floating-point operations hidden in this one call!

---

### Matrix Multiplication

**API**: `projection_matrix * view_matrix * model_matrix`

**What we write**:
```rust
let mvp_matrix = projection_matrix * view_matrix * model_matrix;
```

**Hidden formula** for `C = A × B`:
```
C[i,j] = Σ(k=0→3) A[i,k] × B[k,j]

For each element: 4 multiplications + 3 additions
For 4×4 matrix: 16 elements
Total: 64 multiplications + 48 additions = 112 operations
```

**Explanation**:
- Combines multiple transformations into one matrix
- Order matters: `A × B ≠ B × A` for matrices
- Our pipeline: `model → world → camera → clip`
- GPU multiplies this matrix by every vertex position

**Hidden complexity**:
- We do two 4×4 matrix multiplications
- Total: 224 floating-point operations
- This happens once per frame

**Why it's fast**: Modern CPUs use SIMD instructions to parallelize these operations

---

## 2. Vector Operations (glam)

### Normalization

**API**: `vec.normalize()`

**What we write**:
```rust
let normal = Vec3::new(x, y, z).normalize();
```

**Hidden formula**:
```
||v|| = √(x² + y² + z²)           // Magnitude
v̂ = v / ||v||                     // Normalized vector
v̂ = (x/||v||, y/||v||, z/||v||)
```

**Explanation**:
- Creates a unit vector (length = 1) pointing in the same direction
- Essential for lighting calculations (normals must be normalized)
- Preserves direction but not magnitude

**Hidden complexity**:
- 3 multiplications (x², y², z²)
- 2 additions
- 1 square root (expensive!)
- 3 divisions

**Operations**: ~9 floating-point operations

---

### Cross Product

**API**: `a.cross(b)`

**What we write**:
```rust
let right = forward.cross(self.up).normalize();
```

**Hidden formula**:
```
a × b = ( a.y·b.z - a.z·b.y,
          a.z·b.x - a.x·b.z,
          a.x·b.y - a.y·b.x )
```

**Explanation**:
- Produces a vector perpendicular to both `a` and `b`
- Direction determined by right-hand rule
- Used to construct coordinate systems (camera right vector)
- Result magnitude: `||a|| × ||b|| × sin(θ)` where θ is angle between vectors

**Hidden complexity**: 6 multiplications + 3 subtractions = 9 operations

**Why it's useful**: Generates perpendicular vectors for camera movement (strafe left/right)

---

### Dot Product

**API**: `a.dot(b)`

**What we write**:
```rust
let diffuse = normal.dot(light_dir);
```

**Hidden formula**:
```
a · b = a.x·b.x + a.y·b.y + a.z·b.z
```

**Alternative form** (for normalized vectors):
```
a · b = ||a|| × ||b|| × cos(θ)
```

Where θ is the angle between the vectors.

**Explanation**:
- Measures how "aligned" two vectors are
- If vectors point same direction: result = +1 (parallel)
- If vectors are perpendicular: result = 0 (orthogonal)
- If vectors point opposite directions: result = -1 (antiparallel)

**Hidden complexity**: 3 multiplications + 2 additions = 5 operations

**Why it's everywhere**: Core operation for lighting, projections, and angle calculations

---

## 3. Metal Shader Built-in Functions

### normalize(v)

**What we write**:
```metal
float3 normal = normalize(in.world_normal);
```

**Hidden formula**:
```
normalize(v) = v / √(v.x² + v.y² + v.z²)
```

**Operations**: Same as glam's normalize (9 operations per call)

**Note**: GPU hardware often has optimized instruction for this

---

### dot(a, b)

**What we write**:
```metal
float diffuse = max(dot(normal, light_dir), 0.0);
```

**Hidden formula**:
```
dot(a, b) = a.x·b.x + a.y·b.y + a.z·b.z
```

**Operations**: 3 multiplications + 2 additions = 5 operations

---

### smoothstep(edge0, edge1, x)

**What we write**:
```metal
float night_mix = 1.0 - smoothstep(NIGHT_MIX_START, NIGHT_MIX_END, diffuse);
```

**Hidden formula** (Hermite interpolation):
```
t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
smoothstep(x) = t² × (3 - 2t)
              = 3t² - 2t³
```

**Graph shape**:
```
1.0 |         ╭─────
    |       ╱
0.5 |     ╱
    |   ╱
0.0 |─╯
    └──────────────
    0  0.25 0.5 0.75 1.0
```

**Explanation**:
- Creates smooth S-curve between two values
- Derivatives are zero at edges (smooth start and end)
- Better than linear interpolation for natural-looking transitions

**Hidden complexity**:
- Subtraction, division, clamping
- 2 multiplications for t²
- 3 multiplications for full polynomial
- ~8-10 operations

**Why it's better**: Eliminates harsh transitions, creates organic blending

---

### mix(a, b, t)

**What we write**:
```metal
float3 blended_color = mix(lit_day_color, night_side, night_mix);
```

**Hidden formula** (linear interpolation / lerp):
```
mix(a, b, t) = a × (1 - t) + b × t
             = a + t × (b - a)    // Optimized form
```

**Explanation**:
- Blends between two values based on factor `t`
- When `t = 0`: returns `a`
- When `t = 1`: returns `b`
- When `t = 0.5`: returns halfway between

**Hidden complexity**:
- Works on scalars, vectors, and colors
- For `float3`: 3× (1 subtraction + 2 multiplications + 1 addition) = 12 operations

---

### pow(x, n)

**What we write**:
```metal
fresnel = pow(fresnel, ATMOSPHERE_GLOW_POWER);  // power = 3.0
```

**Hidden formula**:
```
pow(x, n) = x^n
```

**Implementation** (varies):
- For integer n: Repeated multiplication (e.g., `x³ = x × x × x`)
- For general n: `e^(n × ln(x))` using logarithm and exponential
- GPU may use approximations for speed

**Explanation**:
- For `pow(x, 3)`: Sharpens the Fresnel falloff
- Values close to 0 get pushed even closer to 0
- Values close to 1 stay close to 1
- Creates more dramatic edge effect

**Hidden complexity**: 2 multiplications for n=3, but potentially many more operations for arbitrary exponents

---

### max(a, b)

**What we write**:
```metal
float diffuse = max(dot(normal, light_dir), 0.0);
```

**Hidden formula**:
```
max(a, b) = (a >= b) ? a : b
```

**Explanation**:
- Simple comparison and selection
- Used to clamp negative lighting values to zero
- Prevents "negative light" from back-facing surfaces

**Operations**: 1 comparison + 1 conditional select

---

## 4. Graphics Pipeline (Metal)

These operations happen automatically in the GPU pipeline, without explicit function calls.

### Vertex Transformation

**What Metal does automatically**:
```
clip_position = MVP × vertex_position
```

For each vertex, the GPU:
1. Takes the MVP matrix (16 values)
2. Takes the vertex position (4 values: x, y, z, w)
3. Performs matrix-vector multiplication (16 multiplications + 12 additions)

**Operations per vertex**: 28 floating-point operations

**For our sphere**: 2,145 vertices × 28 operations = ~60,000 operations per frame!

---

### Perspective Divide

**What Metal does automatically** (converts clip space to NDC):
```
NDC.x = clip.x / clip.w
NDC.y = clip.y / clip.w
NDC.z = clip.z / clip.w
```

**Explanation**:
- The `w` component (from perspective projection) encodes depth
- Dividing by `w` creates the perspective effect
- This is why distant objects appear smaller
- Happens automatically after vertex shader

**Operations per vertex**: 3 divisions

---

### Perspective-Correct Interpolation

**What Metal does automatically** for fragment shader inputs (normals, UVs, colors):

**Naive interpolation** (wrong):
```
α = α₀·b₀ + α₁·b₁ + α₂·b₂
```

**Perspective-correct interpolation** (what actually happens):
```
α = (α₀/w₀)·b₀ + (α₁/w₁)·b₁ + (α₂/w₂)·b₂
    ──────────────────────────────────────
           b₀/w₀ + b₁/w₁ + b₂/w₂
```

Where:
- α₀, α₁, α₂: Attribute values at triangle vertices
- b₀, b₁, b₂: Barycentric coordinates (interpolation weights)
- w₀, w₁, w₂: Perspective divide values from each vertex

**Explanation**:
- Without this, textures would appear warped on 3D objects
- Compensates for perspective distortion
- Happens automatically for every pixel!

**Hidden complexity**:
- Many divisions, multiplications, and additions
- Performed millions of times per frame (once per pixel, per attribute)

**Why you don't notice**: Modern GPUs have specialized hardware for this

---

## 5. Additional Hidden Math

### Trigonometric Functions

**What we use**:
```rust
let sin_phi = phi.sin();
let cos_phi = phi.cos();
```

**Hidden implementation**:
- **Taylor series** approximation:
  ```
  sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7! + ...
  cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6! + ...
  ```
- **CORDIC algorithm** (used in many CPUs)
- **Lookup tables with interpolation**

**Complexity**: Can require dozens of operations for high precision

---

### Square Root

**What we use**: `sqrt(x)` (inside `normalize`)

**Hidden implementation**:
- **Newton-Raphson method**:
  ```
  x_{n+1} = (x_n + a/x_n) / 2
  ```
  Iteratively converges to √a

- **Hardware instruction**: Modern CPUs have dedicated sqrt instructions

**Complexity**: Multiple iterations or specialized hardware

---

## Summary: What Libraries Do For You

| Component | Function Calls | Hidden Operations |
|-----------|---------------|------------------|
| **Matrix operations** | 3-4 calls per frame | ~250 floating-point operations |
| **Vector operations** | 10-20 calls per frame | ~100-200 floating-point operations |
| **Shader built-ins** | Millions per frame | Varies by function |
| **GPU pipeline** | Automatic | Millions of operations per frame |

**Key Insight**: For every line of high-level code we write, libraries and hardware execute hundreds or thousands of low-level mathematical operations.

---

## Complexity Comparison

**Code we wrote**: ~80 lines of mathematical logic
**Library code beneath**: ~16,000+ lines of optimized primitives
**Ratio**: ~1:200

### What This Means:

✅ **We focus on**:
- Graphics algorithms (lighting models, effects)
- Geometry generation strategies
- Visual appearance and behavior

✅ **Libraries handle**:
- Linear algebra primitives
- Trigonometric functions
- Matrix operations
- Coordinate transformations

✅ **GPU hardware handles**:
- Parallel execution
- Perspective correction
- Rasterization
- Memory management

---

## The Most Complex Hidden Operations

Ranked by mathematical sophistication:

1. **Perspective projection matrix** (4×4 matrix with tangent, divisions, depth encoding)
2. **Look-at matrix** (multiple cross products, normalizations, basis construction)
3. **Perspective-correct interpolation** (per-pixel depth compensation)
4. **Matrix multiplication** (most computationally expensive per operation)

---

## Why This Matters

Understanding what happens "under the hood" helps you:
- **Optimize performance**: Know which operations are expensive
- **Debug issues**: Understand coordinate spaces and transformations
- **Make informed decisions**: Choose appropriate abstractions
- **Appreciate the stack**: Recognize the engineering in your tools

The earth-3d application is possible because we stand on the shoulders of:
- `glam` library authors (linear algebra)
- Apple Metal engineers (graphics API)
- GPU hardware designers (parallel processing)
- Decades of computer graphics research

---

## Further Learning

To understand these formulas deeply:

- **Linear Algebra**: 3Blue1Brown's "Essence of Linear Algebra" (YouTube)
- **Computer Graphics**: "Real-Time Rendering" by Akenine-Möller
- **Matrix Math**: "Foundations of Game Engine Development" by Eric Lengyel
- **Shader Programming**: "The Book of Shaders" (thebookofshaders.com)
- **GPU Architecture**: "GPU Gems" series (free online)
