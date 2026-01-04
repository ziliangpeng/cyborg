use glam::{Vec2, Vec3};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

pub struct SphereMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl SphereMesh {
    pub fn new(radius: f32, segments: u32, rings: u32) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Generate vertices
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
                let normal = Vec3::new(x, y, z).normalize();

                let u = (segment as f32) / (segments as f32);
                let v = (ring as f32) / (rings as f32);
                let uv = Vec2::new(u, v);

                vertices.push(Vertex {
                    position,
                    normal,
                    uv,
                });
            }
        }

        // Generate indices
        for ring in 0..rings {
            for segment in 0..segments {
                let current = ring * (segments + 1) + segment;
                let next = current + segments + 1;

                indices.push(current);
                indices.push(next);
                indices.push(current + 1);

                indices.push(current + 1);
                indices.push(next);
                indices.push(next + 1);
            }
        }

        Self { vertices, indices }
    }

    pub fn vertex_data(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.vertices.as_ptr() as *const u8,
                self.vertices.len() * std::mem::size_of::<Vertex>(),
            )
        }
    }

    pub fn index_data(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.indices.as_ptr() as *const u8,
                self.indices.len() * std::mem::size_of::<u32>(),
            )
        }
    }
}
