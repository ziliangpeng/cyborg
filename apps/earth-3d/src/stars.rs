use glam::Vec3;
use rand::Rng;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Star {
    pub position: Vec3,
    pub brightness: f32,
}

pub struct StarField {
    pub stars: Vec<Star>,
}

impl StarField {
    pub fn new(count: usize, radius: f32) -> Self {
        let mut rng = rand::thread_rng();
        let mut stars = Vec::with_capacity(count);

        for _ in 0..count {
            // Generate random point on sphere surface
            let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
            let phi = rng.gen::<f32>() * std::f32::consts::PI;

            let x = radius * phi.sin() * theta.cos();
            let y = radius * phi.cos();
            let z = radius * phi.sin() * theta.sin();

            let brightness = rng.gen::<f32>() * 0.5 + 0.5; // 0.5 to 1.0

            stars.push(Star {
                position: Vec3::new(x, y, z),
                brightness,
            });
        }

        Self { stars }
    }

    pub fn vertex_data(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.stars.as_ptr() as *const u8,
                self.stars.len() * std::mem::size_of::<Star>(),
            )
        }
    }
}
