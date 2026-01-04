use image::GenericImageView;
use metal::*;

pub struct TextureLoader {
    device: Device,
}

impl TextureLoader {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }

    pub fn load_texture(&self, image_path: &str) -> Result<Texture, String> {
        let img = image::open(image_path)
            .map_err(|e| format!("Failed to load image {}: {}", image_path, e))?;

        let img_rgba = img.to_rgba8();
        let (width, height) = img.dimensions();

        let texture_descriptor = TextureDescriptor::new();
        texture_descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        texture_descriptor.set_width(width as u64);
        texture_descriptor.set_height(height as u64);
        texture_descriptor.set_usage(MTLTextureUsage::ShaderRead);
        texture_descriptor.set_storage_mode(MTLStorageMode::Managed);

        let texture = self.device.new_texture(&texture_descriptor);

        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: width as u64,
                height: height as u64,
                depth: 1,
            },
        };

        texture.replace_region(
            region,
            0,
            img_rgba.as_raw().as_ptr() as *const _,
            (width * 4) as u64,
        );

        Ok(texture)
    }
}
