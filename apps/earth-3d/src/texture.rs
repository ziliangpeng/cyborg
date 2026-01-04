use image::GenericImageView;
use metal::*;
use std::path::PathBuf;

pub struct TextureLoader {
    device: Device,
}

impl TextureLoader {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }

    /// Resolve a path that works both with Bazel runfiles and regular cargo runs
    fn resolve_path(relative_path: &str) -> PathBuf {
        // Try multiple possible locations for the asset:

        // 1. Relative path (for cargo run)
        let cargo_path = PathBuf::from(relative_path);
        if cargo_path.exists() {
            return cargo_path;
        }

        // 2. Check if we're running from Bazel by looking for RUNFILES_DIR
        if let Ok(runfiles_dir) = std::env::var("RUNFILES_DIR") {
            let bazel_path = PathBuf::from(runfiles_dir)
                .join("_main")
                .join("apps")
                .join("earth-3d")
                .join(relative_path);
            if bazel_path.exists() {
                return bazel_path;
            }
        }

        // 3. Check alternative Bazel runfiles location (runfiles manifest)
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                // Check <binary>.runfiles/_main/apps/earth-3d/<path>
                let runfiles_path = exe_dir
                    .join("earth-3d.runfiles")
                    .join("_main")
                    .join("apps")
                    .join("earth-3d")
                    .join(relative_path);
                if runfiles_path.exists() {
                    return runfiles_path;
                }
            }
        }

        // Fall back to original relative path
        PathBuf::from(relative_path)
    }

    pub fn load_texture(&self, image_path: &str) -> Result<Texture, String> {
        let resolved_path = Self::resolve_path(image_path);
        let img = image::open(&resolved_path)
            .map_err(|e| format!("Failed to load image {:?}: {}", resolved_path, e))?;

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
