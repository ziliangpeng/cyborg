#![allow(deprecated)]

use cocoa::appkit::NSView;
use cocoa::base::id as cocoa_id;
use core_graphics_types::geometry::CGSize;
use glam::Mat4;
use metal::*;
use objc::runtime::YES;
use std::time::Instant;
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

use crate::camera::Camera;
use crate::geometry::sphere::SphereMesh;
use crate::texture::TextureLoader;

#[repr(C)]
struct Uniforms {
    mvp_matrix: [[f32; 4]; 4],
    model_matrix: [[f32; 4]; 4],
}

pub struct Renderer {
    device: Device,
    command_queue: CommandQueue,
    layer: MetalLayer,
    pipeline_state: RenderPipelineState,
    depth_stencil_state: DepthStencilState,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    index_count: usize,
    camera: Camera,
    start_time: Instant,
    earth_texture: Texture,
    sampler_state: SamplerState,
}

impl Renderer {
    pub fn new(window: &winit::window::Window) -> Self {
        let device = Device::system_default().expect("No Metal device found");
        let command_queue = device.new_command_queue();

        let layer = Self::create_metal_layer(window, &device);

        // Create sphere geometry
        let sphere = SphereMesh::new(1.0, 64, 32);

        let vertex_buffer = device.new_buffer_with_data(
            sphere.vertex_data().as_ptr() as *const _,
            sphere.vertex_data().len() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        let index_buffer = device.new_buffer_with_data(
            sphere.index_data().as_ptr() as *const _,
            sphere.index_data().len() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        let index_count = sphere.indices.len();

        // Load Earth texture
        let texture_loader = TextureLoader::new(&device);
        let earth_texture = texture_loader
            .load_texture("assets/earth_texture.jpg")
            .expect("Failed to load Earth texture");

        // Create sampler state
        let sampler_descriptor = SamplerDescriptor::new();
        sampler_descriptor.set_min_filter(MTLSamplerMinMagFilter::Linear);
        sampler_descriptor.set_mag_filter(MTLSamplerMinMagFilter::Linear);
        sampler_descriptor.set_mip_filter(MTLSamplerMipFilter::Linear);
        sampler_descriptor.set_address_mode_s(MTLSamplerAddressMode::Repeat);
        sampler_descriptor.set_address_mode_t(MTLSamplerAddressMode::Repeat);
        let sampler_state = device.new_sampler(&sampler_descriptor);

        // Compile shaders from source at runtime
        let shader_source = include_str!("../shaders/textured.metal");
        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .expect("Failed to compile Metal shaders");

        let vertex_function = library.get_function("vertex_main", None).unwrap();
        let fragment_function = library.get_function("fragment_main", None).unwrap();

        // Create render pipeline
        let pipeline_descriptor = RenderPipelineDescriptor::new();
        pipeline_descriptor.set_vertex_function(Some(&vertex_function));
        pipeline_descriptor.set_fragment_function(Some(&fragment_function));

        // Set up vertex descriptor
        let vertex_descriptor = VertexDescriptor::new();

        let position_attr = vertex_descriptor.attributes().object_at(0).unwrap();
        position_attr.set_format(MTLVertexFormat::Float3);
        position_attr.set_offset(0);
        position_attr.set_buffer_index(0);

        let normal_attr = vertex_descriptor.attributes().object_at(1).unwrap();
        normal_attr.set_format(MTLVertexFormat::Float3);
        normal_attr.set_offset(12);
        normal_attr.set_buffer_index(0);

        let uv_attr = vertex_descriptor.attributes().object_at(2).unwrap();
        uv_attr.set_format(MTLVertexFormat::Float2);
        uv_attr.set_offset(24);
        uv_attr.set_buffer_index(0);

        let layout = vertex_descriptor.layouts().object_at(0).unwrap();
        layout.set_stride(32);
        layout.set_step_function(MTLVertexStepFunction::PerVertex);

        pipeline_descriptor.set_vertex_descriptor(Some(vertex_descriptor));

        let color_attachments = pipeline_descriptor.color_attachments();
        let color_attachment = color_attachments.object_at(0).unwrap();
        color_attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        pipeline_descriptor.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);

        let pipeline_state = device
            .new_render_pipeline_state(&pipeline_descriptor)
            .expect("Failed to create pipeline state");

        // Create depth stencil state
        let depth_stencil_descriptor = DepthStencilDescriptor::new();
        depth_stencil_descriptor.set_depth_compare_function(MTLCompareFunction::Less);
        depth_stencil_descriptor.set_depth_write_enabled(true);

        let depth_stencil_state = device.new_depth_stencil_state(&depth_stencil_descriptor);

        let size = window.inner_size();
        let aspect = size.width as f32 / size.height as f32;
        let camera = Camera::new(aspect);

        println!("Metal renderer initialized");
        println!("  Device: {}", device.name());
        println!(
            "  Sphere: {} vertices, {} indices",
            sphere.vertices.len(),
            index_count
        );
        println!(
            "  Earth texture: {}x{}",
            earth_texture.width(),
            earth_texture.height()
        );

        Self {
            device,
            command_queue,
            layer,
            pipeline_state,
            depth_stencil_state,
            vertex_buffer,
            index_buffer,
            index_count,
            camera,
            start_time: Instant::now(),
            earth_texture,
            sampler_state,
        }
    }

    fn create_metal_layer(window: &winit::window::Window, device: &Device) -> MetalLayer {
        let layer = MetalLayer::new();
        layer.set_device(device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);

        unsafe {
            let wh = window.window_handle().unwrap();
            match wh.as_raw() {
                RawWindowHandle::AppKit(handle) => {
                    let view = handle.ns_view.as_ptr() as cocoa_id;
                    view.setWantsLayer(YES);
                    view.setLayer(layer.as_ref() as *const metal::MetalLayerRef as *mut _);
                }
                _ => panic!("Unsupported platform"),
            }
        }

        let size = window.inner_size();
        layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));

        layer
    }

    fn create_depth_texture(&self, width: u64, height: u64) -> Texture {
        let descriptor = TextureDescriptor::new();
        descriptor.set_pixel_format(MTLPixelFormat::Depth32Float);
        descriptor.set_width(width);
        descriptor.set_height(height);
        descriptor.set_usage(MTLTextureUsage::RenderTarget);
        descriptor.set_storage_mode(MTLStorageMode::Private);

        self.device.new_texture(&descriptor)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.layer
            .set_drawable_size(CGSize::new(width as f64, height as f64));
        self.camera.update_aspect(width as f32 / height as f32);
    }

    pub fn render(&mut self) {
        let drawable = match self.layer.next_drawable() {
            Some(drawable) => drawable,
            None => return,
        };

        // Calculate rotation
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let rotation_angle = elapsed * 0.5;

        // Build transformation matrices
        let model_matrix = Mat4::from_rotation_y(rotation_angle);
        let view_matrix = self.camera.view_matrix();
        let projection_matrix = self.camera.projection_matrix();
        let mvp_matrix = projection_matrix * view_matrix * model_matrix;

        let uniforms = Uniforms {
            mvp_matrix: mvp_matrix.to_cols_array_2d(),
            model_matrix: model_matrix.to_cols_array_2d(),
        };

        let uniforms_buffer = self.device.new_buffer_with_data(
            &uniforms as *const Uniforms as *const _,
            std::mem::size_of::<Uniforms>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        // Create depth texture
        let texture = drawable.texture();
        let depth_texture = self.create_depth_texture(texture.width(), texture.height());

        // Set up render pass
        let render_pass_descriptor = RenderPassDescriptor::new();

        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
        color_attachment.set_store_action(MTLStoreAction::Store);

        let depth_attachment = render_pass_descriptor.depth_attachment().unwrap();
        depth_attachment.set_texture(Some(&depth_texture));
        depth_attachment.set_load_action(MTLLoadAction::Clear);
        depth_attachment.set_clear_depth(1.0);
        depth_attachment.set_store_action(MTLStoreAction::DontCare);

        // Encode rendering commands
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);

        encoder.set_render_pipeline_state(&self.pipeline_state);
        encoder.set_depth_stencil_state(&self.depth_stencil_state);

        encoder.set_vertex_buffer(0, Some(&self.vertex_buffer), 0);
        encoder.set_vertex_buffer(1, Some(&uniforms_buffer), 0);

        encoder.set_fragment_texture(0, Some(&self.earth_texture));
        encoder.set_fragment_sampler_state(0, Some(&self.sampler_state));

        encoder.set_cull_mode(MTLCullMode::Back);
        encoder.set_front_facing_winding(MTLWinding::CounterClockwise);

        encoder.draw_indexed_primitives(
            MTLPrimitiveType::Triangle,
            self.index_count as u64,
            MTLIndexType::UInt32,
            &self.index_buffer,
            0,
        );

        encoder.end_encoding();

        command_buffer.present_drawable(drawable);
        command_buffer.commit();
    }
}
