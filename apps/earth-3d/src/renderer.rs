#![allow(deprecated)]

use cocoa::appkit::NSView;
use cocoa::base::id as cocoa_id;
use core_graphics_types::geometry::CGSize;
use glam::Mat4;
use metal::*;
use objc::runtime::YES;
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

use crate::camera::Camera;
use crate::geometry::sphere::SphereMesh;
use crate::stars::StarField;
use crate::texture::TextureLoader;

// =============================================================================
// Rendering Configuration Constants
// =============================================================================

// Sphere mesh detail
const SPHERE_SEGMENTS: u32 = 64; // Longitude divisions
const SPHERE_RINGS: u32 = 32; // Latitude divisions

// Star field
const STAR_COUNT: usize = 2000;
const STAR_FIELD_RADIUS: f32 = 50.0;

// Lighting
const AMBIENT_LIGHT_STRENGTH: f32 = 0.15; // Low ambient for dark side contrast
const LIGHT_DIRECTION: [f32; 3] = [-0.6, 0.4, 0.5]; // Sun position (upper-left-front)

// Animation
const DEFAULT_ROTATION_SPEED: f32 = 0.5; // Radians per second
const MIN_ROTATION_SPEED: f32 = 0.0;
const MAX_ROTATION_SPEED: f32 = 5.0;

// =============================================================================

#[repr(C)]
struct Uniforms {
    mvp_matrix: [[f32; 4]; 4],
    model_matrix: [[f32; 4]; 4],
    light_direction: [f32; 3],
    ambient_strength: f32,
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
    earth_texture: Texture,
    night_texture: Texture,
    sampler_state: SamplerState,
    rotation_angle: f32,
    rotation_speed: f32,
    paused: bool,
    // Stars
    stars_pipeline_state: RenderPipelineState,
    stars_vertex_buffer: Buffer,
    stars_count: usize,
}

impl Renderer {
    pub fn new(window: &winit::window::Window) -> Self {
        let device = Device::system_default().expect("No Metal device found");
        let command_queue = device.new_command_queue();

        let layer = Self::create_metal_layer(window, &device);

        // Create sphere geometry
        let sphere = SphereMesh::new(1.0, SPHERE_SEGMENTS, SPHERE_RINGS);

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

        // Load Earth textures
        let texture_loader = TextureLoader::new(&device);
        let earth_texture = texture_loader
            .load_texture("assets/earth_texture.jpg")
            .expect("Failed to load Earth texture");
        let night_texture = texture_loader
            .load_texture("assets/earth_night.jpg")
            .expect("Failed to load night texture");

        // Create sampler state
        let sampler_descriptor = SamplerDescriptor::new();
        sampler_descriptor.set_min_filter(MTLSamplerMinMagFilter::Linear);
        sampler_descriptor.set_mag_filter(MTLSamplerMinMagFilter::Linear);
        sampler_descriptor.set_mip_filter(MTLSamplerMipFilter::Linear);
        sampler_descriptor.set_address_mode_s(MTLSamplerAddressMode::Repeat);
        sampler_descriptor.set_address_mode_t(MTLSamplerAddressMode::Repeat);
        let sampler_state = device.new_sampler(&sampler_descriptor);

        // Compile shaders from source at runtime
        let shader_source = include_str!("../shaders/lit_textured.metal");
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

        // Create stars
        let star_field = StarField::new(STAR_COUNT, STAR_FIELD_RADIUS);
        let stars_count = star_field.stars.len();

        let stars_vertex_buffer = device.new_buffer_with_data(
            star_field.vertex_data().as_ptr() as *const _,
            star_field.vertex_data().len() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        // Compile stars shader
        let stars_shader_source = include_str!("../shaders/stars.metal");
        let stars_library = device
            .new_library_with_source(stars_shader_source, &CompileOptions::new())
            .expect("Failed to compile stars shaders");

        let stars_vertex_function = stars_library.get_function("vertex_main", None).unwrap();
        let stars_fragment_function = stars_library.get_function("fragment_main", None).unwrap();

        // Create stars pipeline
        let stars_pipeline_descriptor = RenderPipelineDescriptor::new();
        stars_pipeline_descriptor.set_vertex_function(Some(&stars_vertex_function));
        stars_pipeline_descriptor.set_fragment_function(Some(&stars_fragment_function));

        // Set up stars vertex descriptor
        let stars_vertex_descriptor = VertexDescriptor::new();

        let stars_position_attr = stars_vertex_descriptor.attributes().object_at(0).unwrap();
        stars_position_attr.set_format(MTLVertexFormat::Float3);
        stars_position_attr.set_offset(0);
        stars_position_attr.set_buffer_index(0);

        let stars_brightness_attr = stars_vertex_descriptor.attributes().object_at(1).unwrap();
        stars_brightness_attr.set_format(MTLVertexFormat::Float);
        stars_brightness_attr.set_offset(12);
        stars_brightness_attr.set_buffer_index(0);

        let stars_layout = stars_vertex_descriptor.layouts().object_at(0).unwrap();
        stars_layout.set_stride(16);
        stars_layout.set_step_function(MTLVertexStepFunction::PerVertex);

        stars_pipeline_descriptor.set_vertex_descriptor(Some(stars_vertex_descriptor));

        let stars_color_attachments = stars_pipeline_descriptor.color_attachments();
        let stars_color_attachment = stars_color_attachments.object_at(0).unwrap();
        stars_color_attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        stars_pipeline_descriptor.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);

        let stars_pipeline_state = device
            .new_render_pipeline_state(&stars_pipeline_descriptor)
            .expect("Failed to create stars pipeline state");

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
            earth_texture,
            night_texture,
            sampler_state,
            rotation_angle: 0.0,
            rotation_speed: DEFAULT_ROTATION_SPEED,
            paused: false,
            stars_pipeline_state,
            stars_vertex_buffer,
            stars_count,
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

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    pub fn adjust_rotation_speed(&mut self, delta: f32) {
        self.rotation_speed =
            (self.rotation_speed + delta).clamp(MIN_ROTATION_SPEED, MAX_ROTATION_SPEED);
    }

    pub fn update(&mut self, delta_time: f32) {
        if !self.paused {
            self.rotation_angle += self.rotation_speed * delta_time;
        }
    }

    pub fn render(&mut self) {
        let drawable = match self.layer.next_drawable() {
            Some(drawable) => drawable,
            None => return,
        };

        // Build transformation matrices
        let model_matrix = Mat4::from_rotation_y(self.rotation_angle);
        let view_matrix = self.camera.view_matrix();
        let projection_matrix = self.camera.projection_matrix();
        let mvp_matrix = projection_matrix * view_matrix * model_matrix;

        // Define sun light coming from upper-left-front (natural angle)
        // Light direction points FROM surface TOWARD the light source
        let light_direction = glam::Vec3::from_array(LIGHT_DIRECTION).normalize();

        let uniforms = Uniforms {
            mvp_matrix: mvp_matrix.to_cols_array_2d(),
            model_matrix: model_matrix.to_cols_array_2d(),
            light_direction: light_direction.to_array(),
            ambient_strength: AMBIENT_LIGHT_STRENGTH,
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

        // Render stars first (background)
        encoder.set_render_pipeline_state(&self.stars_pipeline_state);
        encoder.set_depth_stencil_state(&self.depth_stencil_state);

        let view_projection = projection_matrix * view_matrix;
        let stars_uniforms = view_projection.to_cols_array_2d();
        let stars_uniforms_buffer = self.device.new_buffer_with_data(
            &stars_uniforms as *const [[f32; 4]; 4] as *const _,
            std::mem::size_of::<[[f32; 4]; 4]>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        encoder.set_vertex_buffer(0, Some(&self.stars_vertex_buffer), 0);
        encoder.set_vertex_buffer(1, Some(&stars_uniforms_buffer), 0);

        encoder.draw_primitives(MTLPrimitiveType::Point, 0, self.stars_count as u64);

        // Render Earth
        encoder.set_render_pipeline_state(&self.pipeline_state);
        encoder.set_depth_stencil_state(&self.depth_stencil_state);

        encoder.set_vertex_buffer(0, Some(&self.vertex_buffer), 0);
        encoder.set_vertex_buffer(1, Some(&uniforms_buffer), 0);

        encoder.set_fragment_buffer(0, Some(&uniforms_buffer), 0);
        encoder.set_fragment_texture(0, Some(&self.earth_texture));
        encoder.set_fragment_texture(1, Some(&self.night_texture));
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
