use cocoa::appkit::NSView;
use cocoa::base::id as cocoa_id;
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::runtime::YES;
use std::mem;
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

pub struct Renderer {
    device: Device,
    command_queue: CommandQueue,
    layer: MetalLayer,
}

impl Renderer {
    pub fn new(window: &winit::window::Window) -> Self {
        let device = Device::system_default().expect("No Metal device found");
        let command_queue = device.new_command_queue();

        let layer = Self::create_metal_layer(window, &device);

        println!("Metal renderer initialized");
        println!("  Device: {}", device.name());

        Self {
            device,
            command_queue,
            layer,
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
                    view.setLayer(mem::transmute(layer.as_ref()));
                }
                _ => panic!("Unsupported platform"),
            }
        }

        let size = window.inner_size();
        layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));

        layer
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.layer
            .set_drawable_size(CGSize::new(width as f64, height as f64));
    }

    pub fn render(&mut self) {
        let drawable = match self.layer.next_drawable() {
            Some(drawable) => drawable,
            None => return,
        };

        let command_buffer = self.command_queue.new_command_buffer();

        let render_pass_descriptor = RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();

        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
        color_attachment.set_store_action(MTLStoreAction::Store);

        let encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        encoder.end_encoding();

        command_buffer.present_drawable(drawable);
        command_buffer.commit();
    }
}
