#![allow(deprecated)]

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

mod camera;
mod geometry;
mod renderer;

use renderer::Renderer;

fn main() {
    println!("Earth 3D - Initializing...");

    let event_loop = EventLoop::new().unwrap();
    let window_attributes = Window::default_attributes()
        .with_title("Earth 3D - Metal Visualization")
        .with_inner_size(winit::dpi::LogicalSize::new(1024, 768));
    let window = event_loop.create_window(window_attributes).unwrap();

    let mut renderer = Renderer::new(&window);
    let mut last_frame_time = std::time::Instant::now();
    let mut frame_count = 0;

    event_loop
        .run(move |event, event_loop_window_target| {
            event_loop_window_target.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    println!("Close requested, exiting...");
                    event_loop_window_target.exit();
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => {
                    renderer.resize(size.width, size.height);
                }
                Event::AboutToWait => {
                    window.request_redraw();
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    ..
                } => {
                    renderer.render();

                    frame_count += 1;
                    let elapsed = last_frame_time.elapsed();
                    if elapsed.as_secs_f32() >= 1.0 {
                        let fps = frame_count as f32 / elapsed.as_secs_f32();
                        frame_count = 0;
                        last_frame_time = std::time::Instant::now();

                        window.set_title(&format!("Earth 3D - {:.1} FPS", fps));
                    }
                }
                _ => {}
            }
        })
        .unwrap();
}
