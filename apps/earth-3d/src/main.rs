#![allow(deprecated)]

use std::collections::HashSet;
use winit::{
    event::{ElementState, Event, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::Window,
};

mod camera;
mod geometry;
mod renderer;
mod stars;
mod texture;

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
    let mut last_update_time = std::time::Instant::now();
    let mut frame_count = 0;
    let mut pressed_keys = HashSet::new();

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
                Event::WindowEvent {
                    event:
                        WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    logical_key, state, ..
                                },
                            ..
                        },
                    ..
                } => match state {
                    ElementState::Pressed => {
                        pressed_keys.insert(logical_key.clone());
                    }
                    ElementState::Released => {
                        pressed_keys.remove(&logical_key);
                    }
                },
                Event::AboutToWait => {
                    // Update camera based on input
                    let delta_time = last_update_time.elapsed().as_secs_f32();
                    last_update_time = std::time::Instant::now();

                    let move_speed = 2.0 * delta_time; // 2 units per second

                    // Handle camera movement (scoped to release borrow)
                    {
                        let camera = renderer.camera_mut();
                        if pressed_keys.contains(&Key::Named(NamedKey::ArrowUp)) {
                            camera.move_forward(move_speed);
                        }
                        if pressed_keys.contains(&Key::Named(NamedKey::ArrowDown)) {
                            camera.move_backward(move_speed);
                        }
                        if pressed_keys.contains(&Key::Named(NamedKey::ArrowLeft)) {
                            camera.strafe_left(move_speed);
                        }
                        if pressed_keys.contains(&Key::Named(NamedKey::ArrowRight)) {
                            camera.strafe_right(move_speed);
                        }

                        // Handle zoom with +/- keys
                        if pressed_keys.contains(&Key::Character("=".into()))
                            || pressed_keys.contains(&Key::Character("+".into()))
                        {
                            camera.move_forward(move_speed * 2.0);
                        }
                        if pressed_keys.contains(&Key::Character("-".into())) {
                            camera.move_backward(move_speed * 2.0);
                        }
                    }

                    // Handle spacebar for pause
                    if pressed_keys.contains(&Key::Named(NamedKey::Space)) {
                        renderer.toggle_pause();
                        pressed_keys.remove(&Key::Named(NamedKey::Space)); // Prevent repeated toggles
                    }

                    // Handle rotation speed with [ ] keys
                    if pressed_keys.contains(&Key::Character("[".into())) {
                        renderer.adjust_rotation_speed(-0.5 * delta_time);
                    }
                    if pressed_keys.contains(&Key::Character("]".into())) {
                        renderer.adjust_rotation_speed(0.5 * delta_time);
                    }

                    // Update renderer (rotation)
                    renderer.update(delta_time);

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
