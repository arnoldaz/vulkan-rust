#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps,
    unused_imports
)]

mod vertex;
mod app;
mod vulkan;
mod swapchain;

use app::App;
use vertex::{Vertex, VERTICES};

use thiserror::Error;
use anyhow::{anyhow, Result};
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowBuilder};

use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;

use log::*;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::window as vk_window;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::ExtDebugUtilsExtension;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;
use vulkanalia::bytecode::Bytecode;

use std::mem::size_of;
use cgmath::{vec2, vec3};
type Vec2 = cgmath::Vector2<f32>;
type Vec3 = cgmath::Vector3<f32>;

use std::ptr::copy_nonoverlapping as memcpy;


// NEXT: https://kylemayes.github.io/vulkanalia/vertex/staging_buffer.html


fn main() -> Result<()> {
    pretty_env_logger::init();

    // Window
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Liurauras ass (Animal Well 2)")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    // App
    let mut app = unsafe { App::create(&window)? };
    let mut minimized = false;
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            // Request a redraw when all events were processed.
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::RedrawRequested if !elwt.exiting() && !minimized => {
                    unsafe { app.render(&window) }.unwrap();
                },
                WindowEvent::CloseRequested => {
                    elwt.exit();
                    unsafe { app.device.device_wait_idle().unwrap(); }
                    unsafe { app.destroy(); }
                },
                WindowEvent::KeyboardInput {
                    event: KeyEvent { logical_key: key, state: ElementState::Pressed, .. },
                    ..
                } => match key.as_ref() {
                    // WARNING: Consider using `key_without_modifiers()` if available on your platform.
                    // See the `key_binding` example
                    Key::Named(NamedKey::Escape) => {
                        elwt.exit();
                        unsafe { app.device.device_wait_idle().unwrap(); }
                        unsafe { app.destroy(); }
                    },
                    _ => (),
                },
                WindowEvent::Resized(size) => {
                    if size.width == 0 || size.height == 0 {
                        minimized = true;
                    } else {
                        minimized = false;
                        app.resized = true;
                    }
                }
                _ => {}
            }
            _ => {}
        }
    })?;

    Ok(())
}
