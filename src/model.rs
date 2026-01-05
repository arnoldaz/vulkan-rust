#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps,
    unused_imports
)]

use thiserror::Error;
use anyhow::{anyhow, Result};
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowBuilder};

use std::collections::HashSet;
use std::ffi::CStr;
use std::fs::File;
use std::io::BufReader;
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

use crate::app::AppData;
use crate::vertex::Vertex;
type Vec2 = cgmath::Vector2<f32>;
type Vec3 = cgmath::Vector3<f32>;

use std::collections::HashMap;
use std::hash::{Hash, Hasher};


pub fn load_model(data: &mut AppData) -> Result<()> {
    let mut reader = BufReader::new(File::open("cube.obj")?);

    let (models, _) = tobj::load_obj_buf(
        &mut reader,
        &tobj::LoadOptions { triangulate: true, single_index: true, ..Default::default() },
        |_| Ok(Default::default()),
    )?;

    let mut unique_vertices = HashMap::new();

    println!("{} len models", models.len().to_string());

    for model in &models {
        for index in &model.mesh.indices {
            let pos_offset = (3 * index) as usize;
            let tex_coord_offset = (2 * index) as usize;

            let vertex = Vertex {
                pos: vec3(
                    model.mesh.positions[pos_offset],
                    model.mesh.positions[pos_offset + 1],
                    model.mesh.positions[pos_offset + 2],
                ),
                color: vec3(1.0, 1.0, 1.0),
                tex_coord: vec2(
                    model.mesh.texcoords[tex_coord_offset],
                    1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                ),
            };

            // data.vertices.push(vertex);
            // data.indices.push(data.indices.len() as u32);

            if let Some(index) = unique_vertices.get(&vertex) {
                data.indices.push(*index as u32);
            } else {
                let index = data.vertices.len();
                unique_vertices.insert(vertex, index);
                data.vertices.push(vertex);
                data.indices.push(index as u32);
            }

        }
    }

    println!("Loaded {} vertices, {} indices", data.vertices.len(), data.indices.len());

    Ok(())
}