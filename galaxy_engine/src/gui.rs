// Copyright (c) 2025 Ben Sutherland.

use std::sync::Arc;

use ash::prelude::VkResult;
use ash::vk;
use egui::epaint;
use ultraviolet::Vec2;

use crate::engine::GalaxyEngine;
use crate::meshes::MeshBuffer;
use crate::pipelines::{GraphicsPipeline, Pipeline};
use crate::vulkan::command_buffer::{RenderingCmdBuf, TransientPrimaryCommandPool};
use crate::vulkan::debug::debug_only_name;
use crate::vulkan::descriptors::DescriptorPool;
use crate::vulkan::device::Device;
use crate::vulkan::gpu_alloc::MemResult;

// Handles GUI rendering using egui.
pub(crate) struct GuiRenderer {
    ctx: egui::Context,
    window: Arc<winit::window::Window>,
    window_state: egui_winit::State,
    primitives: Vec<egui::ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
    descriptor_pool: DescriptorPool<{ GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    pipeline: Arc<GraphicsPipeline>,
    // TODO: These mesh buffers cause issues on shutdown.
    mesh_buffers: [Option<MeshBuffer<epaint::Vertex>>; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
}

impl GuiRenderer {
    const GUI_PIPELINE_ID: &'static str = "/engine/gui/gui";
    const MAX_FONT_TEXTURES: u32 = 512;
    //const MAX_UI_VERTICES: usize = 1024 * 1024;
    //const MAX_UI_INDICES: usize = 1024 * 1024;

    pub fn new(
        ctx: egui::Context,
        window: Arc<winit::window::Window>,
        event_loop: &winit::event_loop::ActiveEventLoop,
        engine: &GalaxyEngine,
    ) -> VkResult<Self> {
        let window_state = egui_winit::State::new(
            ctx.clone(),
            ctx.viewport_id(),
            event_loop,
            Some(window.scale_factor() as f32),
            event_loop.system_theme(),
            Some(
                engine
                    .device
                    .physical_device()
                    .properties
                    .base
                    .limits
                    .max_image_dimension2_d as usize,
            ),
        );

        // TODO: Use custom GUI layout.

        // Create descriptor pool.
        let descriptor_pool_sizes = [
            // Scene uniform buffer.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(GalaxyEngine::MAX_FRAMES_IN_FLIGHT as u32),
            // Transforms + draw data + material constants.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(GalaxyEngine::MAX_FRAMES_IN_FLIGHT as u32 * 3),
            // Scene texture descriptor array.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count((GalaxyEngine::MAX_FRAMES_IN_FLIGHT as u32 * Self::MAX_FONT_TEXTURES).max(1)),
        ];
        let mut descriptor_pool = DescriptorPool::new(&engine.device, &descriptor_pool_sizes)?;

        descriptor_pool.allocate_descriptor_sets(
            &engine.device,
            &[engine.pipeline_manager.scene_set_layout; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
        )?;

        let pipeline = engine
            .pipeline_manager
            .get_cloned_graphics_pipeline(Self::GUI_PIPELINE_ID)
            .expect("GUI graphics pipeline not found");

        Ok(Self {
            ctx,
            window,
            window_state,
            primitives: Vec::new(),
            textures_delta: egui::TexturesDelta::default(),
            descriptor_pool,
            pipeline,
            mesh_buffers: [None, None],
        })
    }

    // Returns true if input should be passed to the game.
    pub fn on_window_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        let response = self.window_state.on_window_event(&self.window, event);
        if response.repaint {
            self.ctx.request_repaint();
        }
        !response.consumed
    }

    pub fn on_mouse_motion(&mut self, delta: (f64, f64)) {
        self.window_state.on_mouse_motion(delta);
    }

    pub fn build_ui(&mut self, ui_fn: impl FnMut(&egui::Context)) {
        // Get viewport info.
        let mut viewport_info = egui::ViewportInfo::default();
        egui_winit::update_viewport_info(&mut viewport_info, &self.ctx, &self.window, false);

        // Get accumulated egui input.
        let mut egui_input = self.window_state.take_egui_input(&self.window);
        egui_input.viewports.insert(egui_input.viewport_id, viewport_info);

        // Run GUI building callback.
        let run_output = self.ctx.run(egui_input, ui_fn);

        // Update state.
        self.window_state
            .handle_platform_output(&self.window, run_output.platform_output);

        // Save render data.
        self.primitives = self.ctx.tessellate(run_output.shapes, run_output.pixels_per_point);
        self.textures_delta = run_output.textures_delta;
    }

    pub fn render(
        &mut self,
        device: &Device,
        framebuffer_size: vk::Extent2D,
        frame_index: usize,
        transient_cmd_pool: &mut TransientPrimaryCommandPool,
        cmd_buf: &mut RenderingCmdBuf,
    ) -> MemResult<()> {
        let scale_factor = self.window.scale_factor() as f32;
        let screen_size = Vec2::new(framebuffer_size.width as f32, framebuffer_size.height as f32) / scale_factor;

        struct DrawParams {
            first_index: u32,
            index_count: u32,
            vertex_offset: i32,
            scissor_rect: vk::Rect2D,
        }

        // Set up mesh buffers.
        let primitives = std::mem::take(&mut self.primitives);
        let mesh_iter = primitives
            .iter()
            .filter_map(|clipped_primitive| match &clipped_primitive.primitive {
                epaint::Primitive::Mesh(mesh) => Some((clipped_primitive.clip_rect, mesh)),
                _ => None,
            });
        let mut total = (0, 0);
        let draw_params: Vec<_> = mesh_iter
            .clone()
            .scan(&mut total, |(first_index_mut, vertex_offset_mut), (rect, mesh)| {
                let first_index = *first_index_mut;
                let vertex_offset = *vertex_offset_mut;
                *first_index_mut += mesh.indices.len() as u32;
                *vertex_offset_mut += mesh.vertices.len() as i32;

                let scissor_rect = Self::get_rect_scissor(scale_factor, framebuffer_size, rect);
                Some(DrawParams {
                    first_index,
                    index_count: mesh.indices.len() as u32,
                    vertex_offset,
                    scissor_rect,
                })
            })
            .collect();

        let (indices, vertices) = mesh_iter.fold(
            (
                Vec::with_capacity(total.0 as usize),
                Vec::with_capacity(total.1 as usize),
            ),
            |(mut indices, mut vertices), (_, mesh)| {
                indices.extend_from_slice(&mesh.indices);
                vertices.extend_from_slice(&mesh.vertices);
                log::info!("Mesh vertices colour: {:?}", &vertices[0].color.a());
                (indices, vertices)
            },
        );

        let mesh_buffer = MeshBuffer::new_from_vertices_and_indices(
            debug_only_name!("GUI mesh buffer"),
            &vertices,
            &indices,
            device,
            transient_cmd_pool,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
        )?;

        // Set up textures.

        //let descriptor_writes: Vec<_> = self
        //    .descriptor_pool
        //    .iter()
        //    .map(|set| {
        //        // Textures array.
        //        vk::WriteDescriptorSet::default()
        //            .dst_set(*set)
        //            .dst_binding(4) // Texture buffer is index 3 in the scene descriptor set layout.
        //            .dst_array_element(0)
        //            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        //            .image_info(&texture_image_infos)
        //    })
        //    .collect();

        //unsafe { device.loader().update_descriptor_sets(&descriptor_writes, &[]) };

        // Queue draw commands.
        let pipeline_layout = self.pipeline.layout();
        cmd_buf.bind_graphics_pipeline(&self.pipeline);

        // Bind descriptor set at index 0.
        cmd_buf.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout,
            0,
            &[self.descriptor_pool.get(frame_index)],
            &[],
        );

        cmd_buf.push_constants(
            pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytemuck::bytes_of(&screen_size),
        );
        mesh_buffer.bind(cmd_buf);

        let mut current_scissor = None;
        for draw in draw_params {
            if current_scissor != Some(draw.scissor_rect) {
                current_scissor = Some(draw.scissor_rect);
                cmd_buf.set_scissor(draw.scissor_rect);
            }
            let texture_index = 0u32;
            cmd_buf.push_constants(
                self.pipeline.layout(),
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                std::mem::size_of::<Vec2>() as u32,
                bytemuck::bytes_of(&texture_index),
            );
            cmd_buf.draw_indexed(draw.index_count, 1, draw.first_index, draw.vertex_offset, 0);
        }

        // Keep mesh buffer around for until it's finished.
        self.mesh_buffers[frame_index] = Some(mesh_buffer);

        Ok(())
    }

    // https://github.com/hakolao/egui_winit_vulkano/blob/00bd40b491f397611fcf555e79134f7d2928fef1/src/renderer.rs#L547
    fn get_rect_scissor(scale_factor: f32, framebuffer_dimensions: vk::Extent2D, rect: epaint::Rect) -> vk::Rect2D {
        let min = rect.min;
        let min = egui::Pos2 {
            x: min.x * scale_factor,
            y: min.y * scale_factor,
        };
        let min = egui::Pos2 {
            x: min.x.clamp(0.0, framebuffer_dimensions.width as f32),
            y: min.y.clamp(0.0, framebuffer_dimensions.height as f32),
        };
        let max = rect.max;
        let max = egui::Pos2 {
            x: max.x * scale_factor,
            y: max.y * scale_factor,
        };
        let max = egui::Pos2 {
            x: max.x.clamp(min.x, framebuffer_dimensions.width as f32),
            y: max.y.clamp(min.y, framebuffer_dimensions.height as f32),
        };
        vk::Rect2D {
            offset: vk::Offset2D {
                x: min.x.round() as i32,
                y: min.y.round() as i32,
            },
            extent: vk::Extent2D {
                width: (max.x.round() - min.x) as u32,
                height: (max.y.round() - min.y) as u32,
            },
        }
    }
}
