// Copyright (c) 2025 Ben Sutherland.

use std::collections::HashMap;
use std::sync::Arc;

use ash::prelude::VkResult;
use ash::vk;
use egui::epaint;
use indexmap::IndexMap;
use itertools::izip;
use ultraviolet::Vec2;

use crate::engine::GalaxyEngine;
use crate::meshes::MeshBuffer;
use crate::pipelines::{GraphicsPipeline, Pipeline, PipelineManager};
use crate::vulkan::buffer::{Buffer, Staging};
use crate::vulkan::command_buffer::{RenderingCmdBuf, TransientPrimaryCommandPool};
use crate::vulkan::debug::debug_only_name;
use crate::vulkan::descriptors::DescriptorPool;
use crate::vulkan::device::{Device, SharedDeviceLoader};
use crate::vulkan::gpu_alloc::MemResult;
use crate::vulkan::image::Image;

#[derive(Default)]
struct GuiDrawData {
    primitives: Vec<egui::ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
}

// Handles GUI rendering using egui.
pub(crate) struct GuiIntegration {
    ctx: egui::Context,
    window: Arc<winit::window::Window>,
    window_state: egui_winit::State,
    draw_data: GuiDrawData,
}

impl GuiIntegration {
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

        Ok(Self {
            ctx,
            window,
            window_state,
            draw_data: GuiDrawData::default(),
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
        self.draw_data.primitives = self.ctx.tessellate(run_output.shapes, run_output.pixels_per_point);
        self.draw_data.textures_delta = run_output.textures_delta;
    }

    pub fn render(
        &mut self,
        renderer: &mut GuiRenderer,
        device: &Device,
        framebuffer_size: vk::Extent2D,
        frame_index: usize,
        transient_cmd_pool: &mut TransientPrimaryCommandPool,
        cmd_buf: &mut RenderingCmdBuf,
    ) -> MemResult<()> {
        let scale_factor = self.window.scale_factor() as f32;

        renderer.render(
            device,
            std::mem::take(&mut self.draw_data),
            scale_factor,
            framebuffer_size,
            frame_index,
            transient_cmd_pool,
            cmd_buf,
        )
    }
}

struct FontTexture {
    image: Image,
    sampler: vk::Sampler,
}

pub(crate) struct GuiRenderer {
    loader: SharedDeviceLoader,
    descriptor_pool: DescriptorPool<{ GalaxyEngine::MAX_FRAMES_IN_FLIGHT }>,
    pipeline: Arc<GraphicsPipeline>,
    mesh_buffers: [Option<MeshBuffer<epaint::Vertex>>; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
    textures: IndexMap<egui::TextureId, FontTexture>,
    samplers: HashMap<egui::TextureOptions, vk::Sampler>,
    descriptors_dirty: [bool; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
}

impl GuiRenderer {
    const GUI_PIPELINE_ID: &'static str = "/engine/gui/gui";
    const MAX_FONT_TEXTURES: u32 = 512;
    //const MAX_UI_VERTICES: usize = 1024 * 1024;
    //const MAX_UI_INDICES: usize = 1024 * 1024;

    pub fn new(device: &Device, pipeline_manager: &PipelineManager) -> VkResult<Self> {
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
        let mut descriptor_pool = DescriptorPool::new(device, &descriptor_pool_sizes)?;

        descriptor_pool.allocate_descriptor_sets(
            device,
            &[pipeline_manager.scene_set_layout; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
        )?;

        let pipeline = pipeline_manager
            .get_cloned_graphics_pipeline(Self::GUI_PIPELINE_ID)
            .expect("GUI graphics pipeline not found");

        Ok(Self {
            loader: device.cloned_loader(),
            descriptor_pool,
            pipeline,
            mesh_buffers: Default::default(),
            textures: IndexMap::new(),
            samplers: HashMap::new(),
            descriptors_dirty: [true; GalaxyEngine::MAX_FRAMES_IN_FLIGHT],
        })
    }

    fn render(
        &mut self,
        device: &Device,
        draw_data: GuiDrawData,
        scale_factor: f32,
        framebuffer_size: vk::Extent2D,
        frame_index: usize,
        transient_cmd_pool: &mut TransientPrimaryCommandPool,
        cmd_buf: &mut RenderingCmdBuf,
    ) -> MemResult<()> {
        let screen_size = Vec2::new(framebuffer_size.width as f32, framebuffer_size.height as f32) / scale_factor;

        struct DrawParams {
            first_index: u32,
            index_count: u32,
            vertex_offset: i32,
            scissor_rect: vk::Rect2D,
            texture_id: egui::TextureId,
        }

        // Set up mesh buffers.
        let mesh_iter =
            draw_data
                .primitives
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
                    texture_id: mesh.texture_id,
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
                //log::info!("Mesh vertices colour: {:?}", &vertices[0].color.a());
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
        self.upload_textures(device, &draw_data.textures_delta, frame_index, transient_cmd_pool)?;

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
            let texture_index = self.textures.get_index_of(&draw.texture_id).unwrap() as u32;
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

    fn get_or_create_sampler(&mut self, device: &Device, options: &egui::TextureOptions) -> VkResult<vk::Sampler> {
        if let Some(sampler) = self.samplers.get(options) {
            return Ok(*sampler);
        }

        fn into_vk_filter(filter: egui::TextureFilter) -> vk::Filter {
            match filter {
                egui::TextureFilter::Nearest => vk::Filter::NEAREST,
                egui::TextureFilter::Linear => vk::Filter::LINEAR,
            }
        }

        fn into_vk_wrap_mode(wrap_mode: egui::TextureWrapMode) -> vk::SamplerAddressMode {
            match wrap_mode {
                egui::TextureWrapMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
                egui::TextureWrapMode::Repeat => vk::SamplerAddressMode::REPEAT,
                egui::TextureWrapMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            }
        }

        fn into_vk_mipmap_mode(filter: egui::TextureFilter) -> vk::SamplerMipmapMode {
            match filter {
                egui::TextureFilter::Nearest => vk::SamplerMipmapMode::NEAREST,
                egui::TextureFilter::Linear => vk::SamplerMipmapMode::LINEAR,
            }
        }

        // Create a new sampler with the given options.
        let wrap_mode = into_vk_wrap_mode(options.wrap_mode);
        let default_sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(into_vk_filter(options.magnification))
            .min_filter(into_vk_filter(options.minification))
            .address_mode_u(wrap_mode)
            .address_mode_v(wrap_mode)
            .address_mode_w(wrap_mode)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .anisotropy_enable(false)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mipmap_mode(
                options
                    .mipmap_mode
                    .map(into_vk_mipmap_mode)
                    .unwrap_or(vk::SamplerMipmapMode::LINEAR),
            )
            .mip_lod_bias(0.)
            .min_lod(0.)
            .max_lod(vk::LOD_CLAMP_NONE);
        let sampler = unsafe { device.loader().create_sampler(&default_sampler_info, None) }?;

        self.samplers.insert(*options, sampler);

        Ok(sampler)
    }

    fn write_texture_descriptors(&mut self, frame_index: usize) {
        if !self.descriptors_dirty[frame_index] {
            return;
        }

        // Write descriptors.
        let texture_image_infos: Vec<_> = self
            .textures
            .values()
            .map(|texture| {
                vk::DescriptorImageInfo::default()
                    .image_view(texture.image.view().handle())
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .sampler(texture.sampler)
            })
            .collect();

        let descriptor_write = vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_pool.get(frame_index))
            .dst_binding(4) // Texture buffer is index 4 in the scene descriptor set layout.
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&texture_image_infos);

        unsafe { self.loader.update_descriptor_sets(&[descriptor_write], &[]) };

        self.descriptors_dirty[frame_index] = false;
    }

    fn upload_textures(
        &mut self,
        device: &Device,
        textures_delta: &egui::TexturesDelta,
        frame_index: usize,
        transient_cmd_pool: &mut TransientPrimaryCommandPool,
    ) -> MemResult<()> {
        fn image_size(delta: &epaint::ImageDelta) -> usize {
            // TODO: If supported, fonts could use a 2-byte format to save memory:
            // https://github.com/hakolao/egui_winit_vulkano/blob/00bd40b491f397611fcf555e79134f7d2928fef1/src/renderer.rs#L309
            match &delta.image {
                egui::ImageData::Color(image) => image.width() * image.height() * 4,
                egui::ImageData::Font(font) => font.width() * font.height() * 4,
            }
        }
        fn image_extent(delta: &epaint::ImageDelta) -> vk::Extent3D {
            vk::Extent3D {
                width: delta.image.width() as u32,
                height: delta.image.height() as u32,
                depth: 1,
            }
        }

        // Free textures.
        for id in textures_delta.free.iter() {
            self.textures.swap_remove(id);
        }

        let mut total_bytes = 0;
        let offsets: Vec<_> = textures_delta
            .set
            .iter()
            .scan(&mut total_bytes, |total_bytes, (_, delta)| {
                let offset = **total_bytes;
                let image_size = image_size(delta);
                **total_bytes += image_size as vk::DeviceSize;
                Some(offset)
            })
            .collect();

        if total_bytes == 0 {
            self.write_texture_descriptors(frame_index);
            return Ok(());
        }
        self.descriptors_dirty[frame_index] = true;

        let mut upload_cmd_buf = transient_cmd_pool.allocate_transient_cmd_buffer()?;

        let mut staging_buffer = Buffer::<Staging>::new(
            debug_only_name!("GUI texture upload"),
            device,
            total_bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;

        let texture_format = vk::Format::R8G8B8A8_UNORM;

        let mut texture_layouts = HashMap::new();
        for (id, delta) in textures_delta.set.iter() {
            let mut image_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            let sampler = self.get_or_create_sampler(device, &delta.options)?;

            if delta.pos.is_some() {
                assert!(
                    self.textures.contains_key(id),
                    "GUI delta op applied to non-existent texture"
                );
            } else if let indexmap::map::Entry::Vacant(entry) = self.textures.entry(*id) {
                // Insert new texture.
                let info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(texture_format)
                    .extent(image_extent(delta))
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED);
                let subresource = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1);

                entry.insert(FontTexture {
                    image: Image::new(debug_only_name!("GUI texture {id:?}"), device, &info, subresource)?,
                    sampler,
                });

                image_layout = vk::ImageLayout::UNDEFINED;
            }
            texture_layouts.insert(*id, image_layout);
        }

        // Transition textures that are updated to transfer destination layout.
        //let images: Vec<_> = textures_delta.set.iter().map(|(id, _)| &self.textures[id]).collect();
        let barriers: Vec<_> = self
            .textures
            .iter_mut()
            .filter_map(|(id, texture)| {
                texture_layouts.get(id).map(|initial_layout| {
                    texture.image.layout_transition_barrier(
                        *initial_layout,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        Some(0),
                    )
                })
            })
            .collect();
        let dep_info = vk::DependencyInfoKHR::default().image_memory_barriers(&barriers);
        upload_cmd_buf.pipeline_barrier2(device, &dep_info);

        let mut font_vec = Vec::new();
        for ((id, delta), offset) in izip!(textures_delta.set.iter(), offsets) {
            let texture = self.textures.get_mut(id).unwrap();

            let colour_data = match &delta.image {
                egui::ImageData::Color(image) => &image.pixels,
                egui::ImageData::Font(font) => {
                    font_vec.clear();
                    font_vec.extend(font.srgba_pixels(None));
                    &font_vec
                }
            };
            staging_buffer.copy_slice_into_buffer(colour_data, offset as usize)?;

            let mut region = vk::BufferImageCopy::default()
                .buffer_offset(offset as vk::DeviceSize)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(image_extent(delta));

            if let Some(pos) = delta.pos {
                region.image_offset = vk::Offset3D {
                    x: pos[0] as i32,
                    y: pos[1] as i32,
                    z: 0,
                };
            }

            // TODO: Consolidate uploads to the same texture into a single copy command.
            upload_cmd_buf.copy_buffer_to_image(
                &staging_buffer,
                &mut texture.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        // Transition textures to shader read-only layout.
        let barriers: Vec<_> = self
            .textures
            .iter_mut()
            .filter_map(|(id, texture)| {
                texture_layouts.contains_key(id).then(|| {
                    texture.image.layout_transition_barrier(
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        Some(0),
                    )
                })
            })
            .collect();
        let dep_info = vk::DependencyInfoKHR::default().image_memory_barriers(&barriers);
        upload_cmd_buf.pipeline_barrier2(device, &dep_info);

        let exec = upload_cmd_buf.end()?;
        let pending = exec.submit(&[], &[])?;

        self.write_texture_descriptors(frame_index);

        pending.wait_for_fence()?;

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

impl Drop for GuiRenderer {
    fn drop(&mut self) {
        // Drop samplers.
        for sampler in self.samplers.values() {
            unsafe {
                self.loader.destroy_sampler(*sampler, None);
            }
        }
    }
}
