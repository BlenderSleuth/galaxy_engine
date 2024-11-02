// Copyright (c) 2024 Ben Sutherland.

use std::marker::PhantomData;
use std::slice;

use arrayvec::ArrayVec;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::DependencyInfo;
use castaway::match_type;
// Public exports
pub use command_buffer_states::RenderingState;

use crate::vulkan::buffer::{Buffer, GpuOnly, MemLocation};
use crate::vulkan::device::{Device, DeviceExt, SharedDeviceLoader};
use crate::vulkan::extensions::DeviceExtensions;
use crate::vulkan::image::Image;
use crate::vulkan::pipeline::{ComputePipeline, GraphicsPipeline, Pipeline, PipelineLayout};
use crate::vulkan::queue::queue_type::{ComputeQueueType, PrimaryQueue, QueueType};
use crate::vulkan::queue::Queue;
use crate::vulkan::sync::{Fence, WaitSemaphore};

pub type PrimaryCommandPool<T> = CommandPool<PrimaryQueue, T>;
pub type ResettablePrimaryCommandPool = CommandPool<PrimaryQueue, Resettable<PrimaryQueue>>;
pub type TransientPrimaryCommandPool = CommandPool<PrimaryQueue, Transient>;

pub trait CommandPoolType: Default {
    const FLAGS: vk::CommandPoolCreateFlags = vk::CommandPoolCreateFlags::empty();
}

#[derive(Default)]
pub struct Transient;
impl CommandPoolType for Transient {
    const FLAGS: vk::CommandPoolCreateFlags = vk::CommandPoolCreateFlags::TRANSIENT;
}

pub struct Resettable<Q: QueueType> {
    persistent_cmd_buffers: Vec<PersistentCmdBuf<Q>>,
}

impl<Q: QueueType> Default for Resettable<Q> {
    fn default() -> Self {
        Self {
            persistent_cmd_buffers: Vec::new(),
        }
    }
}

impl<Q: QueueType> CommandPoolType for Resettable<Q> {}

pub struct CommandPool<Q: QueueType, T: CommandPoolType> {
    loader: SharedDeviceLoader,
    handle: vk::CommandPool,
    queue: vk::Queue,
    queue_type: PhantomData<Q>,
    pool_storage: T,
}

impl<Q: QueueType, T: CommandPoolType> CommandPool<Q, T> {
    pub fn new(device: &Device, queue: &Queue<Q>) -> VkResult<Self> {
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(T::FLAGS)
            .queue_family_index(queue.family_index());
        let handle = unsafe { device.loader().create_command_pool(&command_pool_info, None) }?;

        Ok(Self {
            loader: device.cloned_loader(),
            handle,
            queue: queue.handle(),
            queue_type: PhantomData,
            pool_storage: T::default(),
        })
    }
}

// Transient command pools give away their buffers immediately, to be freed later.
impl<Q: QueueType> CommandPool<Q, Transient> {
    pub fn allocate_transient_cmd_buffer(&mut self) -> VkResult<CommandBuffer<Q, Recording<OutsideRenderPass>>> {
        let handle = unsafe {
            self.loader
                .allocate_command_buffer(self.handle, vk::CommandBufferLevel::PRIMARY)
        }?;

        let cmd_buffer = CommandBuffer {
            loader: self.loader.clone(),
            handle,
            pool: self.handle,
            queue: self.queue,
            queue_type: PhantomData,
            state: PhantomData,
        };

        cmd_buffer.begin()
    }
}

// Resettable command pools own their buffers and reset them all at once.
impl<Q: QueueType> CommandPool<Q, Resettable<Q>> {
    pub fn allocate_cmd_buffer(&mut self, level: vk::CommandBufferLevel) -> VkResult<&mut PersistentCmdBuf<Q>> {
        let handle = unsafe { self.loader.allocate_command_buffer(self.handle, level) }?;

        self.pool_storage
            .persistent_cmd_buffers
            .push(PersistentCmdBuf::new(CommandBuffer {
                loader: self.loader.clone(),
                handle,
                pool: self.handle,
                queue: self.queue,
                queue_type: PhantomData::<Q>,
                state: PhantomData::<Initial>,
            }));

        Ok(self.pool_storage.persistent_cmd_buffers.last_mut().unwrap())
    }

    pub fn allocate_cmd_buffers<const N: usize>(
        &mut self,
        level: vk::CommandBufferLevel,
    ) -> VkResult<&mut [PersistentCmdBuf<Q>]> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.handle)
            .level(level)
            .command_buffer_count(N as u32);
        let handles = unsafe { self.loader.allocate_command_buffers_av::<N>(&allocate_info) }?;

        self.pool_storage
            .persistent_cmd_buffers
            .extend(handles.iter().map(|&handle| {
                PersistentCmdBuf::new(CommandBuffer {
                    loader: self.loader.clone(),
                    handle,
                    pool: self.handle,
                    queue: self.queue,
                    queue_type: PhantomData::<Q>,
                    state: PhantomData::<Initial>,
                })
            }));

        let range = (self.pool_storage.persistent_cmd_buffers.len() - N)..;
        Ok(&mut self.pool_storage.persistent_cmd_buffers[range])
    }

    pub fn get_cmd_buffers(&mut self) -> &mut [PersistentCmdBuf<Q>] {
        &mut self.pool_storage.persistent_cmd_buffers
    }

    pub fn get_cmd_buffer(&mut self, idx: usize) -> &mut PersistentCmdBuf<Q> {
        &mut self.pool_storage.persistent_cmd_buffers[idx]
    }

    pub fn reset(&mut self) -> Result<(), CmdBufStateTransitionError> {
        for cmd_buf in self.pool_storage.persistent_cmd_buffers.iter_mut() {
            cmd_buf.reset()?;
        }

        // Reset the pool resets all allocated command buffers.
        unsafe {
            self.loader
                .reset_command_pool(self.handle, vk::CommandPoolResetFlags::empty())
        }?;

        Ok(())
    }
}

impl<Q: QueueType, T: CommandPoolType> Drop for CommandPool<Q, T> {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_command_pool(self.handle, None) };
    }
}

mod command_buffer_states {
    pub trait CmdBufState: 'static {}
    pub trait ResettableState: CmdBufState {}

    // Rendering state (with a render pass).
    pub trait RenderingState: 'static {}
    pub struct InsideRenderPass;
    impl RenderingState for InsideRenderPass {}
    pub struct OutsideRenderPass;
    impl RenderingState for OutsideRenderPass {}

    // Initial state.
    pub struct Initial;
    impl CmdBufState for Initial {}
    impl ResettableState for Initial {}

    // Recording state.
    pub struct Recording<R: RenderingState>(std::marker::PhantomData<R>);
    impl<R: RenderingState> CmdBufState for Recording<R> {}
    impl<R: RenderingState> ResettableState for Recording<R> {}

    // Executable state.
    pub struct Executable();
    impl CmdBufState for Executable {}
    impl ResettableState for Executable {}

    // Pending state.
    // Command buffer cannot be reset when in the pending state.
    pub trait PendingState: CmdBufState {}
    pub struct Pending;
    impl CmdBufState for Pending {}
    impl PendingState for Pending {}

    // Invalid state.
    pub struct Invalid;
    impl CmdBufState for Invalid {}
    impl ResettableState for Invalid {}
}
use command_buffer_states::*;

// Command buffer state types.
pub type InitialCmdBuf<Q = PrimaryQueue> = CommandBuffer<Q, Initial>;
pub type RecordingCmdBuf<Q = PrimaryQueue, R = OutsideRenderPass> = CommandBuffer<Q, Recording<R>>;
pub type RenderingCmdBuf<Q = PrimaryQueue> = CommandBuffer<Q, Recording<InsideRenderPass>>;
pub type ExecutableCmdBuf<Q = PrimaryQueue> = CommandBuffer<Q, Executable>;
pub type PendingCmdBuf<Q = PrimaryQueue> = CommandBuffer<Q, Pending>;
pub type InvalidCmdBuf<Q = PrimaryQueue> = CommandBuffer<Q, Invalid>;

pub struct CommandBuffer<Q: QueueType, C: CmdBufState> {
    loader: SharedDeviceLoader,
    handle: vk::CommandBuffer,
    pool: vk::CommandPool,
    queue: vk::Queue,
    queue_type: PhantomData<Q>,
    state: PhantomData<C>,
}

impl<Q: QueueType, C: CmdBufState> CommandBuffer<Q, C> {
    #[deprecated = "This should only be used until the relevant commands have been moved to the cmd buffer API."]
    pub fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    // TODO: Is this a no-op function?
    fn next_state<T: CmdBufState>(self) -> CommandBuffer<Q, T> {
        CommandBuffer {
            loader: self.loader,
            handle: self.handle,
            pool: self.pool,
            queue: self.queue,
            queue_type: PhantomData,
            state: PhantomData,
        }
    }

    // Should only be used internally.
    fn free(self) {
        unsafe { self.loader.free_command_buffers(self.pool, &[self.handle]) };
    }
}

impl<Q: QueueType, C: ResettableState> CommandBuffer<Q, C> {
    // This is for internal state-management use only.
    // Buffers are either persistent and reset by the command pool, or transient and immediately freed.
    fn reset(self) -> CommandBuffer<Q, Initial> {
        self.next_state()
    }
}

impl<Q: QueueType> CommandBuffer<Q, Initial> {
    pub fn begin(self) -> VkResult<CommandBuffer<Q, Recording<OutsideRenderPass>>> {
        // Always one-time submit (don't bother reusing command buffers).
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.loader.begin_command_buffer(self.handle, &begin_info) }?;
        Ok(self.next_state())
    }
}

impl CommandBuffer<PrimaryQueue, Recording<OutsideRenderPass>> {
    pub fn begin_rendering(
        self,
        ext: &DeviceExtensions,
        rendering_info: &vk::RenderingInfo,
    ) -> RecordingCmdBuf<PrimaryQueue, InsideRenderPass> {
        unsafe { ext.dyn_cmd.cmd_begin_rendering(self.handle, rendering_info) };
        self.next_state()
    }
}

// Graphics recording commands.
impl<R: RenderingState> CommandBuffer<PrimaryQueue, Recording<R>> {
    pub fn bind_graphics_pipeline(&self, pipeline: &GraphicsPipeline) {
        unsafe {
            self.loader
                .cmd_bind_pipeline(self.handle, vk::PipelineBindPoint::GRAPHICS, pipeline.handle())
        }
    }

    pub fn bind_index_buffer(&mut self, buffer: &Buffer<GpuOnly>, offset: vk::DeviceSize, index_type: vk::IndexType) {
        unsafe {
            self.loader
                .cmd_bind_index_buffer(self.handle, buffer.handle(), offset, index_type)
        };
    }

    pub fn bind_vertex_buffer(&mut self, buffer: &Buffer<GpuOnly>, vertices_offset: vk::DeviceSize) {
        unsafe {
            self.loader
                .cmd_bind_vertex_buffers(self.handle, 0, &[buffer.handle()], &[vertices_offset])
        };
    }

    pub fn set_viewport(&mut self, viewport: vk::Viewport) {
        unsafe { self.loader.cmd_set_viewport(self.handle, 0, &[viewport]) };
    }

    pub fn set_scissor(&mut self, scissor: vk::Rect2D) {
        unsafe { self.loader.cmd_set_scissor(self.handle, 0, &[scissor]) };
    }
}

// Rendering commands.
impl CommandBuffer<PrimaryQueue, Recording<InsideRenderPass>> {
    // Draw commands are only valid inside a render pass.
    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            self.loader.cmd_draw_indexed(
                self.handle,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        };
    }

    pub fn end_rendering(self, ext: &DeviceExtensions) -> RecordingCmdBuf<PrimaryQueue, OutsideRenderPass> {
        unsafe { ext.dyn_cmd.cmd_end_rendering(self.handle) };
        self.next_state()
    }
}

// Graphics/compute recording commands.
impl<Q: ComputeQueueType, R: RenderingState> CommandBuffer<Q, Recording<R>> {
    pub fn bind_compute_pipeline(&self, pipeline: &ComputePipeline) {
        unsafe {
            self.loader
                .cmd_bind_pipeline(self.handle, vk::PipelineBindPoint::COMPUTE, pipeline.handle())
        }
    }

    pub fn push_constants(
        &mut self,
        pipeline_layout: &PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        data: &[u8],
    ) {
        unsafe {
            self.loader
                .cmd_push_constants(self.handle, pipeline_layout.handle(), stage_flags, offset, data)
        };
    }

    pub fn bind_descriptor_sets(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        unsafe {
            self.loader.cmd_bind_descriptor_sets(
                self.handle,
                bind_point,
                pipeline_layout.handle(),
                first_set,
                descriptor_sets,
                dynamic_offsets,
            )
        };
    }
}

// Compute dispatches.
impl<Q: ComputeQueueType> CommandBuffer<Q, Recording<OutsideRenderPass>> {
    // Dispatch must be called when not rendering.
    pub fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            self.loader
                .cmd_dispatch(self.handle, group_count_x, group_count_y, group_count_z)
        };
    }
}

// Generic recording commands.
impl<Q: QueueType, R: RenderingState> CommandBuffer<Q, Recording<R>> {
    pub fn copy_buffer<L1: MemLocation, L2: MemLocation>(
        &mut self,
        src_buffer: &Buffer<L1>,
        dst_buffer: &mut Buffer<L2>,
        copy_regions: &[vk::BufferCopy],
    ) {
        unsafe {
            self.loader
                .cmd_copy_buffer(self.handle, src_buffer.handle(), dst_buffer.handle(), copy_regions)
        };
    }

    pub fn copy_buffer_to_image<L: MemLocation>(
        &mut self,
        src_buffer: &Buffer<L>,
        dst_image: &mut Image,
        dst_image_layout: vk::ImageLayout,
        copy_regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.loader.cmd_copy_buffer_to_image(
                self.handle,
                src_buffer.handle(),
                dst_image.handle(),
                dst_image_layout,
                copy_regions,
            )
        };
    }

    pub fn pipeline_barrier2(&mut self, ext: &DeviceExtensions, dependency_info: &DependencyInfo) {
        unsafe { ext.sync2.cmd_pipeline_barrier2(self.handle, dependency_info) };
    }
}

impl<Q: QueueType> CommandBuffer<Q, Recording<OutsideRenderPass>> {
    pub fn end(self) -> VkResult<CommandBuffer<Q, Executable>> {
        unsafe { self.loader.end_command_buffer(self.handle) }?;
        Ok(Self::next_state(self))
    }

    // On transient buffers ending, submitting, waiting and freeing are often all done in one go.
    pub fn end_submit_wait_and_free(self) -> VkResult<()> {
        let ended = self.end()?;
        let pending = ended.submit(&[], &[], None)?;
        let completed = pending.queue_wait_idle()?;
        completed.free();
        Ok(())
    }
}

// Executable commands.
impl<Q: QueueType> CommandBuffer<Q, Executable> {
    pub fn submit<const M: usize>(
        self,
        wait_semaphores: &[WaitSemaphore; M],
        signal_semaphores: &[vk::Semaphore],
        fence: Option<&Fence>,
    ) -> VkResult<CommandBuffer<Q, Pending>> {
        let semaphore_handles: ArrayVec<_, M> = wait_semaphores.iter().map(|sem| sem.handle).collect();
        let semaphore_stages: ArrayVec<_, M> = wait_semaphores.iter().map(|sem| sem.stage_mask).collect();

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&semaphore_handles)
            .wait_dst_stage_mask(&semaphore_stages)
            .signal_semaphores(signal_semaphores)
            .command_buffers(slice::from_ref(&self.handle));

        let fence = fence.map(|f| f.handle()).unwrap_or(vk::Fence::null());

        unsafe { self.loader.queue_submit(self.queue, &[submit_info], fence) }?;

        Ok(self.next_state())
    }
}

// Pending commands.
impl<Q: QueueType, C: PendingState> CommandBuffer<Q, C> {
    pub fn queue_wait_idle(self) -> VkResult<CommandBuffer<Q, Invalid>> {
        unsafe { self.loader.queue_wait_idle(self.queue) }?;
        Ok(Self::next_state(self))
    }
}

// Persistent command buffers last more than a frame, and are state-tracked in an enum.
#[derive(thiserror::Error, Debug)]
pub enum CmdBufStateTransitionError {
    #[error("Command buffer is in the wrong state ({0}) for operation {1}")]
    WrongState(&'static str, &'static str),
    #[error("Vulkan error while transitioning command buffer state: {0}")]
    VulkanError(#[from] vk::Result),
}
type CmdBufStateTransitionResult<T> = Result<T, CmdBufStateTransitionError>;

pub enum PersistentCmdBuf<Q: QueueType> {
    Invalid(CommandBuffer<Q, Invalid>),
    Initial(CommandBuffer<Q, Initial>),
    Recording(CommandBuffer<Q, Recording<OutsideRenderPass>>),
    Rendering(CommandBuffer<Q, Recording<InsideRenderPass>>),
    Executable(CommandBuffer<Q, Executable>),
    Pending(CommandBuffer<Q, Pending>),
    // Only transitioning within a method call.
    // Used for transitioning between states (so buffer can be moved in and out).
    Transitioning,
}

impl<Q: QueueType> PersistentCmdBuf<Q> {
    fn new<C: CmdBufState>(cmd_buf: CommandBuffer<Q, C>) -> Self {
        match_type!(cmd_buf, {
            CommandBuffer<Q, Initial> as cmd_buf => Self::Initial(cmd_buf),
            CommandBuffer<Q, Recording<OutsideRenderPass>> as cmd_buf => Self::Recording(cmd_buf),
            CommandBuffer<Q, Recording<InsideRenderPass>> as cmd_buf => Self::Rendering(cmd_buf),
            CommandBuffer<Q, Executable> as cmd_buf => Self::Executable(cmd_buf),
            CommandBuffer<Q, Pending> as cmd_buf => Self::Pending(cmd_buf),
            CommandBuffer<Q, Invalid> as cmd_buf => Self::Invalid(cmd_buf),
            _ => unreachable!(),
        })
    }

    fn check_not_transitioning(&self) {
        debug_assert!(
            !matches!(self, Self::Transitioning),
            "Command buffer is already transitioning."
        );
    }

    const fn state_str(&self) -> &'static str {
        match self {
            Self::Initial(_) => "Initial",
            Self::Recording(_) => "Recording",
            Self::Rendering(_) => "Rendering",
            Self::Executable(_) => "Executable",
            Self::Pending(_) => "Pending",
            Self::Invalid(_) => "Invalid",
            Self::Transitioning => "Transitioning",
        }
    }

    // Resets back to initial state.
    fn reset(&mut self) -> Result<(), CmdBufStateTransitionError> {
        self.check_not_transitioning();

        // TODO: Manage buffer fences in pool, so pending buffers can be reset.
        //if let Self::Pending(_) = self {
        //    return Err(PersistentCmdBufError::WrongState(self.state_str(), "reset command buffer"));
        //}

        // TODO: Check if this whole block is close to a no-op.
        *self = Self::Initial(match std::mem::replace(self, Self::Transitioning) {
            Self::Initial(cmd_buf) => cmd_buf.reset(),
            Self::Recording(cmd_buf) => cmd_buf.reset(),
            Self::Rendering(cmd_buf) => cmd_buf.reset(),
            Self::Executable(cmd_buf) => cmd_buf.reset(),
            //Self::Pending(_) => unreachable!(),
            Self::Pending(cmd_buf) => cmd_buf.queue_wait_idle()?.reset(),
            Self::Invalid(cmd_buf) => cmd_buf.reset(),
            Self::Transitioning => unreachable!(),
        });

        Ok(())
    }

    pub fn begin(&mut self) -> CmdBufStateTransitionResult<&mut CommandBuffer<Q, Recording<OutsideRenderPass>>> {
        self.check_not_transitioning();

        if let Self::Initial(cmd_buf) = std::mem::replace(self, Self::Transitioning) {
            *self = Self::Recording(cmd_buf.begin()?);
            let Self::Recording(cmd_buf) = self else { unreachable!() };
            Ok(cmd_buf)
        } else {
            Err(CmdBufStateTransitionError::WrongState(self.state_str(), "begin"))
        }
    }

    // Because of Rust's mutability rules, the recording buffer returned from begin() cannot be used after this method.
    pub fn end(&mut self) -> CmdBufStateTransitionResult<()> {
        self.check_not_transitioning();

        if let Self::Recording(cmd_buf) = std::mem::replace(self, Self::Transitioning) {
            *self = Self::Executable(cmd_buf.end()?);
            Ok(())
        } else {
            Err(CmdBufStateTransitionError::WrongState(self.state_str(), "end"))
        }
    }

    pub fn submit<const M: usize>(
        &mut self,
        wait_semaphores: &[WaitSemaphore; M],
        signal_semaphores: &[vk::Semaphore],
        fence: Option<&Fence>,
    ) -> CmdBufStateTransitionResult<()> {
        self.check_not_transitioning();

        if let Self::Executable(cmd_buf) = std::mem::replace(self, Self::Transitioning) {
            *self = Self::Pending(cmd_buf.submit(wait_semaphores, signal_semaphores, fence)?);
            Ok(())
        } else {
            Err(CmdBufStateTransitionError::WrongState(self.state_str(), "submit"))
        }
    }
}

impl PersistentCmdBuf<PrimaryQueue> {
    pub fn begin_rendering(
        &mut self,
        ext: &DeviceExtensions,
        render_info: &vk::RenderingInfo,
    ) -> CmdBufStateTransitionResult<&mut CommandBuffer<PrimaryQueue, Recording<InsideRenderPass>>> {
        self.check_not_transitioning();

        if let Self::Recording(cmd_buf) = std::mem::replace(self, Self::Transitioning) {
            *self = Self::Rendering(cmd_buf.begin_rendering(ext, render_info));
            let Self::Rendering(cmd_buf) = self else { unreachable!() };
            Ok(cmd_buf)
        } else {
            Err(CmdBufStateTransitionError::WrongState(
                self.state_str(),
                "begin rendering",
            ))
        }
    }

    pub fn end_rendering(
        &mut self,
        ext: &DeviceExtensions,
    ) -> CmdBufStateTransitionResult<&mut CommandBuffer<PrimaryQueue, Recording<OutsideRenderPass>>> {
        self.check_not_transitioning();

        if let Self::Rendering(cmd_buf) = std::mem::replace(self, Self::Transitioning) {
            *self = Self::Recording(cmd_buf.end_rendering(ext));
            let Self::Recording(cmd_buf) = self else { unreachable!() };
            Ok(cmd_buf)
        } else {
            Err(CmdBufStateTransitionError::WrongState(
                self.state_str(),
                "begin rendering",
            ))
        }
    }
}

//impl Drop for CommandBuffer<Q, C> {
//    fn drop(&mut self) {
//        self.free()
//    }
//}
