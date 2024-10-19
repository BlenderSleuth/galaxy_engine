// Copyright (c) 2024. Ben Sutherland

use std::marker::PhantomData;
use std::slice;

use arrayvec::ArrayVec;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::DependencyInfo;
use castaway::match_type;

use crate::vulkan::buffer::{Buffer, GpuOnly, MemLocation};
use crate::vulkan::device::{Device, DeviceExt, SharedDeviceLoader};
use crate::vulkan::pipeline::{ComputePipeline, GraphicsPipeline, Pipeline, PipelineLayout};
use crate::vulkan::queue::queue_type::{ComputeQueueType, PrimaryQueue, QueueType};
use crate::vulkan::queue::Queue;
use crate::vulkan::sync::{Fence, WaitSemaphore};

pub type PrimaryCommandPool<T> = CommandPool<PrimaryQueue, T>;
pub type ResettablePrimaryCommandPool = CommandPool<PrimaryQueue, Resettable<PrimaryQueue, OneTime>>;
pub type TransientPrimaryCommandPool = CommandPool<PrimaryQueue, Transient>;

pub trait CommandPoolType: Default {
    const FLAGS: vk::CommandPoolCreateFlags = vk::CommandPoolCreateFlags::empty();
}
#[derive(Default)]
pub struct Transient;
impl CommandPoolType for Transient {
    const FLAGS: vk::CommandPoolCreateFlags = vk::CommandPoolCreateFlags::TRANSIENT;
}
pub struct Resettable<Q: QueueType, S: SubmissionType> {
    persistent_cmd_buffers: Vec<PersistentCmdBuf<Q, S>>,
}

impl<Q: QueueType, S: SubmissionType> Default for Resettable<Q, S> {
    fn default() -> Self {
        Self {
            persistent_cmd_buffers: Vec::new(),
        }
    }
}

impl<Q: QueueType, S: SubmissionType> CommandPoolType for Resettable<Q, S> {}

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

impl<Q: QueueType> CommandPool<Q, Transient> {
    pub fn new_one_time(&mut self) -> VkResult<CommandBuffer<Q, Recording<OneTime>>> {
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

impl<Q: QueueType, S: SubmissionType> CommandPool<Q, Resettable<Q, S>> {
    pub fn allocate_cmd_buffer(&mut self, level: vk::CommandBufferLevel) -> VkResult<&mut PersistentCmdBuf<Q, S>> {
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
    ) -> VkResult<&mut [PersistentCmdBuf<Q, S>]> {
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

    pub fn get_cmd_buffers(&mut self) -> &mut [PersistentCmdBuf<Q, S>] {
        &mut self.pool_storage.persistent_cmd_buffers
    }

    pub fn get_cmd_buffer(&mut self, idx: usize) -> &mut PersistentCmdBuf<Q, S> {
        &mut self.pool_storage.persistent_cmd_buffers[idx]
    }

    pub fn reset(&mut self) -> Result<(), PersistentCmdBufError> {
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
    use ash::vk;

    pub trait CmdBufState: 'static {}
    pub trait ResettableState: CmdBufState {}
    pub trait CompletedState: ResettableState {}

    // Some states differ if the command buffer is one-time-submit or persistent.
    pub trait SubmissionType: 'static {
        const FLAGS: vk::CommandBufferUsageFlags;
        type CompletedState: CompletedState;
    }
    pub struct OneTime;
    impl SubmissionType for OneTime {
        const FLAGS: vk::CommandBufferUsageFlags = vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT;
        type CompletedState = Invalid;
    }
    pub struct Persistent;
    impl SubmissionType for Persistent {
        const FLAGS: vk::CommandBufferUsageFlags = vk::CommandBufferUsageFlags::empty();
        type CompletedState = Executable<Persistent>;
    }

    // Initial state.
    pub struct Initial;
    impl CmdBufState for Initial {}
    impl ResettableState for Initial {}

    // Recording state.
    pub struct Recording<S: SubmissionType>(std::marker::PhantomData<S>);
    impl<S: SubmissionType> CmdBufState for Recording<S> {}
    impl<S: SubmissionType> ResettableState for Recording<S> {}

    // Executable state.
    pub struct Executable<S: SubmissionType>(std::marker::PhantomData<S>);
    impl<S: SubmissionType> CmdBufState for Executable<S> {}
    impl<S: SubmissionType> ResettableState for Executable<S> {}
    impl CompletedState for Executable<Persistent> {}

    // Pending state.
    // Command buffer cannot be reset when in the pending state.
    pub trait PendingState: CmdBufState {
        type CompletedState: CompletedState;
    }
    pub struct Pending<S: SubmissionType>(std::marker::PhantomData<S>);
    impl<S: SubmissionType> CmdBufState for Pending<S> {}
    impl<S: SubmissionType> PendingState for Pending<S> {
        type CompletedState = S::CompletedState;
    }

    // Invalid state.
    pub struct Invalid;
    impl CmdBufState for Invalid {}
    impl ResettableState for Invalid {}
    impl CompletedState for Invalid {}
}
// Public exports
pub use command_buffer_states::SubmissionType;
use command_buffer_states::*;

use crate::vulkan::extensions::DeviceExtensions;
use crate::vulkan::image::Image;

// Command buffer state types.
pub type InitialCmdBuf<Q = PrimaryQueue> = CommandBuffer<Q, Initial>;
pub type RecordingCmdBuf<Q = PrimaryQueue, S = OneTime> = CommandBuffer<Q, Recording<S>>;
pub type ExecutableCmdBuf<Q = PrimaryQueue, S = OneTime> = CommandBuffer<Q, Executable<S>>;
pub type PendingCmdBuf<Q = PrimaryQueue, S = OneTime> = CommandBuffer<Q, Pending<S>>;
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
    pub fn begin<S: SubmissionType>(self) -> VkResult<CommandBuffer<Q, Recording<S>>> {
        let begin_info = vk::CommandBufferBeginInfo::default().flags(S::FLAGS);
        unsafe { self.loader.begin_command_buffer(self.handle, &begin_info) }?;
        Ok(self.next_state())
    }
}

impl<S: SubmissionType> CommandBuffer<PrimaryQueue, Recording<S>> {
    pub fn bind_graphics_pipeline(&self, pipeline: &GraphicsPipeline) {
        unsafe {
            self.loader
                .cmd_bind_pipeline(self.handle, vk::PipelineBindPoint::GRAPHICS, pipeline.handle())
        }
    }

    // TODO: Another state for rendering?
    pub fn begin_rendering(&mut self, ext: &DeviceExtensions, rendering_info: &vk::RenderingInfo) {
        unsafe { ext.dyn_cmd.cmd_begin_rendering(self.handle, rendering_info) };
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

    pub fn end_rendering(&mut self, ext: &DeviceExtensions) {
        unsafe { ext.dyn_cmd.cmd_end_rendering(self.handle) };
    }
}

impl<Q: ComputeQueueType, S: SubmissionType> CommandBuffer<Q, Recording<S>> {
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

    pub fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            self.loader
                .cmd_dispatch(self.handle, group_count_x, group_count_y, group_count_z)
        };
    }
}

impl<Q: QueueType, S: SubmissionType> CommandBuffer<Q, Recording<S>> {
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

    pub fn end(self) -> VkResult<CommandBuffer<Q, Executable<S>>> {
        unsafe { self.loader.end_command_buffer(self.handle) }?;
        Ok(Self::next_state(self))
    }
}

impl<Q: QueueType> CommandBuffer<Q, Recording<OneTime>> {
    // On transient buffers ending, submitting, waiting and freeing are often all done in one go.
    pub fn end_submit_wait_and_free(self) -> VkResult<()> {
        let ended = self.end()?;
        let pending = ended.submit(&[], &[], None)?;
        let completed = pending.queue_wait_idle()?;
        completed.free();
        Ok(())
    }
}

impl<Q: QueueType, S: SubmissionType> CommandBuffer<Q, Executable<S>> {
    pub fn submit<const M: usize>(
        self,
        wait_semaphores: &[WaitSemaphore; M],
        signal_semaphores: &[vk::Semaphore],
        fence: Option<&Fence>,
    ) -> VkResult<CommandBuffer<Q, Pending<S>>> {
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

impl<Q: QueueType, C: PendingState> CommandBuffer<Q, C> {
    pub fn queue_wait_idle(self) -> VkResult<CommandBuffer<Q, C::CompletedState>> {
        unsafe { self.loader.queue_wait_idle(self.queue) }?;
        Ok(Self::next_state(self))
    }
}

// Persistent command buffers last more than a frame.
#[derive(thiserror::Error, Debug)]
pub enum PersistentCmdBufError {
    #[error("Command buffer is in the wrong state ({0}) for operation {1}")]
    WrongState(&'static str, &'static str),
    #[error("Vulkan error while transitioning command buffer state: {0}")]
    VulkanError(#[from] vk::Result),
}

pub enum PersistentCmdBuf<Q: QueueType, S: SubmissionType> {
    Invalid(CommandBuffer<Q, Invalid>),
    Initial(CommandBuffer<Q, Initial>),
    Recording(CommandBuffer<Q, Recording<S>>),
    Executable(CommandBuffer<Q, Executable<S>>),
    Pending(CommandBuffer<Q, Pending<S>>),
    // Only transitioning within a method call.
    // Used for transitioning between states (so buffer can be moved in and out).
    Transitioning,
}

impl<Q: QueueType, S: SubmissionType + 'static> PersistentCmdBuf<Q, S> {
    fn new<C: CmdBufState>(cmd_buf: CommandBuffer<Q, C>) -> Self {
        match_type!(cmd_buf, {
            CommandBuffer<Q, Initial> as cmd_buf => Self::Initial(cmd_buf),
            CommandBuffer<Q, Recording<S>> as cmd_buf => Self::Recording(cmd_buf),
            CommandBuffer<Q, Executable<S>> as cmd_buf => Self::Executable(cmd_buf),
            CommandBuffer<Q, Pending<S>> as cmd_buf => Self::Pending(cmd_buf),
            CommandBuffer<Q, Invalid> as cmd_buf => Self::Invalid(cmd_buf),
            _ => unreachable!(),
        })
    }

    fn check_not_transitioning(&self) {
        assert!(
            !matches!(self, Self::Transitioning),
            "Command buffer is already transitioning."
        );
    }

    const fn state_str(&self) -> &'static str {
        match self {
            Self::Initial(_) => "Initial",
            Self::Recording(_) => "Recording",
            Self::Executable(_) => "Executable",
            Self::Pending(_) => "Pending",
            Self::Invalid(_) => "Invalid",
            Self::Transitioning => "Transitioning",
        }
    }

    // Resets back to initial state.
    fn reset(&mut self) -> Result<(), PersistentCmdBufError> {
        self.check_not_transitioning();

        // TODO: Manage buffer fences in pool, so pending buffers can be reset.
        //if let Self::Pending(_) = self {
        //    return Err(PersistentCmdBufError::WrongState(self.state_str(), "reset command buffer"));
        //}

        // TODO: Check if this whole block is close to a no-op.
        *self = Self::Initial(match std::mem::replace(self, Self::Transitioning) {
            Self::Initial(cmd_buf) => cmd_buf.reset(),
            Self::Recording(cmd_buf) => cmd_buf.reset(),
            Self::Executable(cmd_buf) => cmd_buf.reset(),
            //Self::Pending(_) => unreachable!(),
            Self::Pending(cmd_buf) => cmd_buf.queue_wait_idle()?.reset(),
            Self::Invalid(cmd_buf) => cmd_buf.reset(),
            Self::Transitioning => unreachable!(),
        });

        Ok(())
    }

    pub fn begin(&mut self) -> Result<&mut CommandBuffer<Q, Recording<S>>, PersistentCmdBufError> {
        self.check_not_transitioning();

        if let Self::Initial(cmd_buf) = std::mem::replace(self, Self::Transitioning) {
            *self = Self::Recording(cmd_buf.begin::<S>()?);
            let Self::Recording(cmd_buf) = self else { unreachable!() };
            Ok(cmd_buf)
        } else {
            Err(PersistentCmdBufError::WrongState(
                self.state_str(),
                "begin command buffer",
            ))
        }
    }

    // Because of Rust's mutability rules, the recording buffer returned from begin() cannot be used after this method.
    pub fn end(&mut self) -> Result<(), PersistentCmdBufError> {
        self.check_not_transitioning();

        if let Self::Recording(cmd_buf) = std::mem::replace(self, Self::Transitioning) {
            *self = Self::Executable(cmd_buf.end()?);
            Ok(())
        } else {
            Err(PersistentCmdBufError::WrongState(
                self.state_str(),
                "end command buffer",
            ))
        }
    }

    pub fn submit<const M: usize>(
        &mut self,
        wait_semaphores: &[WaitSemaphore; M],
        signal_semaphores: &[vk::Semaphore],
        fence: Option<&Fence>,
    ) -> Result<(), PersistentCmdBufError> {
        self.check_not_transitioning();

        if let Self::Executable(cmd_buf) = std::mem::replace(self, Self::Transitioning) {
            *self = Self::Pending(cmd_buf.submit(wait_semaphores, signal_semaphores, fence)?);
            Ok(())
        } else {
            Err(PersistentCmdBufError::WrongState(
                self.state_str(),
                "submit command buffer",
            ))
        }
    }
}

//impl Drop for CommandBuffer<Q, C> {
//    fn drop(&mut self) {
//        self.free()
//    }
//}
