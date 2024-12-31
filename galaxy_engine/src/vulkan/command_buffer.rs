// Copyright (c) 2024-2025 Ben Sutherland.

use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::slice;

use arrayvec::ArrayVec;
use ash::prelude::VkResult;
use ash::vk;
use castaway::cast;

use crate::pipelines::{ComputePipeline, GraphicsPipeline, Pipeline};
use crate::vulkan::buffer::{Buffer, GpuOnly, MemLocation};
use crate::vulkan::device::{Device, DeviceExt, SharedDeviceLoader};
use crate::vulkan::extensions::DeviceExtensions;
use crate::vulkan::image::Image;
use crate::vulkan::queue::queue_type::{ComputeQueueType, PrimaryQueue, QueueType};
use crate::vulkan::queue::Queue;
use crate::vulkan::sync::{Fence, WaitSemaphore};

pub type PrimaryCommandPool<T> = CommandPool<PrimaryQueue, T>;
pub type ResettablePrimaryCommandPool<const N: usize> = CommandPool<PrimaryQueue, Resettable<PrimaryQueue, N>>;
pub type TransientPrimaryCommandPool = CommandPool<PrimaryQueue, Transient>;

pub trait CommandPoolType: Default + 'static {
    const FLAGS: vk::CommandPoolCreateFlags = vk::CommandPoolCreateFlags::empty();
}

#[derive(Default)]
pub struct Transient;
impl CommandPoolType for Transient {
    const FLAGS: vk::CommandPoolCreateFlags = vk::CommandPoolCreateFlags::TRANSIENT;
}

pub struct Resettable<Q: QueueType, const N: usize> {
    persistent_cmd_buffers: ArrayVec<PersistentCmdBuf<Q>, N>,
}

impl<Q: QueueType, const N: usize> Default for Resettable<Q, N> {
    fn default() -> Self {
        Self {
            persistent_cmd_buffers: ArrayVec::new(),
        }
    }
}

impl<Q: QueueType, const N: usize> CommandPoolType for Resettable<Q, N> {}

pub struct CommandPool<Q: QueueType, T: CommandPoolType> {
    loader: SharedDeviceLoader,
    handle: vk::CommandPool,
    queue: vk::Queue,
    queue_type: PhantomData<Q>,
    pool_storage: ManuallyDrop<T>,
}

impl<Q: QueueType, T: CommandPoolType> CommandPool<Q, T> {
    pub fn new(name: &str, device: &Device, queue: &Queue<Q>) -> VkResult<Self> {
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(T::FLAGS)
            .queue_family_index(queue.family_index());
        let handle = unsafe { device.loader().create_command_pool(&command_pool_info, None) }?;

        crate::vulkan::debug::set_object_name(device, handle, name)?;

        Ok(Self {
            loader: device.cloned_loader(),
            handle,
            queue: queue.handle(),
            queue_type: PhantomData,
            pool_storage: Default::default(),
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

        let cmd_buffer = CommandBuffer::new(self.loader.clone(), handle, self.handle, self.queue)?;
        cmd_buffer.begin()
    }
}

// Resettable command pools own their buffers and reset them all at once.
impl<Q: QueueType, const N: usize> CommandPool<Q, Resettable<Q, N>> {
    pub fn allocate_cmd_buffer(&mut self, level: vk::CommandBufferLevel) -> VkResult<&mut PersistentCmdBuf<Q>> {
        assert!(self.pool_storage.persistent_cmd_buffers.len() < N);

        let handle = unsafe { self.loader.allocate_command_buffer(self.handle, level) }?;

        self.pool_storage
            .persistent_cmd_buffers
            .push(PersistentCmdBuf::new(CommandBuffer::new(
                self.loader.clone(),
                handle,
                self.handle,
                self.queue,
            )?));

        Ok(self.pool_storage.persistent_cmd_buffers.last_mut().unwrap())
    }

    pub fn allocate_cmd_buffers<const M: usize>(
        &mut self,
        level: vk::CommandBufferLevel,
    ) -> VkResult<&mut [PersistentCmdBuf<Q>]> {
        assert!(self.pool_storage.persistent_cmd_buffers.len() + M <= N);

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.handle)
            .level(level)
            .command_buffer_count(M as u32);
        let handles = unsafe { self.loader.allocate_command_buffers_av::<M>(&allocate_info) }?;
        let buffers: VkResult<ArrayVec<CommandBuffer<Q, Initial>, M>> = handles
            .iter()
            .map(|&handle| CommandBuffer::new(self.loader.clone(), handle, self.handle, self.queue))
            .collect();

        self.pool_storage
            .persistent_cmd_buffers
            .extend(buffers?.into_iter().map(|b| PersistentCmdBuf::new(b)));

        let range = (self.pool_storage.persistent_cmd_buffers.len() - M)..;
        Ok(&mut self.pool_storage.persistent_cmd_buffers[range])
    }

    pub fn get_cmd_buffers(&mut self) -> &mut [PersistentCmdBuf<Q>] {
        &mut self.pool_storage.persistent_cmd_buffers
    }

    pub fn get_cmd_buffer(&mut self, idx: usize) -> &mut PersistentCmdBuf<Q> {
        &mut self.pool_storage.persistent_cmd_buffers[idx]
    }

    pub fn reset(&mut self) -> CmdBufStateTransitionResult<()> {
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
        unsafe { ManuallyDrop::drop(&mut self.pool_storage) };
        unsafe { self.loader.destroy_command_pool(self.handle, None) };
    }
}

mod private {
    pub trait Sealed {}
}
use private::Sealed;
pub trait CmdBufState: Sealed + 'static {}
pub trait ResettableState: CmdBufState {}

// Rendering state (with a render pass).
pub trait RenderingState: Sealed + 'static {}
pub enum InsideRenderPass {}
impl Sealed for InsideRenderPass {}
impl RenderingState for InsideRenderPass {}
pub enum OutsideRenderPass {}
impl Sealed for OutsideRenderPass {}
impl RenderingState for OutsideRenderPass {}

// Initial state.
pub enum Initial {}
impl Sealed for Initial {}
impl CmdBufState for Initial {}
impl ResettableState for Initial {}

// Recording state.
pub struct Recording<R: RenderingState>(PhantomData<R>);
impl<R: RenderingState> Sealed for Recording<R> {}
impl<R: RenderingState> CmdBufState for Recording<R> {}
impl<R: RenderingState> ResettableState for Recording<R> {}

// Executable state.
pub enum Executable {}
impl Sealed for Executable {}
impl CmdBufState for Executable {}
impl ResettableState for Executable {}

// Pending state.
// Command buffer cannot be freed or reset when in the pending state.
pub enum Pending {}
impl Sealed for Pending {}
impl CmdBufState for Pending {}

// Invalid state.
pub struct Invalid;
impl Sealed for Invalid {}
impl CmdBufState for Invalid {}
impl ResettableState for Invalid {}

// Command buffer state types.
pub type InitialCmdBuf<Q = PrimaryQueue> = CommandBuffer<Q, Initial>;
pub type RecordingCmdBuf<Q = PrimaryQueue, R = OutsideRenderPass> = CommandBuffer<Q, Recording<R>>;
pub type RenderingCmdBuf<Q = PrimaryQueue> = CommandBuffer<Q, Recording<InsideRenderPass>>;
pub type ExecutableCmdBuf<Q = PrimaryQueue> = CommandBuffer<Q, Executable>;
pub type PendingCmdBuf<Q = PrimaryQueue> = CommandBuffer<Q, Pending>;
pub type InvalidCmdBuf<Q = PrimaryQueue> = CommandBuffer<Q, Invalid>;

// This pattern (inner + transparent) allows transmuting between command buffer states (not between queue types).
// https://users.rust-lang.org/t/using-phantomdata-with-the-type-state-builder-pattern/99087/2
struct CommandBufferInner<Q: QueueType> {
    loader: SharedDeviceLoader,
    handle: vk::CommandBuffer,
    pool: vk::CommandPool,
    queue: vk::Queue,
    fence: Fence,
    queue_type: PhantomData<Q>,
}

#[repr(transparent)]
pub struct CommandBuffer<Q: QueueType, C: CmdBufState> {
    inner: CommandBufferInner<Q>,
    state: PhantomData<C>,
}

impl<Q: QueueType> CommandBuffer<Q, Initial> {
    fn new(
        loader: SharedDeviceLoader,
        handle: vk::CommandBuffer,
        pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> VkResult<CommandBuffer<Q, Initial>> {
        // Initially signalled, so it can be waited upon without being submitted.
        let fence = Fence::new(loader.as_ref(), true)?;
        Ok(Self {
            inner: CommandBufferInner {
                loader,
                handle,
                pool,
                queue,
                fence,
                queue_type: PhantomData,
            },
            state: PhantomData,
        })
    }
}

impl<Q: QueueType, C: CmdBufState> CommandBuffer<Q, C> {
    #[deprecated = "This should only be used until the relevant commands have been moved to the cmd buffer API."]
    pub fn handle_dep(&self) -> vk::CommandBuffer {
        self.handle()
    }

    // Inner accessors.
    fn loader(&self) -> &ash::Device {
        self.inner.loader.as_ref()
    }
    fn handle(&self) -> vk::CommandBuffer {
        self.inner.handle
    }
    fn pool(&self) -> vk::CommandPool {
        self.inner.pool
    }
    fn queue(&self) -> vk::Queue {
        self.inner.queue
    }
    fn fence(&self) -> &Fence {
        &self.inner.fence
    }
    //fn fence_mut(&mut self) -> &mut ManuallyDrop<Fence> {
    //    &mut self.inner.fence
    //}

    fn next_state<N: CmdBufState>(self) -> CommandBuffer<Q, N> {
        // Safe way to transmute between states.
        //CommandBuffer {
        //    inner: self.inner,
        //    state: PhantomData,
        //}

        // Potential no-op way to transmute between states.
        // This is safe because CommandBuffer is a transparent wrapper around CommandBufferInner,
        // which, because we are not changing the queue type, is just transmuted to itself.
        unsafe { std::mem::transmute(self) }
    }

    // Can be called on any state. An invalid state must be reset to be reused, so even if real
    // vulkan command buffer state is not invalid the type-state encoding is still sound.
    pub fn wait_for_fence(self) -> VkResult<CommandBuffer<Q, Invalid>> {
        self.fence().wait(self.loader())?;
        Ok(Self::next_state(self))
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
        unsafe { self.loader().begin_command_buffer(self.handle(), &begin_info) }?;
        Ok(self.next_state())
    }
}

impl CommandBuffer<PrimaryQueue, Recording<OutsideRenderPass>> {
    pub fn begin_rendering(
        self,
        ext: &DeviceExtensions,
        rendering_info: &vk::RenderingInfo,
    ) -> RecordingCmdBuf<PrimaryQueue, InsideRenderPass> {
        unsafe { ext.dyn_cmd.cmd_begin_rendering(self.handle(), rendering_info) };
        self.next_state()
    }
}

// Graphics recording commands.
impl<R: RenderingState> CommandBuffer<PrimaryQueue, Recording<R>> {
    pub fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        unsafe {
            self.loader()
                .cmd_bind_pipeline(self.handle(), vk::PipelineBindPoint::GRAPHICS, pipeline.handle())
        }
    }

    pub fn bind_index_buffer(&mut self, buffer: &Buffer<GpuOnly>, offset: vk::DeviceSize, index_type: vk::IndexType) {
        unsafe {
            self.loader()
                .cmd_bind_index_buffer(self.handle(), buffer.handle(), offset, index_type)
        };
    }

    pub fn bind_vertex_buffer(&mut self, buffer: &Buffer<GpuOnly>, vertices_offset: vk::DeviceSize) {
        unsafe {
            self.loader()
                .cmd_bind_vertex_buffers(self.handle(), 0, &[buffer.handle()], &[vertices_offset])
        };
    }

    pub fn set_viewport(&mut self, viewport: vk::Viewport) {
        unsafe { self.loader().cmd_set_viewport(self.handle(), 0, &[viewport]) };
    }

    pub fn set_scissor(&mut self, scissor: vk::Rect2D) {
        unsafe { self.loader().cmd_set_scissor(self.handle(), 0, &[scissor]) };
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
            self.loader().cmd_draw_indexed(
                self.handle(),
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        };
    }

    pub fn draw_indexed_indirect(
        &mut self,
        buffer: &Buffer<impl MemLocation>,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.loader()
                .cmd_draw_indexed_indirect(self.handle(), buffer.handle(), offset, draw_count, stride)
        };
    }

    pub fn draw_indexed_indirect_count(
        &mut self,
        buffer: &Buffer<GpuOnly>,
        offset: vk::DeviceSize,
        count_buffer: &Buffer<GpuOnly>,
        count_buffer_offset: vk::DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.loader().cmd_draw_indexed_indirect_count(
                self.handle(),
                buffer.handle(),
                offset,
                count_buffer.handle(),
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        };
    }

    pub fn end_rendering(self, ext: &DeviceExtensions) -> RecordingCmdBuf<PrimaryQueue, OutsideRenderPass> {
        unsafe { ext.dyn_cmd.cmd_end_rendering(self.handle()) };
        self.next_state()
    }
}

// Graphics/compute recording commands.
impl<Q: ComputeQueueType, R: RenderingState> CommandBuffer<Q, Recording<R>> {
    pub fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) {
        unsafe {
            self.loader()
                .cmd_bind_pipeline(self.handle(), vk::PipelineBindPoint::COMPUTE, pipeline.handle())
        }
    }

    pub fn push_constants(
        &mut self,
        pipeline_layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        data: &[u8],
    ) {
        unsafe {
            self.loader()
                .cmd_push_constants(self.handle(), pipeline_layout, stage_flags, offset, data)
        };
    }

    pub fn bind_descriptor_sets(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        pipeline_layout: vk::PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        unsafe {
            self.loader().cmd_bind_descriptor_sets(
                self.handle(),
                bind_point,
                pipeline_layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            )
        };
    }

    pub fn debug_marker_begin(&mut self, _ext: &DeviceExtensions, _name: &str) {
        #[cfg(feature = "debug_info")]
        {
            use std::ffi::CString;
            let name = CString::new(_name).unwrap();
            let label = vk::DebugUtilsLabelEXT::default()
                .label_name(&name)
                .color([0.0, 1.0, 0.0, 1.0]);
            _ext.run_debug(|dbg| unsafe { Ok(dbg.cmd_begin_debug_utils_label(self.handle(), &label)) })
                .unwrap();
        }
    }

    pub fn debug_marker_end(&mut self, _ext: &DeviceExtensions) {
        #[cfg(feature = "debug_info")]
        {
            _ext.run_debug(|dbg| unsafe { Ok(dbg.cmd_end_debug_utils_label(self.handle())) })
                .unwrap();
        }
    }
}

// Compute dispatches.
impl<Q: ComputeQueueType> CommandBuffer<Q, Recording<OutsideRenderPass>> {
    // Dispatch must be called when not rendering.
    pub fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            self.loader()
                .cmd_dispatch(self.handle(), group_count_x, group_count_y, group_count_z)
        };
    }
}

// Generic recording commands.
impl<Q: QueueType, R: RenderingState> CommandBuffer<Q, Recording<R>> {
    pub fn pipeline_barrier2(&mut self, device: &Device, dependency_info: &vk::DependencyInfo) {
        unsafe {
            device
                .extensions()
                .sync2
                .cmd_pipeline_barrier2(self.handle(), dependency_info)
        };
    }
}

impl<Q: QueueType> CommandBuffer<Q, Recording<OutsideRenderPass>> {
    pub fn copy_buffer<L1: MemLocation, L2: MemLocation>(
        &mut self,
        src_buffer: &Buffer<L1>,
        dst_buffer: &mut Buffer<L2>,
        copy_regions: &[vk::BufferCopy],
    ) {
        unsafe {
            self.loader()
                .cmd_copy_buffer(self.handle(), src_buffer.handle(), dst_buffer.handle(), copy_regions)
        };
    }

    pub fn update_buffer(&mut self, dst_buffer: &mut Buffer<impl MemLocation>, offset: vk::DeviceSize, data: &[u8]) {
        unsafe {
            self.loader()
                .cmd_update_buffer(self.handle(), dst_buffer.handle(), offset, data)
        };
    }

    pub fn copy_buffer_to_image(
        &mut self,
        src_buffer: &Buffer<impl MemLocation>,
        dst_image: &mut Image,
        dst_image_layout: vk::ImageLayout,
        copy_regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.loader().cmd_copy_buffer_to_image(
                self.handle(),
                src_buffer.handle(),
                dst_image.handle(),
                dst_image_layout,
                copy_regions,
            )
        };
    }

    pub fn end(self) -> VkResult<CommandBuffer<Q, Executable>> {
        unsafe { self.loader().end_command_buffer(self.handle()) }?;
        Ok(Self::next_state(self))
    }

    // On transient buffers ending, submitting, waiting and freeing are often all done in one go.
    pub fn end_submit_wait_and_free(self) -> VkResult<()> {
        let ended = self.end()?;
        let pending = ended.submit(&[], &[])?;
        pending.wait_for_fence()?;
        Ok(())
    }
}

// Executable commands.
impl<Q: QueueType> CommandBuffer<Q, Executable> {
    pub fn submit<const M: usize>(
        self,
        wait_semaphores: &[WaitSemaphore; M],
        signal_semaphores: &[vk::Semaphore],
    ) -> VkResult<CommandBuffer<Q, Pending>> {
        let semaphore_handles: ArrayVec<_, M> = wait_semaphores.iter().map(|sem| sem.handle).collect();
        let semaphore_stages: ArrayVec<_, M> = wait_semaphores.iter().map(|sem| sem.stage_mask).collect();

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&semaphore_handles)
            .wait_dst_stage_mask(&semaphore_stages)
            .signal_semaphores(signal_semaphores)
            .command_buffers(slice::from_ref(&self.inner.handle));

        // Reset fence right before submitting.
        self.fence().reset(self.loader())?;
        unsafe {
            self.loader()
                .queue_submit(self.queue(), &[submit_info], self.fence().handle())
        }?;

        Ok(self.next_state())
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
    fn new(cmd_buf: CommandBuffer<Q, Initial>) -> Self {
        Self::Initial(cmd_buf)
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

        if let Self::Pending(_) = self {
            return Err(CmdBufStateTransitionError::WrongState(
                self.state_str(),
                "reset command buffer",
            ));
        }

        // TODO: Check if this whole block is close to a no-op.
        *self = Self::Initial(match std::mem::replace(self, Self::Transitioning) {
            Self::Initial(cmd_buf) => cmd_buf.reset(),
            Self::Recording(cmd_buf) => cmd_buf.reset(),
            Self::Rendering(cmd_buf) => cmd_buf.reset(),
            Self::Executable(cmd_buf) => cmd_buf.reset(),
            Self::Pending(_) => unreachable!(),
            Self::Invalid(cmd_buf) => cmd_buf.reset(),
            Self::Transitioning => unreachable!(),
        });

        Ok(())
    }

    pub fn begin(&mut self) -> CmdBufStateTransitionResult<&mut CommandBuffer<Q, Recording<OutsideRenderPass>>> {
        self.check_not_transitioning();

        // Verify correct state.
        if !matches!(self, Self::Initial(_)) {
            Err(CmdBufStateTransitionError::WrongState(self.state_str(), "begin"))
        } else if let Self::Initial(cmd_buf) = std::mem::replace(self, Self::Transitioning) {
            *self = Self::Recording(cmd_buf.begin()?);
            let Self::Recording(cmd_buf) = self else { unreachable!() };
            Ok(cmd_buf)
        } else {
            unreachable!()
        }
    }

    // Because of Rust's mutability rules, the recording buffer returned from begin() cannot be used after this method.
    pub fn end(&mut self) -> CmdBufStateTransitionResult<()> {
        self.check_not_transitioning();

        if !matches!(self, Self::Recording(_)) {
            Err(CmdBufStateTransitionError::WrongState(self.state_str(), "end"))
        } else if let Self::Recording(cmd_buf) = std::mem::replace(self, Self::Transitioning) {
            *self = Self::Executable(cmd_buf.end()?);
            Ok(())
        } else {
            unreachable!()
        }
    }

    pub fn submit<const M: usize>(
        &mut self,
        wait_semaphores: &[WaitSemaphore; M],
        signal_semaphores: &[vk::Semaphore],
    ) -> CmdBufStateTransitionResult<()> {
        self.check_not_transitioning();

        if !matches!(self, Self::Executable(_)) {
            Err(CmdBufStateTransitionError::WrongState(self.state_str(), "submit"))
        } else if let Self::Executable(cmd_buf) = std::mem::replace(self, Self::Transitioning) {
            *self = Self::Pending(cmd_buf.submit(wait_semaphores, signal_semaphores)?);
            Ok(())
        } else {
            unreachable!()
        }
    }

    pub fn wait_for_fence(&mut self) -> CmdBufStateTransitionResult<()> {
        self.check_not_transitioning();

        *self = match std::mem::replace(self, Self::Transitioning) {
            PersistentCmdBuf::Invalid(cmd_buf) => PersistentCmdBuf::Invalid(cmd_buf.wait_for_fence()?),
            PersistentCmdBuf::Initial(cmd_buf) => PersistentCmdBuf::Invalid(cmd_buf.wait_for_fence()?),
            PersistentCmdBuf::Recording(cmd_buf) => PersistentCmdBuf::Invalid(cmd_buf.wait_for_fence()?),
            PersistentCmdBuf::Rendering(cmd_buf) => PersistentCmdBuf::Invalid(cmd_buf.wait_for_fence()?),
            PersistentCmdBuf::Executable(cmd_buf) => PersistentCmdBuf::Invalid(cmd_buf.wait_for_fence()?),
            PersistentCmdBuf::Pending(cmd_buf) => PersistentCmdBuf::Invalid(cmd_buf.wait_for_fence()?),
            PersistentCmdBuf::Transitioning => unreachable!(),
        };
        Ok(())
    }

    // Internal use only.
    //fn free_fence(&mut self) {
    //    match self {
    //        Self::Invalid(cmd_buf) => unsafe { ManuallyDrop::drop(cmd_buf.fence_mut()) },
    //        Self::Initial(cmd_buf) => unsafe { ManuallyDrop::drop(cmd_buf.fence_mut()) },
    //        Self::Recording(cmd_buf) => unsafe { ManuallyDrop::drop(cmd_buf.fence_mut()) },
    //        Self::Rendering(cmd_buf) => unsafe { ManuallyDrop::drop(cmd_buf.fence_mut()) },
    //        Self::Executable(cmd_buf) => unsafe { ManuallyDrop::drop(cmd_buf.fence_mut()) },
    //        Self::Pending(cmd_buf) => unsafe { ManuallyDrop::drop(cmd_buf.fence_mut()) },
    //        Self::Transitioning => unreachable!(),
    //    }
    //}
}

impl PersistentCmdBuf<PrimaryQueue> {
    pub fn begin_rendering(
        &mut self,
        ext: &DeviceExtensions,
        render_info: &vk::RenderingInfo,
    ) -> CmdBufStateTransitionResult<&mut CommandBuffer<PrimaryQueue, Recording<InsideRenderPass>>> {
        self.check_not_transitioning();

        if !matches!(self, Self::Recording(_)) {
            Err(CmdBufStateTransitionError::WrongState(
                self.state_str(),
                "begin rendering",
            ))
        } else if let Self::Recording(cmd_buf) = std::mem::replace(self, Self::Transitioning) {
            *self = Self::Rendering(cmd_buf.begin_rendering(ext, render_info));
            let Self::Rendering(cmd_buf) = self else { unreachable!() };
            Ok(cmd_buf)
        } else {
            unreachable!()
        }
    }

    pub fn end_rendering(
        &mut self,
        ext: &DeviceExtensions,
    ) -> CmdBufStateTransitionResult<&mut CommandBuffer<PrimaryQueue, Recording<OutsideRenderPass>>> {
        self.check_not_transitioning();

        if !matches!(self, Self::Rendering(_)) {
            Err(CmdBufStateTransitionError::WrongState(
                self.state_str(),
                "end rendering",
            ))
        } else if let Self::Rendering(cmd_buf) = std::mem::replace(self, Self::Transitioning) {
            *self = Self::Recording(cmd_buf.end_rendering(ext));
            let Self::Recording(cmd_buf) = self else { unreachable!() };
            Ok(cmd_buf)
        } else {
            unreachable!()
        }
    }
}

impl<Q: QueueType, C: CmdBufState> Drop for CommandBuffer<Q, C> {
    fn drop(&mut self) {
        // Convert to immutable reference.
        let this = &*self;
        if let Ok(this) = cast!(this, &CommandBuffer<Q, Pending>) {
            // Wait for the fence now.
            this.fence().wait(self.loader()).unwrap();
        }
        unsafe { self.loader().free_command_buffers(self.pool(), &[self.handle()]) };
    }
}
