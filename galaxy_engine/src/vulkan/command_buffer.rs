// Copyright (c) 2024. Ben Sutherland

use std::marker::PhantomData;
use std::slice;

use arrayvec::ArrayVec;
use ash::prelude::VkResult;
use ash::vk;
use castaway::match_type;

use crate::vulkan::device::{Device, DeviceExt, SharedDeviceLoader};
use crate::vulkan::queue::queue_type::{Primary, QueueType};
use crate::vulkan::queue::Queue;
use crate::vulkan::sync::{Fence, WaitSemaphore};

pub type PrimaryCommandPool<T> = CommandPool<Primary, T>;
pub type ResettablePrimaryCommandPool = CommandPool<Primary, Resettable<Primary, OneTime>>;
pub type TransientPrimaryCommandPool = CommandPool<Primary, Transient>;
pub type PrimaryCommandBuffer<S> = CommandBuffer<Primary, S>;

pub trait CommandPoolType: Default {
    const FLAGS: vk::CommandPoolCreateFlags = vk::CommandPoolCreateFlags::empty();
}
#[derive(Default)]
pub struct Transient;
impl CommandPoolType for Transient {
    const FLAGS: vk::CommandPoolCreateFlags = vk::CommandPoolCreateFlags::TRANSIENT;
}
pub struct Resettable<Q: QueueType, O: OneTimeOrPersistentState> {
    persistent_cmd_buffers: Vec<PersistentCmdBuf<Q, O>>,
}

impl<Q: QueueType, O: OneTimeOrPersistentState> Default for Resettable<Q, O> {
    fn default() -> Self {
        Self {
            persistent_cmd_buffers: Vec::new(),
        }
    }
}

impl<Q: QueueType, O: OneTimeOrPersistentState> CommandPoolType for Resettable<Q, O> {}

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

impl<Q: QueueType, O: OneTimeOrPersistentState> CommandPool<Q, Resettable<Q, O>> {
    pub fn allocate_cmd_buffer(&mut self, level: vk::CommandBufferLevel) -> VkResult<&mut PersistentCmdBuf<Q, O>> {
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
    ) -> VkResult<&mut [PersistentCmdBuf<Q, O>]> {
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

    pub fn get_cmd_buffers(&mut self) -> &mut [PersistentCmdBuf<Q, O>] {
        &mut self.pool_storage.persistent_cmd_buffers
    }

    pub fn get_cmd_buffer(&mut self, idx: usize) -> &mut PersistentCmdBuf<Q, O> {
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

    pub trait State: 'static {}
    pub trait ResettableState: State {}
    pub trait CompletedState: ResettableState {}

    // Some states differ if the command buffer is one-time-submit or persistent.
    pub trait OneTimeOrPersistentState: 'static {
        const FLAGS: vk::CommandBufferUsageFlags;
        type CompletedState: CompletedState;
    }
    pub struct OneTime;
    impl OneTimeOrPersistentState for OneTime {
        const FLAGS: vk::CommandBufferUsageFlags = vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT;
        type CompletedState = Invalid;
    }
    pub struct Persistent;
    impl OneTimeOrPersistentState for Persistent {
        const FLAGS: vk::CommandBufferUsageFlags = vk::CommandBufferUsageFlags::empty();
        type CompletedState = Executable<Persistent>;
    }

    // Initial state.
    pub struct Initial;
    impl State for Initial {}
    impl ResettableState for Initial {}

    // Recording state.
    pub struct Recording<O: OneTimeOrPersistentState>(std::marker::PhantomData<O>);
    impl<O: OneTimeOrPersistentState> State for Recording<O> {}
    impl<O: OneTimeOrPersistentState> ResettableState for Recording<O> {}

    // Executable state.
    pub struct Executable<O: OneTimeOrPersistentState>(std::marker::PhantomData<O>);
    impl<O: OneTimeOrPersistentState> State for Executable<O> {}
    impl<O: OneTimeOrPersistentState> ResettableState for Executable<O> {}
    impl CompletedState for Executable<Persistent> {}

    // Pending state.
    // Command buffer cannot be reset when in the pending state.
    pub trait PendingState: State {
        type CompletedState: CompletedState;
    }
    pub struct Pending<O: OneTimeOrPersistentState>(std::marker::PhantomData<O>);
    impl<O: OneTimeOrPersistentState> State for Pending<O> {}
    impl<O: OneTimeOrPersistentState> PendingState for Pending<O> {
        type CompletedState = O::CompletedState;
    }

    pub struct Invalid;
    impl State for Invalid {}
    impl ResettableState for Invalid {}
    impl CompletedState for Invalid {}
}
// Public exports
pub use command_buffer_states::OneTimeOrPersistentState;
use command_buffer_states::*;

// Primary queue command buffer states.
pub type InitialCmdBuf<Q = Primary> = CommandBuffer<Q, Initial>;
pub type RecordingCmdBuf<Q = Primary, O = OneTime> = CommandBuffer<Q, Recording<O>>;
pub type ExecutableCmdBuf<Q = Primary, O = OneTime> = CommandBuffer<Q, Executable<O>>;
pub type PendingCmdBuf<Q = Primary, O = OneTime> = CommandBuffer<Q, Pending<O>>;
pub type InvalidCmdBuf<Q = Primary> = CommandBuffer<Q, Invalid>;

use crate::vulkan::buffer::{Buffer, MemLocation};
use crate::vulkan::pipeline::{GraphicsPipeline, Pipeline};

pub struct CommandBuffer<Q: QueueType, S: State> {
    loader: SharedDeviceLoader,
    handle: vk::CommandBuffer,
    pool: vk::CommandPool,
    queue: vk::Queue,
    queue_type: PhantomData<Q>,
    state: PhantomData<S>,
}

// Always available methods.
impl<Q: QueueType, S: State> CommandBuffer<Q, S> {
    // TODO: Clean up uses of this method.
    //#[deprecated = "This should only be used until the relevant commands have been moved to the cmd buffer API."]
    pub fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    // TODO: Is this a no-op function?
    fn next_state<T: State>(self) -> CommandBuffer<Q, T> {
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

impl<Q: QueueType, S: ResettableState> CommandBuffer<Q, S> {
    // This is for internal state-management use only.
    // Buffers are either persistent and reset by the command pool, or transient and immediately freed.
    fn reset(self) -> CommandBuffer<Q, Initial> {
        self.next_state()
    }
}

impl<Q: QueueType> CommandBuffer<Q, Initial> {
    pub fn begin<O: OneTimeOrPersistentState>(self) -> VkResult<CommandBuffer<Q, Recording<O>>> {
        let begin_info = vk::CommandBufferBeginInfo::default().flags(O::FLAGS);
        unsafe { self.loader.begin_command_buffer(self.handle, &begin_info) }?;
        Ok(self.next_state())
    }
}

impl<O: OneTimeOrPersistentState> CommandBuffer<Primary, Recording<O>> {
    pub fn bind_graphics_pipeline(&self, pipeline: &GraphicsPipeline) {
        unsafe {
            self.loader
                .cmd_bind_pipeline(self.handle, vk::PipelineBindPoint::GRAPHICS, pipeline.handle())
        }
    }
}

impl<Q: QueueType, O: OneTimeOrPersistentState> CommandBuffer<Q, Recording<O>> {
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

    pub fn end(self) -> VkResult<CommandBuffer<Q, Executable<O>>> {
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

impl<Q: QueueType, O: OneTimeOrPersistentState> CommandBuffer<Q, Executable<O>> {
    pub fn submit<const M: usize>(
        self,
        wait_semaphores: &[WaitSemaphore; M],
        signal_semaphores: &[vk::Semaphore],
        fence: Option<&Fence>,
    ) -> VkResult<CommandBuffer<Q, Pending<O>>> {
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

impl<Q: QueueType, S: PendingState> CommandBuffer<Q, S> {
    pub fn queue_wait_idle(self) -> VkResult<CommandBuffer<Q, S::CompletedState>> {
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

pub enum PersistentCmdBuf<Q: QueueType, O: OneTimeOrPersistentState> {
    Invalid(CommandBuffer<Q, Invalid>),
    Initial(CommandBuffer<Q, Initial>),
    Recording(CommandBuffer<Q, Recording<O>>),
    Executable(CommandBuffer<Q, Executable<O>>),
    Pending(CommandBuffer<Q, Pending<O>>),
    // Only transitioning within a method call.
    // Used for transitioning between states (so buffer can be moved in and out).
    Transitioning,
}

impl<Q: QueueType, O: OneTimeOrPersistentState + 'static> PersistentCmdBuf<Q, O> {
    fn new<S: State>(cmd_buf: CommandBuffer<Q, S>) -> Self {
        match_type!(cmd_buf, {
            CommandBuffer<Q, Initial> as cmd_buf => Self::Initial(cmd_buf),
            CommandBuffer<Q, Recording<O>> as cmd_buf => Self::Recording(cmd_buf),
            CommandBuffer<Q, Executable<O>> as cmd_buf => Self::Executable(cmd_buf),
            CommandBuffer<Q, Pending<O>> as cmd_buf => Self::Pending(cmd_buf),
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

    pub fn begin(&mut self) -> Result<&mut CommandBuffer<Q, Recording<O>>, PersistentCmdBufError> {
        self.check_not_transitioning();

        if let Self::Initial(cmd_buf) = std::mem::replace(self, Self::Transitioning) {
            *self = Self::Recording(cmd_buf.begin::<O>()?);
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

//impl Drop for CommandBuffer<Q, S> {
//    fn drop(&mut self) {
//        self.free()
//    }
//}
