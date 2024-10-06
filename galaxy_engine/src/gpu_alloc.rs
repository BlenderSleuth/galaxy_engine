use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};

pub type SharedAllocator = Arc<Mutex<Allocator>>;
// TODO: Newtype pattern for ManuallyDrop<Allocation>?
pub type ManuallyFreeAllocation = ManuallyDrop<Allocation>;

#[derive(thiserror::Error, Debug)]
pub enum MemoryError {
    #[error("Memory vulkan error: {0}")]
    VkResult(#[from] vk::Result),
    #[error("Allocation error: {0}")]
    AllocationError(#[from] gpu_allocator::AllocationError),
    #[error("Copy error: {0}")]
    CopyError(#[from] presser::CopyError),
}
pub type AllocResult<T> = gpu_allocator::Result<T>;
pub type MemResult<T> = Result<T, MemoryError>;

pub unsafe fn free_or_log_on_fail(allocator: &SharedAllocator, allocation: &mut ManuallyFreeAllocation) {
    // Lock allocator.
    let Ok(mut allocator) = allocator.lock() else {
        log::error!("Failed to lock allocator: mutex poisoned.");
        return;
    };

    // Free memory.
    allocator.free(unsafe { ManuallyDrop::take(allocation) }).unwrap_or_else(|err| {
        log::error!("Failed to free buffer memory: {err}")
    });
}
