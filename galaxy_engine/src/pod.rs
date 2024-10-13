use bytemuck::{Pod, Zeroable};

// New-type wrappers to mark external crate types as Pod.

pub(crate) mod vk {
    use super::*;
    use ash::vk;

    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct DrawIndexedIndirectCommand(vk::DrawIndexedIndirectCommand);

    unsafe impl Zeroable for DrawIndexedIndirectCommand {}
    unsafe impl Pod for DrawIndexedIndirectCommand {}
}