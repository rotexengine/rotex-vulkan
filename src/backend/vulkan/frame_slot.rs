use super::command::CommandBuffer;
use super::sync::{Fence, Semaphore};
use crate::error::Error;

pub struct FrameSlot {
    pub command_buffer: CommandBuffer,
    pub fence: Fence,
    pub image_available: Semaphore,
}

impl FrameSlot {
    pub fn wait_and_reset(&self, device: &super::device::Device) -> Result<(), Error> {
        self.fence.wait(device, u64::MAX)?;
        self.fence.reset(device)
    }
}
