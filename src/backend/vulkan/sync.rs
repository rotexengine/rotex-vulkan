use ash::vk;

use super::device::Device;
use crate::error::{Error, vk_error};

/// GPU timeline semaphore.
pub struct Semaphore {
    pub(crate) handle: vk::Semaphore,
}

impl Semaphore {
    /// Creates an unsignaled semaphore.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `vkCreateSemaphore` fails.
    pub fn new(device: &Device) -> Result<Self, Error> {
        let create_info = vk::SemaphoreCreateInfo::default();

        let handle = unsafe { device.logical_device().create_semaphore(&create_info, None) }
            .map_err(vk_error)?;

        Ok(Self { handle })
    }

    /// `VkSemaphore` handle.
    pub fn handle(&self) -> vk::Semaphore {
        self.handle
    }

    /// Destroys the semaphore.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.logical_device().destroy_semaphore(self.handle, None);
        }
    }
}

/// CPU–GPU synchronization fence.
pub struct Fence {
    pub(crate) handle: vk::Fence,
}

impl Fence {
    /// Creates a fence, optionally in the signaled state.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `vkCreateFence` fails.
    pub fn new(device: &Device, signaled: bool) -> Result<Self, Error> {
        let mut create_info = vk::FenceCreateInfo::default();

        if signaled {
            create_info = create_info.flags(vk::FenceCreateFlags::SIGNALED);
        }

        let handle = unsafe { device.logical_device().create_fence(&create_info, None) }
            .map_err(vk_error)?;

        Ok(Self { handle })
    }

    /// `VkFence` handle.
    pub fn handle(&self) -> vk::Fence {
        self.handle
    }

    /// Blocks until the fence is signaled or `timeout_ns` elapses.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `vkWaitForFences` fails or times out.
    pub fn wait(&self, device: &Device, timeout_ns: u64) -> Result<(), Error> {
        unsafe {
            device
                .logical_device()
                .wait_for_fences(&[self.handle], true, timeout_ns)
        }
        .map_err(vk_error)
    }

    /// Resets the fence to unsignaled.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `vkResetFences` fails.
    pub fn reset(&self, device: &Device) -> Result<(), Error> {
        unsafe { device.logical_device().reset_fences(&[self.handle]) }.map_err(vk_error)
    }

    /// Destroys the fence.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.logical_device().destroy_fence(self.handle, None);
        }
    }
}
