use ash::vk;

use super::device::Device;
use crate::error::{Error, vk_error};

pub struct Semaphore {
    pub(crate) handle: vk::Semaphore,
}

impl Semaphore {
    pub fn new(device: &Device) -> Result<Self, Error> {
        let create_info = vk::SemaphoreCreateInfo::default();

        let handle = unsafe { device.logical_device().create_semaphore(&create_info, None) }
            .map_err(vk_error)?;

        Ok(Self { handle })
    }

    pub fn handle(&self) -> vk::Semaphore {
        self.handle
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.logical_device().destroy_semaphore(self.handle, None);
        }
    }
}

pub struct Fence {
    pub(crate) handle: vk::Fence,
}

impl Fence {
    pub fn new(device: &Device, signaled: bool) -> Result<Self, Error> {
        let mut create_info = vk::FenceCreateInfo::default();

        if signaled {
            create_info = create_info.flags(vk::FenceCreateFlags::SIGNALED);
        }

        let handle = unsafe { device.logical_device().create_fence(&create_info, None) }
            .map_err(vk_error)?;

        Ok(Self { handle })
    }

    pub fn handle(&self) -> vk::Fence {
        self.handle
    }

    pub fn wait(&self, device: &Device, timeout_ns: u64) -> Result<(), Error> {
        unsafe {
            device
                .logical_device()
                .wait_for_fences(&[self.handle], true, timeout_ns)
        }
        .map_err(vk_error)
    }

    pub fn reset(&self, device: &Device) -> Result<(), Error> {
        unsafe { device.logical_device().reset_fences(&[self.handle]) }.map_err(vk_error)
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.logical_device().destroy_fence(self.handle, None);
        }
    }
}
