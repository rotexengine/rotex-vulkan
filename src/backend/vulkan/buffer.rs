use ash::vk;

use super::device::Device;
use crate::core::Instance;
use crate::error::{Error, ErrorKind};

pub struct RotexBuffer {
    handle: vk::Buffer,
    device_memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl RotexBuffer {
    pub fn new(
        instance: &Instance,
        device: &Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<Self, Error> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let handle = unsafe { device.logical_device().create_buffer(&buffer_info, None) }
            .map_err(ErrorKind::Vulkan)
            .map_err(Error::fatal)?;

        let requirements = unsafe {
            device
                .logical_device()
                .get_buffer_memory_requirements(handle)
        };
        let memory_type =
            device.find_memory_type(instance, requirements.memory_type_bits, properties)?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);

        let device_memory = unsafe { device.logical_device().allocate_memory(&alloc_info, None) }
            .map_err(ErrorKind::Vulkan)
            .map_err(Error::fatal)?;

        unsafe {
            device
                .logical_device()
                .bind_buffer_memory(handle, device_memory, 0)
        }
        .map_err(ErrorKind::Vulkan)
        .map_err(Error::fatal)?;

        Ok(Self {
            handle,
            device_memory,
            size,
        })
    }

    pub fn map(&self, device: &Device) -> Result<*mut std::ffi::c_void, Error> {
        unsafe {
            device.logical_device().map_memory(
                self.device_memory,
                0,
                self.size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(ErrorKind::Vulkan)
        .map_err(Error::fatal)
    }

    pub fn unmap(&self, device: &Device) {
        unsafe { device.logical_device().unmap_memory(self.device_memory) };
    }

    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.logical_device().destroy_buffer(self.handle, None);
            device
                .logical_device()
                .free_memory(self.device_memory, None);
        }
    }
}
