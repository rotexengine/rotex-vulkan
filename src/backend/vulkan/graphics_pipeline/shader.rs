use std::ffi::CStr;

use ash::vk;

use super::super::device::Device;
use crate::Error;
use crate::error::vk_error;

pub struct ShaderModule {
    pub(crate) handle: vk::ShaderModule,
}

impl ShaderModule {
    pub fn new(device: &Device, spv_code: &[u32]) -> Result<Self, Error> {
        let create_info = vk::ShaderModuleCreateInfo::default().code(spv_code);

        let handle = unsafe {
            device
                .logical_device()
                .create_shader_module(&create_info, None)
        }
        .map_err(vk_error)?;

        Ok(Self { handle })
    }

    pub fn handle(&self) -> vk::ShaderModule {
        self.handle
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device
                .logical_device()
                .destroy_shader_module(self.handle, None);
        }
    }
}

pub struct ShaderStageDescriptor<'a> {
    pub(crate) stage: vk::ShaderStageFlags,
    pub(crate) module: &'a ShaderModule,
    pub(crate) entry_name: &'a CStr,
}

impl<'a> ShaderStageDescriptor<'a> {
    pub fn new(stage: vk::ShaderStageFlags, module: &'a ShaderModule) -> Self {
        Self {
            stage,
            module,
            entry_name: c"main",
        }
    }

    pub fn with_entry_name(mut self, name: &'a CStr) -> Self {
        self.entry_name = name;
        self
    }
}
