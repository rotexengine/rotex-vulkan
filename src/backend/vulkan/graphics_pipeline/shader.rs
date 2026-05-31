use std::ffi::CStr;

use ash::vk;

use super::super::device::Device;
use crate::error::vk_error;
use crate::Error;

/// Owned `VkShaderModule` created from SPIR-V words.
pub struct ShaderModule {
    pub(crate) handle: vk::ShaderModule,
}

impl ShaderModule {
    /// Creates a shader module from SPIR-V `spv_code`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `vkCreateShaderModule` fails.
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

    /// Returns the `VkShaderModule` handle.
    pub fn handle(&self) -> vk::ShaderModule {
        self.handle
    }

    /// Destroys the shader module.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device
                .logical_device()
                .destroy_shader_module(self.handle, None);
        }
    }
}

/// Shader stage binding for pipeline creation (`VkPipelineShaderStageCreateInfo`).
pub struct ShaderStageDescriptor<'a> {
    pub(crate) stage: vk::ShaderStageFlags,
    pub(crate) module: &'a ShaderModule,
    pub(crate) entry_name: &'a CStr,
}

impl<'a> ShaderStageDescriptor<'a> {
    /// Builds a stage descriptor with entry point `"main"`.
    pub fn new(stage: vk::ShaderStageFlags, module: &'a ShaderModule) -> Self {
        Self {
            stage,
            module,
            entry_name: c"main",
        }
    }

    /// Sets the SPIR-V entry point name.
    pub fn with_entry_name(mut self, name: &'a CStr) -> Self {
        self.entry_name = name;
        self
    }
}
