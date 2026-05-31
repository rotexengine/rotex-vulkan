use ash::vk;

use super::super::device::Device;
use crate::error::vk_error;
use crate::Error;

/// Owned `VkDescriptorSetLayout`.
pub struct DescriptorSetLayout {
    handle: vk::DescriptorSetLayout,
}

impl DescriptorSetLayout {
    /// Creates a descriptor set layout from `bindings`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `vkCreateDescriptorSetLayout` fails.
    pub fn new(
        device: &Device,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> Result<Self, Error> {
        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);

        let handle = unsafe {
            device
                .logical_device()
                .create_descriptor_set_layout(&create_info, None)
        }
        .map_err(vk_error)?;

        Ok(Self { handle })
    }

    /// Returns the `VkDescriptorSetLayout` handle.
    pub fn handle(&self) -> vk::DescriptorSetLayout {
        self.handle
    }

    /// Destroys the descriptor set layout.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device
                .logical_device()
                .destroy_descriptor_set_layout(self.handle, None);
        }
    }
}

/// Owned `VkPipelineLayout` for graphics pipelines.
pub struct GraphicsPipelineLayout {
    handle: vk::PipelineLayout,
}

impl GraphicsPipelineLayout {
    /// Creates a pipeline layout from descriptor set layouts and push constant ranges.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `vkCreatePipelineLayout` fails.
    pub fn new(
        device: &Device,
        set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self, Error> {
        let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(set_layouts)
            .push_constant_ranges(push_constant_ranges);

        let handle = unsafe {
            device
                .logical_device()
                .create_pipeline_layout(&create_info, None)
        }
        .map_err(vk_error)?;

        Ok(Self { handle })
    }

    /// Returns the `VkPipelineLayout` handle.
    pub fn handle(&self) -> vk::PipelineLayout {
        self.handle
    }

    /// Destroys the pipeline layout.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device
                .logical_device()
                .destroy_pipeline_layout(self.handle, None);
        }
    }
}
