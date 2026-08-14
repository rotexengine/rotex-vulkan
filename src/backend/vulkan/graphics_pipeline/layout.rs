use ash::vk;

use super::super::device::Device;
use crate::Error;
use crate::error::vk_error;

pub struct DescriptorSetLayout {
    handle: vk::DescriptorSetLayout,
}

impl DescriptorSetLayout {
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

    pub fn handle(&self) -> vk::DescriptorSetLayout {
        self.handle
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device
                .logical_device()
                .destroy_descriptor_set_layout(self.handle, None);
        }
    }
}

pub struct GraphicsPipelineLayout {
    handle: vk::PipelineLayout,
}

impl GraphicsPipelineLayout {
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

    pub fn handle(&self) -> vk::PipelineLayout {
        self.handle
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device
                .logical_device()
                .destroy_pipeline_layout(self.handle, None);
        }
    }
}
