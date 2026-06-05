use ash::vk;

use super::device::Device;
use crate::Error;
use crate::error::vk_error;

pub struct ComputePipeline {
    handle: vk::Pipeline,
}

impl ComputePipeline {
    pub fn new(
        device: &Device,
        stage: &vk::PipelineShaderStageCreateInfo<'_>,
        layout: vk::PipelineLayout,
    ) -> Result<Self, Error> {
        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(*stage)
            .layout(layout);

        let pipelines = unsafe {
            device.logical_device().create_compute_pipelines(
                vk::PipelineCache::null(),
                &[create_info],
                None,
            )
        }
        .map_err(|(_, err)| vk_error(err))?;

        Ok(Self {
            handle: pipelines[0],
        })
    }

    pub fn handle(&self) -> vk::Pipeline {
        self.handle
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.logical_device().destroy_pipeline(self.handle, None);
        }
    }
}
