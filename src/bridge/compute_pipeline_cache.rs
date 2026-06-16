use std::ffi::CString;

use ash::vk;

use super::VulkanBridge;
use crate::backend::vulkan::{
    ComputePipeline, GraphicsPipelineLayout, ShaderModule,
    ShaderStageDescriptor,
};
use crate::error::{Error, ErrorKind};
use rotex_types::resource::ComputePipelineDescriptor;

use super::bindings;
use super::pipeline_cache::spv_bytes_to_words;

impl VulkanBridge {
    pub(super) fn create_compute_pipeline_resource(
        &mut self,
        descriptor: &ComputePipelineDescriptor,
    ) -> Result<super::types::ComputePipelineResource, Error> {
        let device = self.device.raw();
        let shader_package = &descriptor.shader;
        let spirv = shader_package.spirv_bytes().ok_or_else(|| {
            Error::fatal(ErrorKind::Unsupported("compute_shader_spirv_missing"))
        })?;
        let set_layouts =
            bindings::build_set_layouts_from_abstract_layout(device, &shader_package.layout)?;
        let layout_handles: Vec<_> = set_layouts.iter().map(|layout| layout.handle()).collect();
        let pipeline_layout = GraphicsPipelineLayout::new(device, &layout_handles, &[])?;

        let words = spv_bytes_to_words(spirv);
        let entry = CString::new(shader_package.entry_point.as_str()).map_err(|_| {
            Error::fatal(ErrorKind::Unsupported(
                "Compute shader entry contains interior null byte",
            ))
        })?;
        let shader = ShaderModule::new(device, &words)?;
        let stage = ShaderStageDescriptor::new(vk::ShaderStageFlags::COMPUTE, &shader)
            .with_entry_name(entry.as_c_str());
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(stage.stage)
            .module(stage.module.handle())
            .name(stage.entry_name);
        let pipeline = ComputePipeline::new(device, &stage_info, pipeline_layout.handle())?;
        shader.destroy(device);

        let mut descriptor_sets = Vec::with_capacity(self.frames_in_flight as usize);
        if set_layouts.is_empty() {
            for _ in 0..self.frames_in_flight {
                descriptor_sets.push(Vec::new());
            }
        } else {
            let layout_handles: Vec<_> = set_layouts.iter().map(|layout| layout.handle()).collect();
            for _ in 0..self.frames_in_flight {
                descriptor_sets.push(
                    self.storage_descriptor_pool
                        .allocate_sets(device, &layout_handles)?,
                );
            }
        }

        Ok(super::types::ComputePipelineResource {
            _descriptor: descriptor.clone(),
            pipeline,
            pipeline_layout,
            set_layouts,
            descriptor_sets,
        })
    }

    pub(super) fn update_compute_descriptor_sets(
        &self,
        pipeline_id: rotex_types::resource::ComputePipelineId,
        buffer_intents: &[rotex_types::BufferUsageIntent],
    ) -> Result<(), Error> {
        let resource = self
            .compute_pipelines
            .get(&pipeline_id)
            .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?;
        let device = self.device.raw();
        let intents_by_set: std::collections::BTreeMap<u32, Vec<&rotex_types::BufferUsageIntent>> = buffer_intents
            .iter()
            .fold(std::collections::BTreeMap::new(), |mut map, intent| {
                map.entry(intent.set).or_default().push(intent);
                map
            });

        let slot_index = self.current_frame_index as usize;
        let slot_sets = resource.descriptor_sets.get(slot_index).ok_or(
            Error::fatal(ErrorKind::Unsupported(
                "Compute descriptor set slot out of range",
            )),
        )?;

        for (set_index, intents) in intents_by_set {
            let descriptor_set = slot_sets.get(set_index as usize).ok_or(
                Error::fatal(ErrorKind::Unsupported(
                    "Compute pass references descriptor set not declared on pipeline",
                )),
            )?;
            for intent in intents {
                let buffer = self
                    .buffers
                    .get(&intent.buffer)
                    .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?;
                let range = if intent.size == 0 {
                    buffer.size
                } else {
                    intent.size
                };
                descriptor_set.write_buffer(
                    device,
                    intent.binding,
                    &buffer.buffer,
                    intent.offset,
                    range,
                    vk::DescriptorType::STORAGE_BUFFER,
                );
            }
        }
        Ok(())
    }

    pub(super) fn destroy_all_compute_pipelines(&mut self) {
        let pipelines: Vec<_> = self.compute_pipelines.drain().map(|(_, p)| p).collect();
        for pipeline in pipelines {
            pipeline.destroy(self.device.raw(), &self.storage_descriptor_pool);
        }
    }
}
