#![allow(dead_code)]
use std::collections::BTreeMap;
use std::ffi::CString;

use ash::vk;

use super::VulkanBridge;
use crate::backend::vulkan::{
    ComputePipeline, DescriptorSetLayout, GraphicsPipelineLayout, ShaderModule,
    ShaderStageDescriptor,
};
use crate::error::{Error, ErrorKind};
use rotex_types::resource::{ComputeBindingLayout, ComputePipelineDescriptor};

use super::pipeline_cache::spv_bytes_to_words;

impl VulkanBridge {
    pub(super) fn create_compute_pipeline_resource(
        &mut self,
        descriptor: &ComputePipelineDescriptor,
    ) -> Result<super::types::ComputePipelineResource, Error> {
        let device = self.device.raw();
        let set_layouts = build_compute_set_layouts(device, &descriptor.bindings)?;
        let layout_handles: Vec<_> = set_layouts.iter().map(|layout| layout.handle()).collect();
        let pipeline_layout = GraphicsPipelineLayout::new(device, &layout_handles, &[])?;

        let words = spv_bytes_to_words(&descriptor.shader_spv);
        let entry = CString::new(descriptor.entry_point.as_str()).map_err(|_| {
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

        let mut descriptor_sets = Vec::with_capacity(set_layouts.len());
        if !set_layouts.is_empty() {
            let layout_handles: Vec<_> = set_layouts.iter().map(|layout| layout.handle()).collect();
            descriptor_sets = self
                .storage_descriptor_pool
                .allocate_sets(device, &layout_handles)?;
        }

        Ok(super::types::ComputePipelineResource {
            descriptor: descriptor.clone(),
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
        let intents_by_set: BTreeMap<u32, Vec<&rotex_types::BufferUsageIntent>> = buffer_intents
            .iter()
            .fold(BTreeMap::new(), |mut map, intent| {
                map.entry(intent.set).or_default().push(intent);
                map
            });

        for (set_index, intents) in intents_by_set {
            let descriptor_set =
                resource
                    .descriptor_sets
                    .get(set_index as usize)
                    .ok_or(Error::fatal(ErrorKind::Unsupported(
                        "Compute pass references descriptor set not declared on pipeline",
                    )))?;
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

fn build_compute_set_layouts(
    device: &crate::backend::vulkan::Device,
    bindings: &[ComputeBindingLayout],
) -> Result<Vec<DescriptorSetLayout>, Error> {
    if bindings.is_empty() {
        return Ok(Vec::new());
    }
    let max_set = bindings
        .iter()
        .map(|binding| binding.set)
        .max()
        .unwrap_or(0);
    let mut layouts = Vec::with_capacity(max_set as usize + 1);
    for set in 0..=max_set {
        let set_bindings: Vec<_> = bindings
            .iter()
            .filter(|binding| binding.set == set)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding.binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        layouts.push(DescriptorSetLayout::new(device, &set_bindings)?);
    }
    Ok(layouts)
}
