#![allow(dead_code)]
use ash::vk;

use crate::backend::vulkan::DescriptorSetLayout;
use crate::backend::vulkan::Device;
use crate::error::{Error, ErrorKind};
use rotex_types::{
    AbstractPipelineLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    ShaderStageFlags,
};

pub fn build_set_layout(
    device: &Device,
    desc: &BindGroupLayoutDescriptor,
) -> Result<DescriptorSetLayout, Error> {
    let bindings: Vec<_> = desc
        .entries
        .iter()
        .map(map_layout_entry)
        .collect::<Result<_, _>>()?;
    DescriptorSetLayout::new(device, &bindings)
}

pub fn build_set_layouts_from_abstract_layout(
    device: &Device,
    layout: &AbstractPipelineLayout,
) -> Result<Vec<DescriptorSetLayout>, Error> {
    if layout.bind_groups.is_empty() {
        return Ok(Vec::new());
    }
    let max_set = layout
        .bind_groups
        .iter()
        .map(|group| group.set)
        .max()
        .unwrap_or(0);
    let mut layouts = Vec::with_capacity(max_set as usize + 1);
    for set in 0..=max_set {
        if let Some(desc) = layout.bind_groups.iter().find(|group| group.set == set) {
            layouts.push(build_set_layout(device, desc)?);
        } else {
            layouts.push(DescriptorSetLayout::new(device, &[])?);
        }
    }
    Ok(layouts)
}

pub fn build_material_set_layouts(
    device: &Device,
    layout: &AbstractPipelineLayout,
) -> Result<Vec<DescriptorSetLayout>, Error> {
    build_set_layouts_from_abstract_layout(device, layout)
}

fn map_layout_entry(
    entry: &BindGroupLayoutEntry,
) -> Result<vk::DescriptorSetLayoutBinding<'_>, Error> {
    Ok(vk::DescriptorSetLayoutBinding::default()
        .binding(entry.binding)
        .descriptor_type(map_binding_type(entry.ty))
        .descriptor_count(1)
        .stage_flags(map_shader_stages(entry.visibility)))
}

pub fn map_binding_type(ty: BindingType) -> vk::DescriptorType {
    match ty {
        BindingType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        BindingType::UniformBufferDynamic => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        BindingType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        BindingType::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
    }
}

pub fn map_shader_stages(stages: ShaderStageFlags) -> vk::ShaderStageFlags {
    let mut flags = vk::ShaderStageFlags::empty();
    if stages.contains(ShaderStageFlags::VERTEX) {
        flags |= vk::ShaderStageFlags::VERTEX;
    }
    if stages.contains(ShaderStageFlags::FRAGMENT) {
        flags |= vk::ShaderStageFlags::FRAGMENT;
    }
    if stages.contains(ShaderStageFlags::COMPUTE) {
        flags |= vk::ShaderStageFlags::COMPUTE;
    }
    flags
}

pub fn map_memory_location(location: rotex_types::MemoryLocation) -> vk::MemoryPropertyFlags {
    match location {
        rotex_types::MemoryLocation::CpuToGpu => {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        }
        rotex_types::MemoryLocation::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        rotex_types::MemoryLocation::GpuToCpu => {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        }
    }
}

pub fn allocation_mismatch(context: &'static str) -> Error {
    Error::fatal(ErrorKind::Unsupported(context))
}
