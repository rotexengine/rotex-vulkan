use std::collections::HashMap;
use std::ffi::CString;

use ash::vk;

use super::VulkanBridge;
use super::pass_targets::PassTargetCache;
use crate::backend::vulkan::{
    CommandPool, DescriptorPool, DescriptorSetLayout, DeviceDescriptor, Fence,
    GraphicsPipelineLayout, QueueCategory as BackendQueueCategory,
    QueueRequest as BackendQueueRequest, RotexBuffer, VulkanInstance,
};
use crate::core::InstanceOptions;
use crate::error::{Error, ErrorKind};

use rotex_types::{
    DeviceDescriptor as FrontendDeviceDescriptor, DeviceFeatures as FrontendDeviceFeatures,
    Extent2D as FrontendExtent2D, InstanceDescriptor as FrontendInstanceDescriptor, QueueCategory,
};

pub(super) const GLOBAL_UBO_SIZE: vk::DeviceSize = 128;
pub(super) const OBJECT_MATRIX_SIZE: usize = 64;
pub(super) const INITIAL_OBJECT_CAPACITY: u32 = 256;

impl VulkanBridge {
    pub fn new(
        instance_descriptor: FrontendInstanceDescriptor,
        device_descriptor: FrontendDeviceDescriptor,
    ) -> Result<Self, Error> {
        let options = InstanceOptions {
            enable_validation: instance_descriptor.enable_validation,
            enable_debug_utils: instance_descriptor.enable_validation,
            ..Default::default()
        };
        let extension_names = instance_descriptor
            .required_instance_extensions
            .iter()
            .map(|name| {
                CString::new(name.as_str()).map_err(|_| {
                    Error::fatal(ErrorKind::Unsupported(
                        "Instance extension contains interior NUL byte",
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let extension_ptrs = extension_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let instance = VulkanInstance::new(&options, &extension_ptrs)?;
        let backend_desc = DeviceDescriptor {
            enable_swapchain: device_descriptor.enable_swapchain,
            queues: device_descriptor
                .queues
                .into_iter()
                .map(map_queue_request)
                .collect(),
            required_features: map_device_features(device_descriptor.required_features),
        };
        let device = instance.request_device(backend_desc)?;
        let command_pool = CommandPool::new(device.raw())?;
        let mut command_buffers = command_pool.allocate_buffers(device.raw(), 1)?;
        let command_buffer = command_buffers.pop().expect("one command buffer");
        let in_flight_fence = Fence::new(device.raw(), true)?;

        let global_set_layout = DescriptorSetLayout::new(
            device.raw(),
            &[vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)],
        )?;
        let material_set_layout = DescriptorSetLayout::new(
            device.raw(),
            &[vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)],
        )?;
        let object_set_layout = DescriptorSetLayout::new(
            device.raw(),
            &[vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX)],
        )?;

        let material_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 4096,
        }];
        let material_descriptor_pool =
            DescriptorPool::new(device.raw(), 4096, &material_pool_sizes)?;

        let uniform_pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                descriptor_count: 2,
            },
        ];
        let uniform_descriptor_pool = DescriptorPool::new(device.raw(), 4, &uniform_pool_sizes)?;

        let storage_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 4096,
        }];
        let storage_descriptor_pool = DescriptorPool::new(device.raw(), 1024, &storage_pool_sizes)?;

        let global_uniform_buffer = RotexBuffer::new(
            instance.raw(),
            device.raw(),
            GLOBAL_UBO_SIZE,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let object_aligned_stride = device.raw().pad_uniform_buffer_size(OBJECT_MATRIX_SIZE) as u32;
        let object_buffer_capacity = INITIAL_OBJECT_CAPACITY;
        let object_buffer_size =
            object_aligned_stride as vk::DeviceSize * object_buffer_capacity as vk::DeviceSize;
        let object_uniform_buffer = RotexBuffer::new(
            instance.raw(),
            device.raw(),
            object_buffer_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let uniform_layouts = [global_set_layout.handle(), object_set_layout.handle()];
        let mut uniform_sets =
            uniform_descriptor_pool.allocate_sets(device.raw(), &uniform_layouts)?;
        let object_descriptor_set = uniform_sets.pop().expect("object descriptor set");
        let global_descriptor_set = uniform_sets.pop().expect("global descriptor set");

        global_descriptor_set.write_buffer(
            device.raw(),
            0,
            &global_uniform_buffer,
            0,
            GLOBAL_UBO_SIZE,
            vk::DescriptorType::UNIFORM_BUFFER,
        );
        object_descriptor_set.write_buffer(
            device.raw(),
            0,
            &object_uniform_buffer,
            0,
            object_aligned_stride as vk::DeviceSize,
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        );

        let shared_pipeline_layout = GraphicsPipelineLayout::new(
            device.raw(),
            &[
                global_set_layout.handle(),
                material_set_layout.handle(),
                object_set_layout.handle(),
            ],
            &[],
        )?;

        let graphics_queue_index = device
            .raw()
            .queues()
            .iter()
            .find(|q| q.category == BackendQueueCategory::Graphics)
            .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?
            .family_index;

        let mut bridge = Self {
            instance,
            device,
            command_pool,
            command_buffer,
            in_flight_fence,
            graphics_queue_index,
            surface_state: None,
            meshes: HashMap::new(),
            materials: HashMap::new(),
            textures: HashMap::new(),
            default_texture: None,
            material_descriptor_pool,
            uniform_descriptor_pool,
            global_set_layout,
            material_set_layout,
            object_set_layout,
            global_uniform_buffer,
            object_uniform_buffer,
            global_descriptor_set,
            object_descriptor_set,
            object_aligned_stride,
            object_buffer_capacity,
            shared_pipeline_layout,
            pass_target_cache: PassTargetCache::new(),
            material_pipelines: HashMap::new(),
            pipelines_by_material: HashMap::new(),
            vertex_layouts: HashMap::new(),
            buffers: HashMap::new(),
            compute_pipelines: HashMap::new(),
            storage_descriptor_pool,
            next_mesh_id: 1,
            next_material_id: 1,
            next_texture_id: 1,
            next_buffer_id: 1,
            next_compute_pipeline_id: 1,
        };
        bridge.ensure_default_texture()?;
        Ok(bridge)
    }

    pub(super) fn destroy_uniform_resources(&mut self) {
        self.shared_pipeline_layout.destroy(self.device.raw());
        self.global_uniform_buffer.destroy(self.device.raw());
        self.object_uniform_buffer.destroy(self.device.raw());
        self.uniform_descriptor_pool.destroy(self.device.raw());
    }
}

fn map_queue_request(req: rotex_types::QueueRequest) -> BackendQueueRequest {
    BackendQueueRequest {
        category: match req.category {
            QueueCategory::Graphics => BackendQueueCategory::Graphics,
            QueueCategory::Compute => BackendQueueCategory::Compute,
            QueueCategory::Transfer => BackendQueueCategory::Transfer,
        },
        count: req.count,
    }
}

fn map_device_features(features: FrontendDeviceFeatures) -> vk::PhysicalDeviceFeatures {
    let mut mapped = vk::PhysicalDeviceFeatures::default();
    mapped.sampler_anisotropy = features.sampler_anisotropy as u32;
    mapped.fill_mode_non_solid = features.fill_mode_non_solid as u32;
    mapped.wide_lines = features.wide_lines as u32;
    mapped
}

pub(super) fn to_vk_extent(extent: FrontendExtent2D) -> vk::Extent2D {
    let extent = extent.clamped();
    vk::Extent2D {
        width: extent.width,
        height: extent.height,
    }
}
