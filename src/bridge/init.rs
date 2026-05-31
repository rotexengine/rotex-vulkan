use std::collections::HashMap;
use std::ffi::CString;

use ash::vk;

use super::VulkanBridge;
use crate::backend::vulkan::{
    CommandPool, DescriptorPool, DescriptorSetLayout, DeviceDescriptor, Fence,
    QueueCategory as BackendQueueCategory, QueueRequest as BackendQueueRequest, VulkanInstance,
};
use crate::core::InstanceOptions;
use crate::error::{Error, ErrorKind};
use rotex_types::{
    DeviceDescriptor as FrontendDeviceDescriptor, DeviceFeatures as FrontendDeviceFeatures,
    Extent2D as FrontendExtent2D, InstanceDescriptor as FrontendInstanceDescriptor,
    QueueCategory,
};

impl VulkanBridge {
    /// Creates a bridge from rotex instance and device descriptors.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if instance creation, device selection, or required Vulkan objects fail.
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
            queues: device_descriptor.queues.into_iter().map(map_queue_request).collect(),
            required_features: map_device_features(device_descriptor.required_features),
        };
        let device = instance.request_device(backend_desc)?;
        let command_pool = CommandPool::new(device.raw())?;
        let mut command_buffers = command_pool.allocate_buffers(device.raw(), 1)?;
        let command_buffer = command_buffers.pop().expect("one command buffer");
        let in_flight_fence = Fence::new(device.raw(), true)?;
        let texture_layout_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let texture_set_layout = DescriptorSetLayout::new(device.raw(), &texture_layout_bindings)?;
        let texture_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 4096,
        }];
        let texture_descriptor_pool = DescriptorPool::new(device.raw(), 4096, &texture_pool_sizes)?;
        let graphics_queue_index = device
            .raw()
            .queues()
            .iter()
            .find(|q| q.category == BackendQueueCategory::Graphics)
            .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?
            .family_index;
        Ok(Self {
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
            texture_descriptor_pool,
            texture_set_layout,
            material_pipelines: HashMap::new(),
            pipelines_by_material: HashMap::new(),
            vertex_layouts: HashMap::new(),
            next_mesh_id: 1,
            next_material_id: 1,
            next_texture_id: 1,
        })
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
