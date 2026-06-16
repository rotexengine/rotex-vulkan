use std::collections::HashMap;
use std::ffi::CString;

use ash::vk;

use super::VulkanBridge;
use crate::backend::vulkan::{
    general_pool_sizes, storage_pool_sizes, CommandPool, DeferredDeleteQueue,
    DescriptorPoolManager, DescriptorSetLayout, DeviceDescriptor, Fence, FrameSlot,
    QueueCategory as BackendQueueCategory, QueueRequest as BackendQueueRequest, RotexSampler,
    SamplerDescriptor, VulkanInstance,
};
use crate::core::InstanceOptions;
use crate::error::{Error, ErrorKind};
use rotex_types::{
    DeviceDescriptor as FrontendDeviceDescriptor, DeviceFeatures as FrontendDeviceFeatures,
    Extent2D as FrontendExtent2D, InstanceDescriptor as FrontendInstanceDescriptor,
    QueueCategory,
};

const FRAMES_IN_FLIGHT: u32 = 2;

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
            queues: device_descriptor.queues.into_iter().map(map_queue_request).collect(),
            required_features: map_device_features(device_descriptor.required_features),
        };
        let device = instance.request_device(backend_desc)?;
        let command_pool = CommandPool::new(device.raw())?;

        let mut command_buffers = command_pool.allocate_buffers(device.raw(), 1)?;
        let command_buffer = command_buffers.pop().ok_or_else(|| {
            Error::fatal(ErrorKind::Unsupported("failed to allocate command buffer"))
        })?;
        let in_flight_fence = Fence::new(device.raw(), true)?;

        let mut frame_slots = Vec::with_capacity(FRAMES_IN_FLIGHT as usize);
        for _ in 0..FRAMES_IN_FLIGHT {
            let mut buffers = command_pool.allocate_buffers(device.raw(), 1)?;
            let cb = buffers.pop().ok_or_else(|| {
                Error::fatal(ErrorKind::Unsupported("failed to allocate command buffer"))
            })?;
            let fence = Fence::new(device.raw(), true)?;
            let image_available = crate::backend::vulkan::Semaphore::new(device.raw())?;
            frame_slots.push(FrameSlot {
                command_buffer: cb,
                fence,
                image_available,
            });
        }

        let graphics_queue_index = device
            .raw()
            .queues()
            .iter()
            .find(|q| q.category == BackendQueueCategory::Graphics)
            .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?
            .family_index;

        let general_pool = DescriptorPoolManager::new(general_pool_sizes(), 256);
        let storage_pool = DescriptorPoolManager::new(storage_pool_sizes(), 4096);
        let empty_set_layout = DescriptorSetLayout::new(device.raw(), &[])?;
        let fallback_sampler = RotexSampler::new(
            device.raw(),
            SamplerDescriptor::default().with_filters(vk::Filter::LINEAR, vk::Filter::LINEAR),
        )?;

        Ok(Self {
            instance,
            device,
            command_pool,
            command_buffer,
            in_flight_fence,
            frame_slots,
            frames_in_flight: FRAMES_IN_FLIGHT,
            current_frame_index: 0,
            current_image_index: 0,
            recording: false,
            graphics_queue_index,
            surface_state: None,
            meshes: HashMap::new(),
            materials: HashMap::new(),
            textures: HashMap::new(),
            bind_group_layouts: HashMap::new(),
            bind_groups: HashMap::new(),
            general_descriptor_pool: general_pool,
            storage_descriptor_pool: storage_pool,
            empty_set_layout,
            fallback_sampler,
            pass_target_cache: super::pass_targets::PassTargetCache::new(),
            material_pipelines: HashMap::new(),
            pipelines_by_material: HashMap::new(),
            vertex_layouts: HashMap::new(),
            buffers: HashMap::new(),
            compute_pipelines: HashMap::new(),
            deferred_delete: DeferredDeleteQueue::new(),
            active_render_pass: None,
            active_pass_extent: None,
            active_pipeline_layout: None,
            active_pass: None,
            active_target_role: None,
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
