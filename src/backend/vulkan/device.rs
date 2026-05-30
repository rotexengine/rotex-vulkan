use std::collections::BTreeMap;
use std::ffi::CStr;

use ash::vk;

use crate::core::Instance;
use crate::error::{Error, ErrorKind, Severity, vk_error};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueCategory {
    Graphics,
    Compute,
    Transfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueRequest {
    pub category: QueueCategory,
    pub count: u32,
}

#[derive(Debug, Clone)]
pub struct DeviceDescriptor {
    pub required_features: vk::PhysicalDeviceFeatures,
    pub enable_swapchain: bool,
    pub queues: Vec<QueueRequest>,
}

pub struct Adapter {
    pub(crate) handle: vk::PhysicalDevice,
    name: String,
    device_type: vk::PhysicalDeviceType,
    limits: vk::PhysicalDeviceLimits,
}

impl Adapter {
    pub(crate) fn new(
        handle: vk::PhysicalDevice,
        name: String,
        device_type: vk::PhysicalDeviceType,
        limits: vk::PhysicalDeviceLimits,
    ) -> Self {
        Self {
            handle,
            name,
            device_type,
            limits,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn device_type(&self) -> vk::PhysicalDeviceType {
        self.device_type
    }

    pub fn limits(&self) -> &vk::PhysicalDeviceLimits {
        &self.limits
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.handle
    }

    pub fn request_device(
        &self,
        instance: &Instance,
        desc: DeviceDescriptor,
    ) -> Result<Device, Error> {
        let queue_families = unsafe {
            instance
                .instance()
                .get_physical_device_queue_family_properties(self.handle)
        };
        let graphics_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(index, _)| index as u32);
        let compute_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(index, _)| index as u32);
        let transfer_any_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::TRANSFER))
            .map(|(index, _)| index as u32);
        let transfer_dedicated_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| {
                family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && !family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            })
            .map(|(index, _)| index as u32);

        if desc.enable_swapchain {
            let extensions = unsafe {
                instance
                    .instance()
                    .enumerate_device_extension_properties(self.handle)
            }
            .map_err(vk_error)?;
            let has_swapchain = extensions.iter().any(|ext| unsafe {
                CStr::from_ptr(ext.extension_name.as_ptr()) == vk::KHR_SWAPCHAIN_NAME
            });
            if !has_swapchain {
                return Err(Error {
                    kind: ErrorKind::NoCompatibleDevice,
                    severity: Severity::Fatal,
                });
            }
        }

        let mut allocations = Vec::new();
        for request in desc.queues {
            if request.count == 0 {
                continue;
            }
            let family_index = match request.category {
                QueueCategory::Graphics => graphics_index,
                QueueCategory::Compute => compute_index,
                QueueCategory::Transfer => transfer_dedicated_index
                    .or(graphics_index)
                    .or(transfer_any_index)
                    .or(compute_index),
            };
            let family_index = match family_index {
                Some(index) => index,
                None => {
                    return Err(Error {
                        kind: ErrorKind::NoCompatibleDevice,
                        severity: Severity::Fatal,
                    });
                }
            };
            allocations.push(QueueAllocation {
                category: request.category,
                family_index,
                count: request.count,
            });
        }

        if allocations.is_empty() {
            return Err(Error {
                kind: ErrorKind::NoCompatibleDevice,
                severity: Severity::Fatal,
            });
        }

        let mut queue_priorities: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
        for allocation in &allocations {
            let entry = queue_priorities
                .entry(allocation.family_index)
                .or_insert_with(Vec::new);
            entry.extend(std::iter::repeat(1.0).take(allocation.count as usize));
        }

        for (family_index, priorities) in queue_priorities.iter_mut() {
            let max_supported = queue_families[*family_index as usize].queue_count as usize;
            if priorities.len() > max_supported {
                priorities.truncate(max_supported);
            }
        }

        let mut priorities_store = Vec::new();
        let mut queue_layouts = Vec::new();
        for (family_index, priorities) in queue_priorities {
            priorities_store.push(priorities);
            let idx = priorities_store.len() - 1;
            queue_layouts.push((family_index, idx));
        }

        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = queue_layouts
            .into_iter()
            .map(|(family_index, idx)| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family_index)
                    .queue_priorities(&priorities_store[idx])
            })
            .collect();

        let device_extensions: Vec<*const i8> = if desc.enable_swapchain {
            vec![vk::KHR_SWAPCHAIN_NAME.as_ptr()]
        } else {
            Vec::new()
        };
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&desc.required_features);

        let device = unsafe {
            instance
                .instance()
                .create_device(self.handle, &device_create_info, None)
        }
        .map_err(vk_error)?;

        let properties = unsafe {
            instance
                .instance()
                .get_physical_device_properties(self.handle)
        };

        Ok(Device {
            handle: self.handle,
            device,
            properties,
            queues: allocations,
        })
    }
}

#[derive(Debug, Clone)]
pub struct QueueAllocation {
    pub category: QueueCategory,
    pub family_index: u32,
    pub count: u32,
}

pub struct Device {
    pub(crate) handle: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    properties: vk::PhysicalDeviceProperties,
    queues: Vec<QueueAllocation>,
}

impl Device {
    pub fn logical_device(&self) -> &ash::Device {
        &self.device
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.handle
    }

    pub fn properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.properties
    }

    pub fn queues(&self) -> &[QueueAllocation] {
        &self.queues
    }

    pub fn get_queue(&self, family_index: u32, queue_index: u32) -> vk::Queue {
        unsafe { self.device.get_device_queue(family_index, queue_index) }
    }

    pub fn find_memory_type(
        &self,
        instance: &Instance,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, Error> {
        let memory_properties = unsafe {
            instance
                .instance()
                .get_physical_device_memory_properties(self.physical_device())
        };

        for (index, memory_type) in memory_properties.memory_types.iter().enumerate() {
            let is_allowed_by_hardware = (type_filter & (1 << index)) != 0;
            let has_required_properties = memory_type.property_flags.contains(properties);

            if is_allowed_by_hardware && has_required_properties {
                return Ok(index as u32);
            }
        }

        Err(Error::fatal(ErrorKind::NoCompatibleDevice))
    }

    pub fn pad_uniform_buffer_size(&self, original_size: usize) -> usize {
        let min_alignment = self.properties.limits.min_uniform_buffer_offset_alignment as usize;
        let mut aligned_size = original_size;

        if min_alignment > 0 {
            aligned_size = (aligned_size + min_alignment - 1) & !(min_alignment - 1);
        }

        aligned_size
    }

    pub fn destroy(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
