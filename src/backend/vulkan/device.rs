use std::collections::BTreeMap;
use std::ffi::CStr;

use ash::vk;

use crate::core::Instance;
use crate::error::{Error, ErrorKind, Severity, vk_error};

/// Queue family category for device creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueCategory {
    /// Graphics queue family.
    Graphics,
    /// Compute queue family.
    Compute,
    /// Transfer queue family.
    Transfer,
}

/// Requested queues of a given [`QueueCategory`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueRequest {
    /// Queue family category.
    pub category: QueueCategory,
    /// Number of queues to create in that family.
    pub count: u32,
}

/// Parameters for [`Adapter::request_device`].
#[derive(Debug, Clone)]
pub struct DeviceDescriptor {
    /// Features enabled on the logical device.
    pub required_features: vk::PhysicalDeviceFeatures,
    /// Whether to enable `VK_KHR_swapchain`.
    pub enable_swapchain: bool,
    /// Queue allocations requested at device creation.
    pub queues: Vec<QueueRequest>,
}

/// Vulkan physical device for selection and logical device creation.
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

    /// Human-readable device name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `VkPhysicalDeviceType` value.
    pub fn device_type(&self) -> vk::PhysicalDeviceType {
        self.device_type
    }

    /// Physical device limits.
    pub fn limits(&self) -> &vk::PhysicalDeviceLimits {
        &self.limits
    }

    /// `VkPhysicalDevice` handle.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.handle
    }

    /// Heuristic score for adapter ranking (higher is preferred).
    pub fn selection_score(&self) -> u32 {
        match self.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 400,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 300,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 200,
            vk::PhysicalDeviceType::CPU => 100,
            _ => 0,
        }
    }

    /// Reports whether `VK_KHR_swapchain` is available on this adapter.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if extension enumeration fails.
    pub fn has_swapchain_extension(&self, instance: &Instance) -> Result<bool, Error> {
        let extensions = unsafe {
            instance
                .instance()
                .enumerate_device_extension_properties(self.handle)
        }
        .map_err(vk_error)?;
        Ok(extensions.iter().any(|ext| unsafe {
            CStr::from_ptr(ext.extension_name.as_ptr()) == vk::KHR_SWAPCHAIN_NAME
        }))
    }

    /// Returns whether `queues` can be satisfied by this adapter's families.
    pub fn supports_queue_requests(&self, instance: &Instance, queues: &[QueueRequest]) -> bool {
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

        let mut has_request = false;
        for request in queues {
            if request.count == 0 {
                continue;
            }
            has_request = true;
            let family_index = match request.category {
                QueueCategory::Graphics => graphics_index,
                QueueCategory::Compute => compute_index,
                QueueCategory::Transfer => transfer_dedicated_index
                    .or(graphics_index)
                    .or(transfer_any_index)
                    .or(compute_index),
            };
            if family_index.is_none() {
                return false;
            }
        }
        has_request
    }

    /// Creates a [`Device`] from `desc` on this adapter.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if swapchain, queues, or device creation cannot be satisfied.
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

/// Queue allocation recorded at device creation.
#[derive(Debug, Clone)]
pub struct QueueAllocation {
    /// Queue family category.
    pub category: QueueCategory,
    /// Vulkan queue family index.
    pub family_index: u32,
    /// Number of queues created in that family.
    pub count: u32,
}

/// Created Vulkan logical device with queue layout metadata.
pub struct Device {
    pub(crate) handle: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    properties: vk::PhysicalDeviceProperties,
    queues: Vec<QueueAllocation>,
}

impl Device {
    /// `ash::Device` reference.
    pub fn logical_device(&self) -> &ash::Device {
        &self.device
    }

    /// `VkPhysicalDevice` handle.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.handle
    }

    /// Physical device properties.
    pub fn properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.properties
    }

    /// Queue allocations created with this device.
    pub fn queues(&self) -> &[QueueAllocation] {
        &self.queues
    }

    /// Obtains a `VkQueue` for `family_index` and `queue_index`.
    pub fn get_queue(&self, family_index: u32, queue_index: u32) -> vk::Queue {
        unsafe { self.device.get_device_queue(family_index, queue_index) }
    }

    /// Finds a memory type index matching `type_filter` and `properties`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if no compatible memory type exists.
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

    /// Rounds `original_size` up to the minimum uniform buffer offset alignment.
    pub fn pad_uniform_buffer_size(&self, original_size: usize) -> usize {
        let min_alignment = self.properties.limits.min_uniform_buffer_offset_alignment as usize;
        let mut aligned_size = original_size;

        if min_alignment > 0 {
            aligned_size = (aligned_size + min_alignment - 1) & !(min_alignment - 1);
        }

        aligned_size
    }

    /// Destroys the logical device.
    pub fn destroy(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
