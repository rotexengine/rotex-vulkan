use ash::vk;

use super::buffer::RotexBuffer;
use super::device::Device;
use crate::error::vk_error;
use crate::Error;

/// Allocated descriptor set handle.
pub struct DescriptorSet {
    handle: vk::DescriptorSet,
}

impl DescriptorSet {
    /// `VkDescriptorSet` handle.
    pub fn handle(&self) -> vk::DescriptorSet {
        self.handle
    }

    /// Writes a buffer binding at `binding`.
    pub fn write_buffer(
        &self,
        device: &Device,
        binding: u32,
        buffer: &RotexBuffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
        descriptor_type: vk::DescriptorType,
    ) {
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer.handle())
            .offset(offset)
            .range(range)];

        let write = [vk::WriteDescriptorSet::default()
            .dst_set(self.handle)
            .dst_binding(binding)
            .descriptor_type(descriptor_type)
            .buffer_info(&buffer_info)];

        unsafe {
            device.logical_device().update_descriptor_sets(&write, &[]);
        }
    }

    /// Writes a combined image sampler at `binding`.
    pub fn write_image_sampler(
        &self,
        device: &Device,
        binding: u32,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        image_layout: vk::ImageLayout,
    ) {
        let image_info = [vk::DescriptorImageInfo::default()
            .image_layout(image_layout)
            .image_view(image_view)
            .sampler(sampler)];

        let write = [vk::WriteDescriptorSet::default()
            .dst_set(self.handle)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info)];

        unsafe {
            device.logical_device().update_descriptor_sets(&write, &[]);
        }
    }
}

/// Pool used to allocate [`DescriptorSet`] handles.
pub struct DescriptorPool {
    handle: vk::DescriptorPool,
}

impl DescriptorPool {
    /// Creates a pool supporting up to `max_sets` with `pool_sizes`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `vkCreateDescriptorPool` fails.
    pub fn new(
        device: &Device,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> Result<Self, Error> {
        let create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        let handle = unsafe {
            device
                .logical_device()
                .create_descriptor_pool(&create_info, None)
        }
        .map_err(vk_error)?;

        Ok(Self { handle })
    }

    /// Allocates one set per entry in `layouts`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if allocation fails.
    pub fn allocate_sets(
        &self,
        device: &Device,
        layouts: &[vk::DescriptorSetLayout],
    ) -> Result<Vec<DescriptorSet>, Error> {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.handle)
            .set_layouts(layouts);

        let sets = unsafe {
            device
                .logical_device()
                .allocate_descriptor_sets(&alloc_info)
        }
        .map_err(vk_error)?;

        Ok(sets
            .into_iter()
            .map(|handle| DescriptorSet { handle })
            .collect())
    }

    /// `VkDescriptorPool` handle.
    pub fn handle(&self) -> vk::DescriptorPool {
        self.handle
    }

    /// Returns `sets` to the pool.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `vkFreeDescriptorSets` fails.
    pub fn free_sets(&self, device: &Device, sets: &[DescriptorSet]) -> Result<(), Error> {
        if sets.is_empty() {
            return Ok(());
        }
        let set_handles = sets.iter().map(|set| set.handle).collect::<Vec<_>>();
        unsafe {
            device
                .logical_device()
                .free_descriptor_sets(self.handle, &set_handles)
        }
        .map_err(vk_error)
    }

    /// Destroys the pool and all sets allocated from it.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device
                .logical_device()
                .destroy_descriptor_pool(self.handle, None);
        }
    }
}
