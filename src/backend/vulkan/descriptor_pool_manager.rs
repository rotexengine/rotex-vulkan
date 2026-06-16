use ash::vk;

use super::descriptor::{DescriptorPool, DescriptorSet};
use super::device::Device;
use crate::error::Error;

pub struct DescriptorPoolManager {
    pools: Vec<DescriptorPool>,
    pool_sizes: Vec<vk::DescriptorPoolSize>,
    max_sets_per_pool: u32,
}

impl DescriptorPoolManager {
    pub fn new(pool_sizes: Vec<vk::DescriptorPoolSize>, initial_max_sets: u32) -> Self {
        Self {
            pools: Vec::new(),
            pool_sizes,
            max_sets_per_pool: initial_max_sets.max(1),
        }
    }

    pub fn allocate_sets(
        &mut self,
        device: &Device,
        layouts: &[vk::DescriptorSetLayout],
    ) -> Result<Vec<DescriptorSet>, Error> {
        if layouts.is_empty() {
            return Ok(Vec::new());
        }

        if self.pools.is_empty() {
            self.grow_pool(device)?;
        }

        let mut pool_index = self.pools.len() - 1;
        loop {
            match self.pools[pool_index].allocate_sets(device, layouts) {
                Ok(sets) => return Ok(sets),
                Err(err) if is_pool_exhausted(&err) => {
                    self.grow_pool(device)?;
                    pool_index = self.pools.len() - 1;
                }
                Err(err) => return Err(err),
            }
        }
    }

    pub fn free_sets(&self, device: &Device, sets: &[DescriptorSet]) -> Result<(), Error> {
        if sets.is_empty() {
            return Ok(());
        }
        for pool in &self.pools {
            if pool.free_sets(device, sets).is_ok() {
                return Ok(());
            }
        }
        Ok(())
    }

    pub fn destroy(&mut self, device: &Device) {
        for pool in self.pools.drain(..) {
            pool.destroy(device);
        }
    }

    fn grow_pool(&mut self, device: &Device) -> Result<(), Error> {
        let pool = DescriptorPool::new(device, self.max_sets_per_pool, &self.pool_sizes)?;
        self.pools.push(pool);
        self.max_sets_per_pool = self.max_sets_per_pool.saturating_mul(2).max(1);
        Ok(())
    }
}

fn is_pool_exhausted(err: &Error) -> bool {
    matches!(
        err.vk_result_code(),
        Some(code) if code == vk::Result::ERROR_OUT_OF_POOL_MEMORY.as_raw()
            || code == vk::Result::ERROR_FRAGMENTED_POOL.as_raw()
    )
}

pub fn storage_pool_sizes() -> Vec<vk::DescriptorPoolSize> {
    vec![vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 4096,
    }]
}

pub fn general_pool_sizes() -> Vec<vk::DescriptorPoolSize> {
    vec![
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 256,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: 256,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 256,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 256,
        },
    ]
}
