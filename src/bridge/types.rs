use ash::vk;

use crate::backend::vulkan::{
    ComputePipeline, DescriptorPool, DescriptorSet, DescriptorSetLayout, Device, Framebuffer,
    GraphicsPipeline, GraphicsPipelineLayout, RenderPass, RotexBuffer, RotexImage, RotexSampler,
    Semaphore, VulkanSurface, VulkanSwapchain,
};
use rotex_types::resource::{
    BufferUsage, ComputePipelineDescriptor, MaterialDescriptor, MaterialId, TextureDescriptor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct VertexLayoutId(pub(super) u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum DepthMode {
    Disabled,
    Enabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct MaterialPipelineKey {
    pub(super) material_id: MaterialId,
    pub(super) vertex_layout_id: VertexLayoutId,
    pub(super) depth_mode: DepthMode,
    pub(super) render_pass: vk::RenderPass,
}

impl DepthMode {
    pub(super) fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

pub(super) struct MeshResource {
    pub(super) vertex_buffer: RotexBuffer,
    pub(super) index_buffer: RotexBuffer,
    pub(super) index_type: vk::IndexType,
    pub(super) index_count: u32,
    pub(super) vertex_layout_id: VertexLayoutId,
}

pub(super) struct MaterialResource {
    pub(super) descriptor: MaterialDescriptor,
}

pub(super) struct TextureResource {
    pub(super) descriptor: TextureDescriptor,
    pub(super) image: RotexImage,
    pub(super) sampler: RotexSampler,
    pub(super) descriptor_set: DescriptorSet,
}

impl TextureResource {
    pub(super) fn destroy(self, device: &Device, descriptor_pool: &DescriptorPool) {
        let _ = descriptor_pool.free_sets(device, &[self.descriptor_set]);
        self.sampler.destroy(device);
        self.image.destroy(device);
    }
}

pub(super) struct MaterialPipeline {
    pub(super) pipeline: GraphicsPipeline,
}

pub(super) struct BufferResource {
    pub(super) buffer: RotexBuffer,
    pub(super) size: u64,
    pub(super) usage: BufferUsage,
}

pub(super) struct ComputePipelineResource {
    pub(super) descriptor: ComputePipelineDescriptor,
    pub(super) pipeline: ComputePipeline,
    pub(super) pipeline_layout: GraphicsPipelineLayout,
    pub(super) set_layouts: Vec<DescriptorSetLayout>,
    pub(super) descriptor_sets: Vec<DescriptorSet>,
}

impl ComputePipelineResource {
    pub(super) fn destroy(self, device: &Device, descriptor_pool: &DescriptorPool) {
        let _ = descriptor_pool.free_sets(device, &self.descriptor_sets);
        self.pipeline.destroy(device);
        self.pipeline_layout.destroy(device);
        for layout in self.set_layouts {
            layout.destroy(device);
        }
    }
}

pub(super) struct RenderTargets {
    pub(super) render_pass: RenderPass,
    pub(super) framebuffers: Vec<Framebuffer>,
    pub(super) depth_image: Option<RotexImage>,
}

impl RenderTargets {
    pub(super) fn destroy(self, device: &Device) {
        for framebuffer in self.framebuffers {
            framebuffer.destroy(device);
        }
        self.render_pass.destroy(device);
        if let Some(depth_image) = self.depth_image {
            depth_image.destroy(device);
        }
    }
}

pub(super) struct SurfaceState {
    pub(super) surface: VulkanSurface,
    pub(super) swapchain: VulkanSwapchain,
    pub(super) extent: vk::Extent2D,
    pub(super) image_available: Semaphore,
    pub(super) render_finished: Vec<Semaphore>,
}
