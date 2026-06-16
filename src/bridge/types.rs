#![allow(dead_code)]
use ash::vk;

use crate::backend::vulkan::{
    ComputePipeline, Device, Framebuffer, GraphicsPipeline,
    GraphicsPipelineLayout, RenderPass, RotexBuffer, RotexImage, Semaphore,
    VulkanSurface, VulkanSwapchain,
};
use rotex_types::resource::{
    ComputePipelineDescriptor, MaterialDescriptor, MaterialId, TextureDescriptor,
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
}

impl TextureResource {
    pub(super) fn destroy(self, device: &Device) {
        self.image.destroy(device);
    }
}

pub(super) struct MaterialPipeline {
    pub(super) layout: GraphicsPipelineLayout,
    pub(super) pipeline: GraphicsPipeline,
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
    pub(super) color_targets: RenderTargets,
    pub(super) depth_targets: Option<RenderTargets>,
    pub(super) image_available: Semaphore,
    pub(super) render_finished: Vec<Semaphore>,
}

pub(super) struct BindGroupLayoutResource {
    pub(super) layout: crate::backend::vulkan::DescriptorSetLayout,
    pub(super) desc: rotex_types::resource::BindGroupLayoutDescriptor,
}

pub(super) struct BindGroupResource {
    pub(super) descriptor_set: crate::backend::vulkan::DescriptorSet,
}

pub(super) struct BufferResource {
    pub(super) buffer: RotexBuffer,
    pub(super) size: u64,
}

pub(super) struct ComputePipelineResource {
    pub(super) descriptor: ComputePipelineDescriptor,
    pub(super) pipeline: ComputePipeline,
    pub(super) pipeline_layout: GraphicsPipelineLayout,
    pub(super) set_layouts: Vec<crate::backend::vulkan::DescriptorSetLayout>,
    pub(super) descriptor_sets: Vec<crate::backend::vulkan::DescriptorSet>,
}

impl ComputePipelineResource {
    pub(super) fn destroy(
        self,
        device: &Device,
        storage_pool_mgr: &crate::backend::vulkan::DescriptorPoolManager,
    ) {
        for set in self.descriptor_sets {
            let _ = storage_pool_mgr.free_sets(device, &[set]);
        }
        for layout in self.set_layouts {
            layout.destroy(device);
        }
        self.pipeline_layout.destroy(device);
        self.pipeline.destroy(device);
    }
}
