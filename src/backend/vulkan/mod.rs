mod buffer;
mod command;
mod descriptor;
mod device;
mod framebuffer;
mod graphics_pipeline;
mod image;
mod pass;
mod swapchain;
mod sync;

pub use buffer::RotexBuffer;
pub use command::{CommandBuffer, CommandPool};
pub use descriptor::{DescriptorPool, DescriptorSet};
pub use device::{Adapter, Device, DeviceDescriptor, QueueAllocation, QueueCategory, QueueRequest};
pub use framebuffer::{Framebuffer, FramebufferBuilder};
pub use graphics_pipeline::{
    ColorBlendAttachmentState, ColorBlendState, DepthStencilState, DescriptorSetLayout,
    GraphicsPipeline, GraphicsPipelineBuilder, GraphicsPipelineLayout, RasterizationState,
    ShaderModule, ShaderStageDescriptor, Vertex, VertexInputDescriptor,
};
pub use image::{ImageDescriptor, RotexImage, RotexSampler, SamplerDescriptor};
pub use pass::{RenderPass, RenderPassBuilder, SubpassBlueprint};
pub use swapchain::{Surface, Swapchain};
pub use sync::{Fence, Semaphore};

use ash::vk;
use std::cmp::Reverse;

use crate::core::{DebugMessenger, Instance, InstanceOptions};
use crate::error::{Error, ErrorKind, Severity};

pub(crate) struct VulkanInstance {
    raw: Instance,
    debug: Option<DebugMessenger>,
}

impl VulkanInstance {
    pub(crate) fn new(options: &InstanceOptions, extensions: &[*const i8]) -> Result<Self, Error> {
        let (raw, debug) = Instance::new_with_options(options, extensions)?;
        Ok(Self { raw, debug })
    }

    pub(crate) fn request_device(&self, desc: DeviceDescriptor) -> Result<VulkanDevice, Error> {
        let mut adapters = self.raw.enumerate_adapters();
        if adapters.is_empty() {
            return Err(Error {
                kind: ErrorKind::NoCompatibleDevice,
                severity: Severity::Fatal,
            });
        }
        adapters.sort_by_key(|adapter| Reverse(adapter.selection_score()));

        let mut last_error = None;
        for adapter in adapters {
            if !adapter.supports_queue_requests(&self.raw, &desc.queues) {
                continue;
            }
            if desc.enable_swapchain && !adapter.has_swapchain_extension(&self.raw)? {
                continue;
            }
            match adapter.request_device(&self.raw, desc.clone()) {
                Ok(device) => {
                    eprintln!(
                        "[vulkan] selected adapter: {} ({:?})",
                        adapter.name(),
                        adapter.device_type()
                    );
                    return Ok(VulkanDevice { raw: device });
                }
                Err(err) => {
                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or(Error {
            kind: ErrorKind::NoCompatibleDevice,
            severity: Severity::Fatal,
        }))
    }

    pub(crate) fn create_surface_from_raw(&self, raw_surface: vk::SurfaceKHR) -> VulkanSurface {
        VulkanSurface {
            raw: Surface::new(&self.raw, raw_surface),
        }
    }

    pub(crate) fn raw_entry(&self) -> &ash::Entry {
        self.raw.entry()
    }

    pub(crate) fn raw_instance(&self) -> &ash::Instance {
        self.raw.instance()
    }

    pub(crate) fn raw(&self) -> &Instance {
        &self.raw
    }

    pub(crate) fn destroy(mut self) {
        if let Some(debug) = self.debug.take() {
            debug.destroy();
        }
        self.raw.destroy();
    }
}

pub(crate) struct VulkanDevice {
    raw: Device,
}

impl VulkanDevice {
    pub(crate) fn raw(&self) -> &Device {
        &self.raw
    }

    pub(crate) fn destroy(mut self) {
        self.raw.destroy();
    }
}

pub(crate) struct VulkanSurface {
    raw: Surface,
}

impl VulkanSurface {
    pub(crate) fn create_swapchain(
        &self,
        instance: &VulkanInstance,
        device: &VulkanDevice,
        extent_hint: vk::Extent2D,
    ) -> Result<VulkanSwapchain, Error> {
        let swapchain = Swapchain::new(&instance.raw, device.raw(), &self.raw, extent_hint)?;
        Ok(VulkanSwapchain { raw: swapchain })
    }

    pub(crate) fn create_swapchain_with_old(
        &self,
        instance: &VulkanInstance,
        device: &VulkanDevice,
        extent_hint: vk::Extent2D,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<VulkanSwapchain, Error> {
        let swapchain = Swapchain::new_with_old(
            instance.raw(),
            device.raw(),
            &self.raw,
            extent_hint,
            old_swapchain,
        )?;
        Ok(VulkanSwapchain { raw: swapchain })
    }

    pub(crate) fn destroy(mut self) {
        self.raw.destroy();
    }
}

pub(crate) struct VulkanSwapchain {
    raw: Swapchain,
}

impl VulkanSwapchain {
    pub(crate) fn raw(&self) -> &Swapchain {
        &self.raw
    }

    pub(crate) fn destroy(mut self, device: &VulkanDevice) {
        self.raw.destroy(device.raw());
    }

    pub(crate) fn destroy_in_place(&mut self, device: &VulkanDevice) {
        self.raw.destroy(device.raw());
    }

    pub(crate) fn handle(&self) -> vk::SwapchainKHR {
        self.raw.swapchain
    }
}
