use ash::vk;

use super::device::{Device, QueueCategory};
use super::sync::Semaphore;
use crate::core::Instance;
use crate::error::{Error, ErrorKind, Severity, vk_error};

pub struct Surface {
    pub(crate) loader: ash::khr::surface::Instance,
    pub(crate) surface: vk::SurfaceKHR,
}

impl Surface {
    pub fn new(instance: &Instance, surface: vk::SurfaceKHR) -> Self {
        let loader = ash::khr::surface::Instance::new(instance.entry(), instance.instance());
        Self { loader, surface }
    }

    pub fn loader(&self) -> &ash::khr::surface::Instance {
        &self.loader
    }

    pub fn handle(&self) -> vk::SurfaceKHR {
        self.surface
    }

    pub fn destroy(&mut self) {
        unsafe {
            self.loader.destroy_surface(self.surface, None);
        }
    }
}

pub struct Swapchain {
    pub(crate) loader: ash::khr::swapchain::Device,
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) _surface: vk::SurfaceKHR,
    pub(crate) format: vk::Format,
    pub(crate) color_space: vk::ColorSpaceKHR,
    pub(crate) extent: vk::Extent2D,
    pub(crate) images: Vec<vk::Image>,
    pub(crate) image_views: Vec<vk::ImageView>,
}

fn clamp_extent(extent: vk::Extent2D, capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    vk::Extent2D {
        width: extent.width.clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        ),
        height: extent.height.clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        ),
    }
}

impl Swapchain {
    pub fn new(
        instance: &Instance,
        device: &Device,
        surface: &Surface,
        extent_hint: vk::Extent2D,
    ) -> Result<Self, Error> {
        Self::new_with_old(
            instance,
            device,
            surface,
            extent_hint,
            vk::SwapchainKHR::null(),
        )
    }

    pub fn new_with_old(
        instance: &Instance,
        device: &Device,
        surface: &Surface,
        extent_hint: vk::Extent2D,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<Self, Error> {
        let queue = device
            .queues()
            .iter()
            .find(|allocation| allocation.category == QueueCategory::Graphics)
            .ok_or(Error {
                kind: ErrorKind::NoCompatibleDevice,
                severity: Severity::Fatal,
            })?;

        let surface_capabilities = unsafe {
            surface.loader().get_physical_device_surface_capabilities(
                device.physical_device(),
                surface.handle(),
            )
        }
        .map_err(vk_error)?;
        let surface_formats = unsafe {
            surface
                .loader()
                .get_physical_device_surface_formats(device.physical_device(), surface.handle())
        }
        .map_err(vk_error)?;
        let present_modes = unsafe {
            surface.loader().get_physical_device_surface_present_modes(
                device.physical_device(),
                surface.handle(),
            )
        }
        .map_err(vk_error)?;

        let preferred_format = [vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_SRGB];
        let preferred_color_space = vk::ColorSpaceKHR::SRGB_NONLINEAR;
        let surface_format = surface_formats
            .iter()
            .find(|format| {
                preferred_format.contains(&format.format)
                    && format.color_space == preferred_color_space
            })
            .or_else(|| surface_formats.first())
            .ok_or(Error {
                kind: ErrorKind::Vulkan(vk::Result::ERROR_FORMAT_NOT_SUPPORTED),
                severity: Severity::Fatal,
            })?;

        let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };

        let extent = if surface_capabilities.current_extent.width != u32::MAX {
            surface_capabilities.current_extent
        } else {
            clamp_extent(extent_hint, &surface_capabilities)
        };

        let mut image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0 {
            image_count = image_count.min(surface_capabilities.max_image_count);
        }

        let queue_family_indices = [queue.family_index];
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.handle())
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        let swapchain_loader =
            ash::khr::swapchain::Device::new(&instance.instance(), &device.logical_device());
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }
            .map_err(vk_error)?;
        let images =
            unsafe { swapchain_loader.get_swapchain_images(swapchain) }.map_err(vk_error)?;

        let image_views = images
            .iter()
            .map(|image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    );
                unsafe { device.logical_device().create_image_view(&view_info, None) }
                    .map_err(vk_error)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            loader: swapchain_loader,
            swapchain,
            _surface: surface.handle(),
            format: surface_format.format,
            color_space: surface_format.color_space,
            extent,
            images,
            image_views,
        })
    }

    pub fn acquire_next_image(&self, semaphore: &Semaphore) -> Result<(u32, bool), Error> {
        unsafe {
            self.loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                semaphore.handle,
                vk::Fence::null(),
            )
        }
        .map_err(vk_error)
    }

    pub fn present(
        &self,
        queue: vk::Queue,
        image_index: u32,
        wait_semaphore: &Semaphore,
    ) -> Result<bool, Error> {
        let wait_semaphores = [wait_semaphore.handle];
        let swapchains = [self.swapchain];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe { self.loader.queue_present(queue, &present_info) }.map_err(vk_error)
    }

    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn color_space(&self) -> vk::ColorSpaceKHR {
        self.color_space
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }

    pub fn image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            for view in self.image_views.drain(..) {
                device.logical_device().destroy_image_view(view, None);
            }
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}
