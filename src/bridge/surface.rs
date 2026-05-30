use super::{VulkanBridge, surface_not_attached_error};
use crate::backend::vulkan::{Device, Semaphore, Swapchain};
use crate::core::Instance;
use crate::error::{Error, vk_error};
use rotex_types::{Extent2D as FrontendExtent2D, SurfaceDescriptor as FrontendSurfaceDescriptor};

impl VulkanBridge {
    pub fn attach_surface(
        &mut self,
        surface_descriptor: FrontendSurfaceDescriptor,
    ) -> Result<(), Error> {
        self.destroy_surface_state();
        let raw_surface = unsafe {
            ash_window::create_surface(
                self.instance.raw_entry(),
                self.instance.raw_instance(),
                surface_descriptor.display_handle,
                surface_descriptor.window_handle,
                None,
            )
        }
        .map_err(vk_error)?;
        let surface = self.instance.create_surface_from_raw(raw_surface);
        let extent = super::init::to_vk_extent(surface_descriptor.extent);
        let swapchain = surface.create_swapchain(&self.instance, &self.device, extent)?;
        let color_targets =
            Self::create_targets(self.instance.raw(), self.device.raw(), swapchain.raw(), false)?;
        let image_available = Semaphore::new(self.device.raw())?;
        let render_finished =
            Self::create_render_finished(self.device.raw(), swapchain.raw().images().len())?;
        self.surface_state = Some(super::types::SurfaceState {
            surface,
            swapchain,
            extent,
            color_targets,
            depth_targets: None,
            image_available,
            render_finished,
        });
        Ok(())
    }

    pub(super) fn create_targets(
        instance: &Instance,
        device: &Device,
        swapchain: &Swapchain,
        with_depth: bool,
    ) -> Result<super::types::RenderTargets, Error> {
        let depth_format = if with_depth {
            Some(super::render::find_depth_format(instance, device)?)
        } else {
            None
        };
        let render_pass = super::render::create_render_pass(device, swapchain.format(), depth_format)?;
        let depth_image = match depth_format {
            Some(format) => Some(super::render::create_depth_image(
                instance,
                device,
                swapchain.extent(),
                format,
            )?),
            None => None,
        };
        let framebuffers = super::render::build_framebuffers(
            device,
            swapchain,
            render_pass.handle(),
            depth_image.as_ref(),
        )?;
        Ok(super::types::RenderTargets {
            render_pass,
            framebuffers,
            depth_image,
        })
    }

    pub(super) fn recreate_swapchain(&mut self) -> Result<(), Error> {
        unsafe { self.device.raw().logical_device().device_wait_idle() }.map_err(vk_error)?;
        self.destroy_all_pipelines();
        let Some(state) = self.surface_state.as_mut() else {
            return Err(surface_not_attached_error());
        };
        for sem in state.render_finished.drain(..) {
            sem.destroy(self.device.raw());
        }
        let old_swapchain_handle = state.swapchain.handle();
        let mut old_swapchain = std::mem::replace(
            &mut state.swapchain,
            state.surface.create_swapchain_with_old(
                &self.instance,
                &self.device,
                state.extent,
                old_swapchain_handle,
            )?,
        );
        old_swapchain.destroy_in_place(&self.device);

        let old_color_targets = std::mem::replace(
            &mut state.color_targets,
            Self::create_targets(
                self.instance.raw(),
                self.device.raw(),
                state.swapchain.raw(),
                false,
            )?,
        );
        old_color_targets.destroy(self.device.raw());

        if let Some(old_depth_targets) = state.depth_targets.take() {
            old_depth_targets.destroy(self.device.raw());
            state.depth_targets = Some(Self::create_targets(
                self.instance.raw(),
                self.device.raw(),
                state.swapchain.raw(),
                true,
            )?);
        }

        state.render_finished =
            Self::create_render_finished(self.device.raw(), state.swapchain.raw().images().len())?;
        Ok(())
    }

    pub(super) fn ensure_depth_targets(&mut self) -> Result<(), Error> {
        let Some(state) = self.surface_state.as_mut() else {
            return Err(surface_not_attached_error());
        };
        if state.depth_targets.is_none() {
            state.depth_targets = Some(Self::create_targets(
                self.instance.raw(),
                self.device.raw(),
                state.swapchain.raw(),
                true,
            )?);
        }
        Ok(())
    }

    pub(super) fn create_render_finished(
        device: &Device,
        count: usize,
    ) -> Result<Vec<Semaphore>, Error> {
        (0..count).map(|_| Semaphore::new(device)).collect()
    }

    pub fn resize(&mut self, extent: FrontendExtent2D) -> Result<(), Error> {
        if let Some(state) = self.surface_state.as_mut() {
            state.extent = super::init::to_vk_extent(extent);
            return self.recreate_swapchain();
        }
        Err(surface_not_attached_error())
    }

    pub fn destroy(mut self) {
        let _ = self.in_flight_fence.wait(self.device.raw(), u64::MAX);
        unsafe {
            let _ = self.device.raw().logical_device().device_wait_idle();
        }
        self.destroy_all_pipelines();
        for (_, mesh) in self.meshes.drain() {
            mesh.vertex_buffer.destroy(self.device.raw());
            mesh.index_buffer.destroy(self.device.raw());
        }
        for (_, texture) in self.textures.drain() {
            texture.destroy(self.device.raw(), &self.texture_descriptor_pool);
        }
        if let Some(default_texture) = self.default_texture.take() {
            default_texture.destroy(self.device.raw(), &self.texture_descriptor_pool);
        }
        self.destroy_surface_state();
        self.texture_descriptor_pool.destroy(self.device.raw());
        self.texture_set_layout.destroy(self.device.raw());
        self.command_pool.destroy(self.device.raw());
        self.in_flight_fence.destroy(self.device.raw());
        self.device.destroy();
        self.instance.destroy();
    }

    pub(super) fn destroy_surface_state(&mut self) {
        if let Some(state) = self.surface_state.take() {
            for sem in state.render_finished {
                sem.destroy(self.device.raw());
            }
            state.image_available.destroy(self.device.raw());
            state.color_targets.destroy(self.device.raw());
            if let Some(depth_targets) = state.depth_targets {
                depth_targets.destroy(self.device.raw());
            }
            state.swapchain.destroy(&self.device);
            state.surface.destroy();
        }
    }
}
