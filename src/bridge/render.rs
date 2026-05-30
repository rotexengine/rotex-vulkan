use ash::vk;

use super::{VulkanBridge, surface_not_attached_error};
use super::types::DepthMode;
use crate::backend::vulkan::{
    Device, Framebuffer, FramebufferBuilder, ImageDescriptor, RenderPass, RenderPassBuilder,
    RotexImage, Swapchain, SubpassBlueprint,
};
use crate::core::Instance;
use crate::error::{Error, ErrorKind, vk_error};
use rotex_types::{
    FrameDescriptor as FrontendFrameDescriptor, MeshInstanceDescriptor,
    SceneDescriptor as FrontendSceneDescriptor,
};

impl VulkanBridge {
    pub fn render(
        &mut self,
        scene: &FrontendSceneDescriptor,
        frame: &FrontendFrameDescriptor,
    ) -> Result<(), Error> {
        if frame.passes.is_empty() {
            return Err(Error::fatal(ErrorKind::NoCompatibleDevice));
        }
        if self.surface_state.is_none() {
            return Err(surface_not_attached_error());
        }

        self.in_flight_fence.wait(self.device.raw(), u64::MAX)?;
        let image_index = match self
            .surface_state
            .as_ref()
            .expect("checked")
            .swapchain
            .raw()
            .acquire_next_image(&self.surface_state.as_ref().expect("checked").image_available)
        {
            Ok((index, _)) => index,
            Err(err) if is_swapchain_outdated(&err) => {
                self.recreate_swapchain()?;
                return Ok(());
            }
            Err(err) => return Err(err),
        };
        self.in_flight_fence.reset(self.device.raw())?;

        unsafe {
            self.device.raw().logical_device().reset_command_buffer(
                self.command_buffer.handle(),
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .map_err(vk_error)?;
        self.command_buffer.begin(
            self.device.raw(),
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        )?;

        for pass in &frame.passes {
            let draw_list: Vec<usize> = if pass.instance_indices.is_empty() {
                (0..scene.instances.len()).collect()
            } else {
                pass.instance_indices
                    .iter()
                    .copied()
                    .filter(|idx| *idx < scene.instances.len())
                    .collect()
            };

            let pass_uses_depth = pass.clear_depth.is_some()
                || draw_list.iter().any(|idx| {
                    let inst = scene.instances[*idx];
                    self.materials
                        .get(&inst.material)
                        .map(|m| m.descriptor.enable_depth)
                        .unwrap_or(false)
                });

            if pass_uses_depth {
                self.ensure_depth_targets()?;
            }
            let mut clear_values = vec![vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: pass.clear_color,
                },
            }];
            if pass_uses_depth {
                clear_values.push(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: pass.clear_depth.unwrap_or(1.0),
                        stencil: 0,
                    },
                });
            }

            let active_render_pass = {
                let state = self.surface_state.as_ref().expect("checked");
                let targets = if pass_uses_depth {
                    state
                        .depth_targets
                        .as_ref()
                        .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?
                } else {
                    &state.color_targets
                };

                if (image_index as usize) >= targets.framebuffers.len() {
                    return Err(Error::fatal(ErrorKind::NoCompatibleDevice));
                }

                self.command_buffer.begin_render_pass(
                    self.device.raw(),
                    &targets.render_pass,
                    &targets.framebuffers[image_index as usize],
                    &clear_values,
                );
                targets.render_pass.handle()
            };

            for idx in draw_list {
                let instance = scene.instances[idx];
                self.record_instance_draw(instance, pass_uses_depth, active_render_pass)?;
            }

            self.command_buffer.end_render_pass(self.device.raw());
        }

        self.command_buffer.end(self.device.raw())?;
        let queue = self.device.raw().get_queue(self.graphics_queue_index, 0);
        let signal = self
            .surface_state
            .as_ref()
            .expect("checked")
            .render_finished[image_index as usize]
            .handle();
        let wait = self
            .surface_state
            .as_ref()
            .expect("checked")
            .image_available
            .handle();
        let wait_semaphores = [wait];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.command_buffer.handle()];
        let signal_semaphores = [signal];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        unsafe {
            self.device.raw().logical_device().queue_submit(
                queue,
                &[submit],
                self.in_flight_fence.handle(),
            )
        }
        .map_err(vk_error)?;

        let present_result = {
            let state = self.surface_state.as_ref().expect("checked");
            state
                .swapchain
                .raw()
                .present(queue, image_index, &state.render_finished[image_index as usize])
        };
        match present_result {
            Ok(_) => Ok(()),
            Err(err) if is_swapchain_outdated(&err) => self.recreate_swapchain(),
            Err(err) => Err(err),
        }
    }

    pub(super) fn record_instance_draw(
        &mut self,
        instance: MeshInstanceDescriptor,
        pass_uses_depth: bool,
        active_render_pass: vk::RenderPass,
    ) -> Result<(), Error> {
        let (material_depth_enabled, material_texture) = {
            let material = self
                .materials
                .get(&instance.material)
                .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?;
            (material.descriptor.enable_depth, material.descriptor.texture)
        };
        let depth_for_material = pass_uses_depth && material_depth_enabled;
        let mesh_layout_id = self
            .meshes
            .get(&instance.mesh)
            .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?
            .vertex_layout_id;
        let depth_mode = if depth_for_material {
            DepthMode::Enabled
        } else {
            DepthMode::Disabled
        };
        let (pipeline, pipeline_layout) = self.pipeline_handle_for(
            instance.material,
            mesh_layout_id,
            depth_mode,
            active_render_pass,
        )?;
        self.command_buffer
            .bind_graphics_pipeline(self.device.raw(), pipeline);
        let descriptor_set = if let Some(texture_id) = material_texture {
            if let Some(texture) = self.textures.get(&texture_id) {
                texture.descriptor_set.handle()
            } else {
                self.ensure_default_texture()?.descriptor_set.handle()
            }
        } else {
            self.ensure_default_texture()?.descriptor_set.handle()
        };
        self.command_buffer.bind_graphics_descriptor_sets(
            self.device.raw(),
            pipeline_layout,
            0,
            &[descriptor_set],
        );
        let mesh = self
            .meshes
            .get(&instance.mesh)
            .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?;
        self.command_buffer
            .bind_vertex_buffer(self.device.raw(), mesh.vertex_buffer.handle());
        self.command_buffer.bind_index_buffer(
            self.device.raw(),
            &mesh.index_buffer,
            0,
            mesh.index_type,
        );
        self.command_buffer
            .draw_indexed(self.device.raw(), mesh.index_count, 1, 0, 0, 0);
        Ok(())
    }
}

pub(super) fn create_render_pass(
    device: &Device,
    format: vk::Format,
    depth_format: Option<vk::Format>,
) -> Result<RenderPass, Error> {
    let mut builder = RenderPassBuilder::new().with_attachment(
        vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
    );

    let subpass = if let Some(depth_format) = depth_format {
        builder = builder.with_attachment(
            vk::AttachmentDescription::default()
                .format(depth_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        );
        SubpassBlueprint {
            color_attachments: vec![0],
            depth_attachment: Some(1),
        }
    } else {
        SubpassBlueprint {
            color_attachments: vec![0],
            depth_attachment: None,
        }
    };

    builder.with_subpass(subpass).build(device).map_err(vk_error)
}

pub(super) fn build_framebuffers(
    device: &Device,
    swapchain: &Swapchain,
    render_pass: vk::RenderPass,
    depth_image: Option<&RotexImage>,
) -> Result<Vec<Framebuffer>, Error> {
    swapchain
        .image_views()
        .iter()
        .map(|view| {
            let mut builder = FramebufferBuilder::new().with_attachment(*view);
            if let Some(depth_image) = depth_image {
                builder = builder.with_attachment(depth_image.view());
            }
            builder
                .with_extent(swapchain.extent().width, swapchain.extent().height)
                .build(device, render_pass)
        })
        .collect()
}

pub(super) fn create_depth_image(
    instance: &Instance,
    device: &Device,
    extent: vk::Extent2D,
    format: vk::Format,
) -> Result<RotexImage, Error> {
    RotexImage::new(
        instance,
        device,
        ImageDescriptor::default(
            format,
            vk::Extent3D {
                width: extent.width.max(1),
                height: extent.height.max(1),
                depth: 1,
            },
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ),
    )
}

pub(super) fn find_depth_format(instance: &Instance, device: &Device) -> Result<vk::Format, Error> {
    let candidates = [
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];
    for format in candidates {
        let props = unsafe {
            instance
                .instance()
                .get_physical_device_format_properties(device.physical_device(), format)
        };
        if props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            return Ok(format);
        }
    }
    Err(Error::fatal(ErrorKind::NoCompatibleDevice))
}

pub(super) fn is_swapchain_outdated(err: &Error) -> bool {
    matches!(
        err.vk_result_code(),
        Some(code)
            if code == vk::Result::ERROR_OUT_OF_DATE_KHR.as_raw()
                || code == vk::Result::SUBOPTIMAL_KHR.as_raw()
    )
}
