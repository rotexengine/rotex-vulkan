use ash::vk;

use super::init::OBJECT_MATRIX_SIZE;
use super::types::DepthMode;
use super::{VulkanBridge, surface_not_attached_error};
use crate::backend::vulkan::{
    Device, Framebuffer, FramebufferBuilder, ImageDescriptor, RenderPass, RenderPassBuilder,
    RotexBuffer, RotexImage, SubpassBlueprint, Swapchain,
};
use crate::core::Instance;
use crate::error::{Error, ErrorKind, vk_error};
use rotex_types::resource::MaterialId;
use rotex_types::{
    CameraDescriptor, ComputePassDescriptor, MeshInstanceDescriptor, PassColorTarget,
    PassDescriptor, RenderCommand, SceneDescriptor as FrontendSceneDescriptor,
};

impl VulkanBridge {
    pub fn execute(
        &mut self,
        scene: &FrontendSceneDescriptor,
        commands: &[RenderCommand],
    ) -> Result<(), Error> {
        if commands.is_empty() {
            return Err(Error::fatal(ErrorKind::NoCompatibleDevice));
        }

        let needs_swapchain = commands.iter().any(|command| {
            matches!(
                command,
                RenderCommand::DrawGraphics(pass)
                    if pass.color_target == PassColorTarget::Swapchain
            )
        });

        let image_index = if needs_swapchain {
            if self.surface_state.is_none() {
                return Err(surface_not_attached_error());
            }
            self.in_flight_fence.wait(self.device.raw(), u64::MAX)?;
            match self
                .surface_state
                .as_ref()
                .expect("checked")
                .swapchain
                .raw()
                .acquire_next_image(
                    &self
                        .surface_state
                        .as_ref()
                        .expect("checked")
                        .image_available,
                ) {
                Ok((index, _)) => index,
                Err(err) if is_swapchain_outdated(&err) => {
                    self.recreate_swapchain()?;
                    return Ok(());
                }
                Err(err) => return Err(err),
            }
        } else {
            self.in_flight_fence.wait(self.device.raw(), u64::MAX)?;
            0
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

        for (cmd_index, command) in commands.iter().enumerate() {
            let remaining = &commands[cmd_index + 1..];
            match command {
                RenderCommand::TransitionBuffer { buffer, from, to } => {
                    let buffer_resource = self
                        .buffers
                        .get(buffer)
                        .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?;
                    self.command_buffer.record_buffer_transition(
                        self.device.raw(),
                        buffer_resource.buffer.handle(),
                        *from,
                        *to,
                    );
                }
                RenderCommand::DispatchCompute(compute_pass) => {
                    self.record_compute_pass(compute_pass)?;
                }
                RenderCommand::DrawGraphics(pass) => {
                    self.record_graphics_pass(scene, pass, remaining, image_index)?;
                }
            }
        }

        self.command_buffer.end(self.device.raw())?;
        let queue = self.device.raw().get_queue(self.graphics_queue_index, 0);
        let command_buffers = [self.command_buffer.handle()];

        if needs_swapchain {
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
                state.swapchain.raw().present(
                    queue,
                    image_index,
                    &state.render_finished[image_index as usize],
                )
            };
            match present_result {
                Ok(_) => Ok(()),
                Err(err) if is_swapchain_outdated(&err) => self.recreate_swapchain(),
                Err(err) => Err(err),
            }
        } else {
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            unsafe {
                self.device.raw().logical_device().queue_submit(
                    queue,
                    &[submit],
                    self.in_flight_fence.handle(),
                )
            }
            .map_err(vk_error)?;
            Ok(())
        }
    }

    fn record_compute_pass(&mut self, pass: &ComputePassDescriptor) -> Result<(), Error> {
        let pipeline = self
            .compute_pipelines
            .get(&pass.pipeline)
            .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?;
        self.update_compute_descriptor_sets(pass.pipeline, &pass.buffer_intents)?;
        self.command_buffer
            .bind_compute_pipeline(self.device.raw(), pipeline.pipeline.handle());
        if !pipeline.descriptor_sets.is_empty() {
            let sets: Vec<_> = pipeline
                .descriptor_sets
                .iter()
                .map(|set| set.handle())
                .collect();
            self.command_buffer.bind_compute_descriptor_sets(
                self.device.raw(),
                pipeline.pipeline_layout.handle(),
                0,
                &sets,
                &[],
            );
        }
        self.command_buffer
            .dispatch_compute(self.device.raw(), pass.workgroup_count);
        Ok(())
    }

    fn record_graphics_pass(
        &mut self,
        scene: &FrontendSceneDescriptor,
        pass: &PassDescriptor,
        remaining_commands: &[RenderCommand],
        image_index: u32,
    ) -> Result<(), Error> {
        let draw_list: Vec<usize> = if pass.instance_indices.is_empty() {
            (0..scene.instances.len()).collect()
        } else {
            pass.instance_indices
                .iter()
                .copied()
                .filter(|idx| *idx < scene.instances.len())
                .collect()
        };

        let pass_uses_depth = pass.uses_depth_attachment()
            || draw_list.iter().any(|idx| {
                let inst = scene.instances[*idx];
                self.materials
                    .get(&inst.material)
                    .map(|m| m.descriptor.enable_depth)
                    .unwrap_or(false)
            });

        let resolved =
            self.resolve_pass_targets(pass, remaining_commands, pass_uses_depth, image_index)?;
        let targets = self.pass_targets_for_key(&resolved.key)?;
        let clear_values = build_clear_values(pass, pass_uses_depth);
        let active_render_pass = resolved.render_pass;
        let framebuffer_extent = targets.framebuffers[resolved.framebuffer_index].extent();

        self.command_buffer.begin_render_pass(
            self.device.raw(),
            &targets.render_pass,
            &targets.framebuffers[resolved.framebuffer_index],
            &clear_values,
        );

        self.write_global_ubo(scene.camera)?;
        self.ensure_object_buffer_capacity(draw_list.len() as u32)?;
        self.batch_write_object_ubo(&draw_list, scene)?;

        let pipeline_layout = self.shared_pipeline_layout.handle();
        self.command_buffer.bind_graphics_descriptor_sets(
            self.device.raw(),
            pipeline_layout,
            0,
            &[self.global_descriptor_set.handle()],
            &[],
        );

        let mut last_material = None;
        for (slot, idx) in draw_list.iter().enumerate() {
            let instance = scene.instances[*idx];
            last_material = Some(self.record_instance_draw(
                instance,
                pass_uses_depth,
                active_render_pass,
                framebuffer_extent,
                slot as u32,
                pipeline_layout,
                last_material,
            )?);
        }

        self.command_buffer.end_render_pass(self.device.raw());
        Ok(())
    }

    fn write_global_ubo(&mut self, camera: CameraDescriptor) -> Result<(), Error> {
        let ptr = self.global_uniform_buffer.map(self.device.raw())? as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(camera.view.as_ptr() as *const u8, ptr, 64);
            std::ptr::copy_nonoverlapping(camera.projection.as_ptr() as *const u8, ptr.add(64), 64);
        }
        self.global_uniform_buffer.unmap(self.device.raw());
        Ok(())
    }

    fn ensure_object_buffer_capacity(&mut self, required: u32) -> Result<(), Error> {
        if required <= self.object_buffer_capacity {
            return Ok(());
        }
        let new_capacity = required
            .next_power_of_two()
            .max(self.object_buffer_capacity * 2);
        let new_size =
            self.object_aligned_stride as vk::DeviceSize * new_capacity as vk::DeviceSize;
        let new_buffer = RotexBuffer::new(
            self.instance.raw(),
            self.device.raw(),
            new_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        self.object_uniform_buffer.destroy(self.device.raw());
        self.object_uniform_buffer = new_buffer;
        self.object_buffer_capacity = new_capacity;
        self.object_descriptor_set.write_buffer(
            self.device.raw(),
            0,
            &self.object_uniform_buffer,
            0,
            self.object_aligned_stride as vk::DeviceSize,
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        );
        Ok(())
    }

    fn batch_write_object_ubo(
        &mut self,
        draw_list: &[usize],
        scene: &FrontendSceneDescriptor,
    ) -> Result<(), Error> {
        let ptr = self.object_uniform_buffer.map(self.device.raw())? as *mut u8;
        for (slot, idx) in draw_list.iter().enumerate() {
            let transform = scene.instances[*idx].transform;
            let offset = slot as usize * self.object_aligned_stride as usize;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    transform.as_ptr() as *const u8,
                    ptr.add(offset),
                    OBJECT_MATRIX_SIZE,
                );
            }
        }
        self.object_uniform_buffer.unmap(self.device.raw());
        Ok(())
    }

    pub(super) fn record_instance_draw(
        &mut self,
        instance: MeshInstanceDescriptor,
        pass_uses_depth: bool,
        active_render_pass: vk::RenderPass,
        pass_extent: vk::Extent2D,
        slot: u32,
        pipeline_layout: vk::PipelineLayout,
        last_material: Option<MaterialId>,
    ) -> Result<MaterialId, Error> {
        let (material_depth_enabled, material_texture) = {
            let material = self
                .materials
                .get(&instance.material)
                .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?;
            (
                material.descriptor.enable_depth,
                material.descriptor.texture,
            )
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
        let (pipeline, _) = self.pipeline_handle_for(
            instance.material,
            mesh_layout_id,
            depth_mode,
            active_render_pass,
            pass_extent,
        )?;
        self.command_buffer
            .bind_graphics_pipeline(self.device.raw(), pipeline);

        let material_set = if let Some(texture_id) = material_texture {
            if let Some(texture) = self.textures.get(&texture_id) {
                texture.descriptor_set.handle()
            } else {
                self.ensure_default_texture()?.descriptor_set.handle()
            }
        } else {
            self.ensure_default_texture()?.descriptor_set.handle()
        };

        let object_offset = slot * self.object_aligned_stride;
        if last_material != Some(instance.material) {
            self.command_buffer.bind_graphics_descriptor_sets(
                self.device.raw(),
                pipeline_layout,
                1,
                &[material_set],
                &[],
            );
        }
        self.command_buffer.bind_graphics_descriptor_sets(
            self.device.raw(),
            pipeline_layout,
            2,
            &[self.object_descriptor_set.handle()],
            &[object_offset],
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
        Ok(instance.material)
    }
}

fn build_clear_values(pass: &rotex_types::PassDescriptor, uses_depth: bool) -> Vec<vk::ClearValue> {
    let mut clear_values = vec![vk::ClearValue {
        color: vk::ClearColorValue {
            float32: pass.clear_color,
        },
    }];
    if uses_depth {
        clear_values.push(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: pass.clear_depth,
                stencil: 0,
            },
        });
    }
    clear_values
}

pub(super) struct RenderPassConfig {
    pub color_load: vk::AttachmentLoadOp,
    pub color_final_layout: vk::ImageLayout,
    pub depth_load: Option<vk::AttachmentLoadOp>,
    pub depth_store: vk::AttachmentStoreOp,
}

pub(super) fn create_render_pass(
    device: &Device,
    format: vk::Format,
    depth_format: Option<vk::Format>,
    config: RenderPassConfig,
) -> Result<RenderPass, Error> {
    let color_initial_layout = match config.color_load {
        vk::AttachmentLoadOp::LOAD => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        _ => vk::ImageLayout::UNDEFINED,
    };
    let mut builder = RenderPassBuilder::new().with_attachment(
        vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(config.color_load)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(color_initial_layout)
            .final_layout(config.color_final_layout),
    );

    let subpass = if let Some(depth_format) = depth_format {
        let depth_load = config.depth_load.unwrap_or(vk::AttachmentLoadOp::CLEAR);
        let depth_initial_layout = match depth_load {
            vk::AttachmentLoadOp::LOAD => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            _ => vk::ImageLayout::UNDEFINED,
        };
        builder = builder.with_attachment(
            vk::AttachmentDescription::default()
                .format(depth_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(depth_load)
                .store_op(config.depth_store)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(depth_initial_layout)
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

    builder
        .with_subpass(subpass)
        .build(device)
        .map_err(vk_error)
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

pub(super) fn build_texture_framebuffer(
    device: &Device,
    render_pass: vk::RenderPass,
    color_view: vk::ImageView,
    depth_image: Option<&RotexImage>,
    width: u32,
    height: u32,
) -> Result<Vec<Framebuffer>, Error> {
    let mut builder = FramebufferBuilder::new().with_attachment(color_view);
    if let Some(depth_image) = depth_image {
        builder = builder.with_attachment(depth_image.view());
    }
    Ok(vec![
        builder
            .with_extent(width, height)
            .build(device, render_pass)?,
    ])
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
