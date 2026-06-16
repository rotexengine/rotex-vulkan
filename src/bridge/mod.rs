#![allow(dead_code)]
mod bindings;
mod compute_pipeline_cache;
mod init;
mod pass_targets;
mod pipeline_cache;
mod render;
mod resources;
mod surface;
mod types;

use std::collections::{HashMap, HashSet};


use pass_targets::PassTargetCache;
use types::{
    BindGroupLayoutResource, BindGroupResource, BufferResource, ComputePipelineResource,
    DepthMode, MaterialPipeline, MaterialPipelineKey, MaterialResource, MeshResource,
    SurfaceState, TextureResource, VertexLayoutId,
};
use crate::backend::vulkan::{
    CommandBuffer, CommandPool, DeferredDeleteQueue, DescriptorPool, DescriptorPoolManager,
    DescriptorSetLayout, Device, Fence, FrameSlot, VulkanDevice, VulkanInstance,
};
use crate::error::{Error, ErrorKind, vk_error};
use rotex_core::{
    Error as CoreError, ErrorKind as CoreErrorKind, GpuBackend, Severity as CoreSeverity,
};
use rotex_types::resource::{
    BindGroupId, BindGroupLayoutId, BufferId, ComputePipelineId, MaterialId, MeshId, TextureId,
    VertexBufferLayout,
};

pub struct VulkanBridge {
    instance: VulkanInstance,
    device: VulkanDevice,
    command_pool: CommandPool,
    command_buffer: CommandBuffer,
    in_flight_fence: Fence,
    texture_set_layout: DescriptorSetLayout,
    texture_descriptor_pool: DescriptorPool,
    frame_slots: Vec<FrameSlot>,
    frames_in_flight: u32,
    current_frame_index: u32,
    current_image_index: u32,
    recording: bool,
    graphics_queue_index: u32,
    surface_state: Option<SurfaceState>,
    meshes: HashMap<MeshId, MeshResource>,
    materials: HashMap<MaterialId, MaterialResource>,
    textures: HashMap<TextureId, TextureResource>,
    default_texture: Option<TextureResource>,
    bind_group_layouts: HashMap<BindGroupLayoutId, BindGroupLayoutResource>,
    bind_groups: HashMap<BindGroupId, BindGroupResource>,
    general_descriptor_pool: DescriptorPoolManager,
    storage_descriptor_pool: DescriptorPoolManager,
    empty_set_layout: DescriptorSetLayout,
    pass_target_cache: PassTargetCache,
    material_pipelines: HashMap<MaterialPipelineKey, MaterialPipeline>,
    pipelines_by_material: HashMap<MaterialId, HashSet<MaterialPipelineKey>>,
    vertex_layouts: HashMap<VertexLayoutId, VertexBufferLayout>,
    buffers: HashMap<BufferId, BufferResource>,
    compute_pipelines: HashMap<ComputePipelineId, ComputePipelineResource>,
    deferred_delete: DeferredDeleteQueue,
    active_render_pass: Option<vk::RenderPass>,
    active_pass_extent: Option<vk::Extent2D>,
    active_pipeline_layout: Option<vk::PipelineLayout>,
    active_pass: Option<rotex_types::PassDescriptor>,
    active_target_role: Option<pass_targets::TargetPassRole>,
    next_mesh_id: u64,
    next_material_id: u64,
    next_texture_id: u64,
    next_buffer_id: u64,
    next_compute_pipeline_id: u64,
    next_bind_group_layout_id: u64,
    next_bind_group_id: u64,
}

use ash::vk;

fn surface_not_attached_error() -> Error {
    Error::fatal(ErrorKind::Unsupported("Surface is not attached"))
}

fn to_core_error(error: Error) -> CoreError {
    let severity = match error.severity {
        crate::error::Severity::Fatal => CoreSeverity::Fatal,
        crate::error::Severity::Info
        | crate::error::Severity::Warning
        | crate::error::Severity::Recoverable => CoreSeverity::Warning,
    };
    let kind = match error.kind {
        crate::error::ErrorKind::NoCompatibleDevice => CoreErrorKind::NoCompatibleDevice,
        crate::error::ErrorKind::Unsupported(message) => CoreErrorKind::Unsupported(message),
        crate::error::ErrorKind::Vulkan(code) => {
            CoreErrorKind::Backend(format!("Vulkan error: {code:?} ({})", code.as_raw()))
        }
    };
    CoreError { kind, severity }
}

fn map_err_route<T>(r: Result<T, Error>) -> Result<T, CoreError> {
    r.map_err(to_core_error)
}

impl GpuBackend for VulkanBridge {
    fn attach_surface(
        &mut self,
        surface_descriptor: rotex_types::SurfaceDescriptor,
    ) -> Result<(), CoreError> {
        map_err_route(VulkanBridge::attach_surface(self, surface_descriptor))
    }

    fn create_resources(
        &mut self,
        descriptor: rotex_types::ResourceBatchCreate,
    ) -> Result<rotex_types::CreatedResources, CoreError> {
        map_err_route(VulkanBridge::create_resources(self, descriptor))
    }

    fn update_resources(
        &mut self,
        descriptor: rotex_types::ResourceBatchUpdate,
    ) -> Result<(), CoreError> {
        map_err_route(VulkanBridge::update_resources(self, descriptor))
    }

    fn execute(&mut self, commands: &[rotex_types::RhiCommand]) -> Result<(), CoreError> {
        map_err_route(VulkanBridge::execute(self, commands))
    }

    fn resize(&mut self, extent: rotex_types::Extent2D) -> Result<(), CoreError> {
        map_err_route(VulkanBridge::resize(self, extent))
    }

    fn read_texture(
        &mut self,
        id: rotex_types::TextureId,
    ) -> Result<rotex_types::TextureReadback, CoreError> {
        map_err_route(VulkanBridge::read_texture(self, id))
    }

    fn destroy(self: Box<Self>) {
        self.destroy_all();
    }

    fn invalidate_command_cache(&mut self) {}
}

impl VulkanBridge {
    pub fn destroy_all(mut self) {
        unsafe {
            let _ = self.device.raw().logical_device().device_wait_idle();
        }
        self.deferred_delete.destroy_all(self.device.raw());
        self.destroy_all_pipelines();
        self.destroy_all_compute_pipelines();

        let device = self.device.raw();
        for (_, buf) in self.buffers.drain() {
            buf.buffer.destroy(device);
        }
        for (_, mesh) in self.meshes.drain() {
            mesh.vertex_buffer.destroy(device);
            mesh.index_buffer.destroy(device);
        }
        for (_, tex) in self.textures.drain() {
            tex.destroy(device, &self.texture_descriptor_pool);
        }
        if let Some(tex) = self.default_texture.take() {
            tex.destroy(device, &self.texture_descriptor_pool);
        }
        self.pass_target_cache.destroy(device);

        if let Some(state) = self.surface_state.take() {
            for sem in state.render_finished {
                sem.destroy(device);
            }
            state.image_available.destroy(device);
            state.color_targets.destroy(device);
            if let Some(dt) = state.depth_targets {
                dt.destroy(device);
            }
            state.swapchain.destroy(&self.device);
            state.surface.destroy();
        }

        for slot in self.frame_slots.drain(..) {
            slot.fence.destroy(device);
            slot.image_available.destroy(device);
        }
        self.command_pool.destroy(device);
        self.in_flight_fence.destroy(device);

        self.texture_descriptor_pool.destroy(device);
        self.general_descriptor_pool.destroy(device);
        self.storage_descriptor_pool.destroy(device);

        for (_, bgl) in self.bind_group_layouts.drain() {
            bgl.layout.destroy(device);
        }
        self.texture_set_layout.destroy(device);
        self.empty_set_layout.destroy(device);

        self.device.destroy();
        self.instance.destroy();
    }

    pub fn execute(&mut self, commands: &[rotex_types::RhiCommand]) -> Result<(), Error> {
        for command in commands {
            match command {
                rotex_types::RhiCommand::BeginFrame { frame_index } => {
                    let idx = *frame_index as usize % self.frame_slots.len();
                    self.current_frame_index = idx as u32;
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.wait_and_reset(self.device.raw())?;
                    unsafe {
                        self.device.raw().logical_device().reset_command_buffer(
                            slot.command_buffer.handle(),
                            vk::CommandBufferResetFlags::empty(),
                        )
                    }.map_err(vk_error)?;
                    slot.command_buffer.begin(
                        self.device.raw(),
                        vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    )?;
                }
                rotex_types::RhiCommand::AcquireSwapchainImage => {
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    let sem = &slot.image_available;
                    let state = self.surface_state.as_ref()
                        .ok_or_else(surface_not_attached_error)?;
                    let (index, _) = state.swapchain.raw()
                        .acquire_next_image(sem)
                        .map_err(|_| Error::fatal(ErrorKind::NoCompatibleDevice))?;
                    self.current_image_index = index;
                }
                rotex_types::RhiCommand::WriteBuffer { buffer, offset, data } => {
                    let buf_res = self.buffers.get(buffer)
                        .ok_or(Error::fatal(ErrorKind::Unsupported("buffer not found")))?;
                    let mapped = buf_res.buffer.map(self.device.raw())? as *mut u8;
                    let target = unsafe { std::slice::from_raw_parts_mut(
                        mapped.add(*offset as usize), data.len(),
                    )};
                    target.copy_from_slice(data);
                    buf_res.buffer.unmap(self.device.raw());
                }
                rotex_types::RhiCommand::BeginRenderPass { pass, .. } => {
                    if pass.uses_depth_attachment() {
                        let _ = self.ensure_depth_targets();
                    }
                    let state = self.surface_state.as_ref()
                        .ok_or_else(surface_not_attached_error)?;
                    let use_depth = pass.uses_depth_attachment()
                        && state.depth_targets.is_some();
                    let targets = if use_depth {
                        state.depth_targets.as_ref()
                            .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?
                    } else {
                        &state.color_targets
                    };
                    let idx = self.current_image_index as usize;
                    if idx >= targets.framebuffers.len() {
                        return Err(Error::fatal(ErrorKind::NoCompatibleDevice));
                    }
                    let mut clear_values = vec![vk::ClearValue {
                        color: vk::ClearColorValue { float32: pass.clear_color },
                    }];
                    if use_depth {
                        clear_values.push(vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: pass.clear_depth, stencil: 0,
                            },
                        });
                    }
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.command_buffer.begin_render_pass(
                        self.device.raw(),
                        &targets.render_pass,
                        &targets.framebuffers[idx],
                        &clear_values,
                    );
                    self.active_render_pass = Some(targets.render_pass.handle());
                }
                rotex_types::RhiCommand::EndRenderPass => {
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.command_buffer.end_render_pass(self.device.raw());
                    self.active_render_pass = None;
                }
                rotex_types::RhiCommand::BindGraphicsPipeline { material, mesh, depth_enabled } => {
                    let mesh_res = self.meshes.get(mesh)
                        .ok_or(Error::fatal(ErrorKind::Unsupported("mesh not found")))?;
                    let depth_mode = if *depth_enabled { DepthMode::Enabled } else { DepthMode::Disabled };
                    let render_pass = self.active_render_pass
                        .ok_or(Error::fatal(ErrorKind::Unsupported("no active render pass")))?;
                    let (pipeline_handle, layout_handle) = self.pipeline_handle_for(
                        *material, mesh_res.vertex_layout_id, depth_mode, render_pass,
                    )?;
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.command_buffer.bind_graphics_pipeline(
                        self.device.raw(), pipeline_handle,
                    );
                    self.active_pipeline_layout = Some(layout_handle);
                }
                rotex_types::RhiCommand::BindDescriptorSets { first_set, bind_groups, .. } => {
                    let layout = self.active_pipeline_layout
                        .ok_or(Error::fatal(ErrorKind::Unsupported("no pipeline bound")))?;
                    let mut sets = Vec::new();
                    for bg_id in bind_groups {
                        let bg = self.bind_groups.get(bg_id)
                            .ok_or(Error::fatal(ErrorKind::Unsupported("bind group not found")))?;
                        sets.push(bg.descriptor_set.handle());
                    }
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.command_buffer.bind_graphics_descriptor_sets(
                        self.device.raw(),
                        layout,
                        *first_set,
                        &sets,
                    );
                }
                rotex_types::RhiCommand::SetVertexBuffers { mesh, .. } => {
                    let mesh_res = self.meshes.get(mesh)
                        .ok_or(Error::fatal(ErrorKind::Unsupported("mesh not found")))?;
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.command_buffer.bind_vertex_buffer(
                        self.device.raw(),
                        mesh_res.vertex_buffer.handle(),
                    );
                }
                rotex_types::RhiCommand::SetIndexBuffer { mesh } => {
                    let mesh_res = self.meshes.get(mesh)
                        .ok_or(Error::fatal(ErrorKind::Unsupported("mesh not found")))?;
                    let buf = &mesh_res.index_buffer;
                    let idx_type = mesh_res.index_type;
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.command_buffer.bind_index_buffer(
                        self.device.raw(),
                        buf,
                        0,
                        idx_type,
                    );
                }
                rotex_types::RhiCommand::DrawIndexed {
                    index_count, instance_count, first_index, vertex_offset, first_instance,
                } => {
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.command_buffer.draw_indexed(
                        self.device.raw(),
                        *index_count,
                        *instance_count,
                        *first_index,
                        *vertex_offset,
                        *first_instance,
                    );
                }
                rotex_types::RhiCommand::SubmitFrame { present } => {
                    let slot = &self.frame_slots[self.current_frame_index as usize];
                    slot.command_buffer.end(self.device.raw())?;
                    let command_buffers = [slot.command_buffer.handle()];
                    let queue = self.device.raw().get_queue(self.graphics_queue_index, 0);
                    if *present {
                        let state = self.surface_state.as_ref()
                            .ok_or_else(surface_not_attached_error)?;
                        let wait_sem = slot.image_available.handle();
                        let wait_semaphores = [wait_sem];
                        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                        let signal_sem = state.render_finished[self.current_image_index as usize].handle();
                        let signal_semaphores = [signal_sem];
                        let submit = vk::SubmitInfo::default()
                            .wait_semaphores(&wait_semaphores)
                            .wait_dst_stage_mask(&wait_stages)
                            .command_buffers(&command_buffers)
                            .signal_semaphores(&signal_semaphores);
                        unsafe {
                            self.device.raw().logical_device().queue_submit(
                                queue, &[submit], slot.fence.handle(),
                            )
                        }.map_err(vk_error)?;
                        let _ = state.swapchain.raw().present(
                            queue, self.current_image_index,
                            &state.render_finished[self.current_image_index as usize],
                        );
                    } else {
                        let submit = vk::SubmitInfo::default()
                            .command_buffers(&command_buffers);
                        unsafe {
                            self.device.raw().logical_device().queue_submit(
                                queue, &[submit], slot.fence.handle(),
                            )
                        }.map_err(vk_error)?;
                    }
                }
                rotex_types::RhiCommand::TransitionBuffer { .. } | rotex_types::RhiCommand::DispatchCompute(_) | rotex_types::RhiCommand::PushConstants { .. } => {}
            }
        }
        Ok(())
    }

    pub fn read_texture(
        &mut self,
        _id: rotex_types::TextureId,
    ) -> Result<rotex_types::TextureReadback, Error> {
        Err(Error::fatal(ErrorKind::Unsupported(
            "Texture readback not implemented",
        )))
    }

    pub(super) fn current_command_buffer(
        &self,
    ) -> Result<&crate::backend::vulkan::CommandBuffer, Error> {
        let frame_index = self.current_frame_index as usize;
        self.frame_slots
            .get(frame_index)
            .map(|slot| &slot.command_buffer)
            .ok_or_else(|| bindings::allocation_mismatch("frame slot out of range"))
    }

    pub(super) fn wait_frame_slot(&self, frame_index: u32) -> Result<(), Error> {
        let slot = self
            .frame_slots
            .get(frame_index as usize)
            .ok_or_else(|| bindings::allocation_mismatch("frame slot out of range"))?;
        slot.wait_and_reset(self.device.raw())
    }

    pub(super) fn submit_one_shot<F>(&mut self, record: F) -> Result<(), Error>
    where
        F: FnOnce(&Device, &crate::backend::vulkan::CommandBuffer) -> Result<(), Error>,
    {
        let frame_index = 0usize;
        self.wait_frame_slot(frame_index as u32)?;
        let device = self.device.raw();
        let slot = &self.frame_slots[frame_index];
        unsafe {
            device.logical_device().reset_command_buffer(
                slot.command_buffer.handle(),
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .map_err(vk_error)?;
        slot.command_buffer.begin(
            device,
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        )?;
        record(device, &slot.command_buffer)?;
        slot.command_buffer.end(device)?;
        let queue = device.get_queue(self.graphics_queue_index, 0);
        let command_buffers = [slot.command_buffer.handle()];
        let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
        unsafe {
            device.logical_device().queue_submit(
                queue,
                &[submit],
                slot.fence.handle(),
            )
        }
        .map_err(vk_error)?;
        slot.fence.wait(device, u64::MAX)?;
        Ok(())
    }
}
