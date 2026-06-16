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
    MaterialPipeline, MaterialPipelineKey, MaterialResource, MeshResource, SurfaceState,
    TextureResource, VertexLayoutId,
};
use crate::backend::vulkan::{
    CommandPool, DeferredDeleteQueue, DescriptorPoolManager, DescriptorSetLayout, Device,
    FrameSlot, VulkanDevice, VulkanInstance,
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
    vertex_layouts: HashMap<VertexLayoutId, Vec<VertexBufferLayout>>,
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

impl GpuBackend for VulkanBridge {
    fn attach_surface(
        &mut self,
        surface_descriptor: rotex_types::SurfaceDescriptor,
    ) -> Result<(), CoreError> {
        VulkanBridge::attach_surface(self, surface_descriptor).map_err(to_core_error)
    }

    fn create_resources(
        &mut self,
        descriptor: rotex_types::ResourceBatchCreate,
    ) -> Result<rotex_types::CreatedResources, CoreError> {
        VulkanBridge::create_resources(self, descriptor).map_err(to_core_error)
    }

    fn update_resources(
        &mut self,
        descriptor: rotex_types::ResourceBatchUpdate,
    ) -> Result<(), CoreError> {
        VulkanBridge::update_resources(self, descriptor).map_err(to_core_error)
    }

    fn execute(&mut self, commands: &[rotex_types::RhiCommand]) -> Result<(), CoreError> {
        VulkanBridge::execute(self, commands).map_err(to_core_error)
    }

    fn resize(&mut self, extent: rotex_types::Extent2D) -> Result<(), CoreError> {
        VulkanBridge::resize(self, extent).map_err(to_core_error)
    }

    fn read_texture(
        &mut self,
        id: rotex_types::TextureId,
    ) -> Result<rotex_types::TextureReadback, CoreError> {
        VulkanBridge::read_texture(self, id).map_err(to_core_error)
    }

    fn destroy(self: Box<Self>) {
        VulkanBridge::destroy(*self);
    }
}

impl VulkanBridge {
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
