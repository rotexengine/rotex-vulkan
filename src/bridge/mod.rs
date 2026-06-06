mod compute_pipeline_cache;
mod init;
mod pass_targets;
mod pipeline_cache;
mod render;
mod resources;
mod surface;
mod types;

use std::collections::{HashMap, HashSet};

use crate::backend::vulkan::{
    CommandBuffer, CommandPool, DescriptorPool, DescriptorSet, DescriptorSetLayout, Fence,
    GraphicsPipelineLayout, RotexBuffer, VulkanDevice, VulkanInstance,
};
use crate::error::{Error, ErrorKind};
use rotex_core::{
    Error as CoreError, ErrorKind as CoreErrorKind, GpuBackend, Severity as CoreSeverity,
};
use rotex_types::resource::{
    BufferId, ComputePipelineId, MaterialId, MeshId, TextureId, VertexBufferLayout,
};

use self::pass_targets::PassTargetCache;
use self::types::{
    BufferResource, ComputePipelineResource, MaterialPipeline, MaterialPipelineKey,
    MaterialResource, MeshResource, SurfaceState, TextureResource, VertexLayoutId,
};

pub struct VulkanBridge {
    instance: VulkanInstance,
    device: VulkanDevice,
    command_pool: CommandPool,
    command_buffer: CommandBuffer,
    in_flight_fence: Fence,
    graphics_queue_index: u32,
    surface_state: Option<SurfaceState>,
    meshes: HashMap<MeshId, MeshResource>,
    materials: HashMap<MaterialId, MaterialResource>,
    textures: HashMap<TextureId, TextureResource>,
    default_texture: Option<TextureResource>,
    material_descriptor_pool: DescriptorPool,
    uniform_descriptor_pool: DescriptorPool,
    global_set_layout: DescriptorSetLayout,
    material_set_layout: DescriptorSetLayout,
    object_set_layout: DescriptorSetLayout,
    global_uniform_buffer: RotexBuffer,
    object_uniform_buffer: RotexBuffer,
    global_descriptor_set: DescriptorSet,
    object_descriptor_set: DescriptorSet,
    object_aligned_stride: u32,
    object_buffer_capacity: u32,
    shared_pipeline_layout: GraphicsPipelineLayout,
    pass_target_cache: PassTargetCache,
    material_pipelines: HashMap<MaterialPipelineKey, MaterialPipeline>,
    pipelines_by_material: HashMap<MaterialId, HashSet<MaterialPipelineKey>>,
    vertex_layouts: HashMap<VertexLayoutId, VertexBufferLayout>,
    buffers: HashMap<BufferId, BufferResource>,
    compute_pipelines: HashMap<ComputePipelineId, ComputePipelineResource>,
    storage_descriptor_pool: DescriptorPool,
    next_mesh_id: u64,
    next_material_id: u64,
    next_texture_id: u64,
    next_buffer_id: u64,
    next_compute_pipeline_id: u64,
}

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

    fn execute(
        &mut self,
        scene: &rotex_types::SceneDescriptor,
        commands: &[rotex_types::RenderCommand],
    ) -> Result<(), CoreError> {
        VulkanBridge::execute(self, scene, commands).map_err(to_core_error)
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
