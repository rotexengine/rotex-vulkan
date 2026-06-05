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
