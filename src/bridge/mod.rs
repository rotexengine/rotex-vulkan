mod init;
mod pipeline_cache;
mod render;
mod resources;
mod surface;
mod types;

use std::collections::{HashMap, HashSet};

use crate::backend::vulkan::{
    CommandBuffer, CommandPool, DescriptorPool, DescriptorSetLayout, Fence, VulkanDevice,
    VulkanInstance,
};
use crate::error::{Error, ErrorKind};
use rotex_types::resource::{MaterialId, MeshId, TextureId, VertexBufferLayout};

use self::types::{
    MaterialPipeline, MaterialPipelineKey, MaterialResource, MeshResource, SurfaceState,
    TextureResource, VertexLayoutId,
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
    texture_descriptor_pool: DescriptorPool,
    texture_set_layout: DescriptorSetLayout,
    material_pipelines: HashMap<MaterialPipelineKey, MaterialPipeline>,
    pipelines_by_material: HashMap<MaterialId, HashSet<MaterialPipelineKey>>,
    vertex_layouts: HashMap<VertexLayoutId, VertexBufferLayout>,
    next_mesh_id: u64,
    next_material_id: u64,
    next_texture_id: u64,
}

fn surface_not_attached_error() -> Error {
    Error::fatal(ErrorKind::Unsupported("Surface is not attached"))
}
