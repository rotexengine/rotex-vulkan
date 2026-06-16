use std::collections::HashSet;

use ash::vk;

use super::bindings::{build_set_layout, map_binding_type, map_memory_location};
use super::{VulkanBridge, types::VertexLayoutId};
use crate::backend::vulkan::{Device, ImageDescriptor, RotexBuffer, RotexImage, RotexSampler};
use crate::error::{Error, ErrorKind, vk_error};
use rotex_types::resource::{
    BindGroupEntryDescriptor, BindGroupId, BindGroupLayoutId, BufferId, BufferUsage, BufferUsages,
    ComputePipelineId, CreatedResources, IndexFormat, MaterialId, MeshDescriptor, MeshId,
    ResourceBatchCreate, ResourceBatchUpdate, ResourceCreateDescriptor, ResourceHandle,
    ResourceUpdateDescriptor, TextureDescriptor, TextureFormat, TextureId, VertexBufferLayout,
    VertexFormat, VertexStreamData,
};

struct TextureStagingItem {
    buffer: RotexBuffer,
    id: TextureId,
    width: u32,
    height: u32,
}

struct BufferStagingItem {
    staging: RotexBuffer,
    id: MeshId,
    is_vertex: bool,
    size: vk::DeviceSize,
}

impl VulkanBridge {
    pub(super) fn create_texture_allocation(
        &self,
        desc: &TextureDescriptor,
    ) -> Result<(super::types::TextureResource, RotexBuffer), Error> {
        validate_texture_descriptor(desc)?;
        let staging_buffer = RotexBuffer::new(
            self.instance.raw(),
            self.device.raw(),
            desc.data.len() as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        write_bytes(self.device.raw(), &staging_buffer, &desc.data)?;

        let image = RotexImage::new(
            self.instance.raw(),
            self.device.raw(),
            ImageDescriptor::default(
                map_texture_format(desc.format),
                vk::Extent3D {
                    width: desc.width,
                    height: desc.height,
                    depth: 1,
                },
                vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ),
        )?;

        Ok((super::types::TextureResource {
            descriptor: desc.clone(),
            image,
        }, staging_buffer))
    }

    fn upload_staging_batch(
        &mut self,
        textures: &[TextureStagingItem],
        buffers: &[BufferStagingItem],
    ) -> Result<(), Error> {
        if textures.is_empty() && buffers.is_empty() {
            return Ok(());
        }
        self.in_flight_fence.wait(self.device.raw(), u64::MAX)?;
        self.in_flight_fence.reset(self.device.raw())?;
        unsafe {
            self.device.raw().logical_device().reset_command_buffer(
                self.command_buffer.handle(),
                vk::CommandBufferResetFlags::empty(),
            )
        }.map_err(vk_error)?;
        self.command_buffer.begin(
            self.device.raw(),
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        )?;

        for item in textures {
            let tex_res = self.textures.get_mut(&item.id)
                .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?;
            tex_res.image.transition_layout(
                self.device.raw(),
                &self.command_buffer,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            self.command_buffer.copy_buffer_to_image(
                self.device.raw(),
                item.buffer.handle(),
                tex_res.image.handle(),
                item.width,
                item.height,
            );
            tex_res.image.transition_layout(
                self.device.raw(),
                &self.command_buffer,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        }

        for item in buffers {
            let mesh = self.meshes.get(&item.id)
                .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?;
            let dest = if item.is_vertex {
                mesh.vertex_buffer.handle()
            } else {
                mesh.index_buffer.handle()
            };
            self.command_buffer.copy_buffer(
                self.device.raw(),
                item.staging.handle(),
                dest,
                item.size,
            );
        }

        self.command_buffer.end(self.device.raw())?;
        let command_buffers = [self.command_buffer.handle()];
        let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
        let queue = self.device.raw().get_queue(self.graphics_queue_index, 0);
        unsafe {
            self.device.raw().logical_device().queue_submit(
                queue,
                &[submit],
                self.in_flight_fence.handle(),
            )
        }.map_err(vk_error)?;
        self.in_flight_fence.wait(self.device.raw(), u64::MAX)?;
        Ok(())
    }

    pub(super) fn create_mesh_allocation(
        &self,
        desc: &MeshDescriptor,
        vertex_layout_id: VertexLayoutId,
    ) -> Result<(super::types::MeshResource, Option<RotexBuffer>, Option<RotexBuffer>), Error> {
        if desc.index_count == 0 {
            return Err(Error::fatal(ErrorKind::NoCompatibleDevice));
        }
        let (vertex_size, vertex_data_slice) = match &desc.vertex_streams[0].data {
            VertexStreamData::Static(data) => (data.len() as vk::DeviceSize, data.as_slice()),
            VertexStreamData::External(_) => (0, &[][..]),
        };
        let index_size = desc.index_data.len() as vk::DeviceSize;
        let index_type = map_index_type(desc.index_format);
        let min_index_bytes = desc.index_count as usize * index_format_size(desc.index_format);
        if desc.index_data.len() < min_index_bytes {
            return Err(Error::fatal(ErrorKind::Unsupported(
                "Index data is shorter than index_count requires",
            )));
        }

        let vertex_buffer = RotexBuffer::new(
            self.instance.raw(),
            self.device.raw(),
            vertex_size.max(1),
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let index_buffer = RotexBuffer::new(
            self.instance.raw(),
            self.device.raw(),
            index_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let vertex_staging = if !vertex_data_slice.is_empty() {
            let staging = RotexBuffer::new(
                self.instance.raw(),
                self.device.raw(),
                vertex_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            write_bytes(self.device.raw(), &staging, vertex_data_slice)?;
            Some(staging)
        } else {
            None
        };

        let index_staging = {
            let staging = RotexBuffer::new(
                self.instance.raw(),
                self.device.raw(),
                index_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            write_bytes(self.device.raw(), &staging, &desc.index_data)?;
            Some(staging)
        };

        Ok((super::types::MeshResource {
            vertex_buffer,
            index_buffer,
            index_type,
            index_count: desc.index_count,
            vertex_layout_id,
        }, vertex_staging, index_staging))
    }

    pub fn create_resources(
        &mut self,
        descriptor: ResourceBatchCreate,
    ) -> Result<CreatedResources, Error> {
        let mut handles = Vec::with_capacity(descriptor.resources.len());
        let mut texture_staging = Vec::new();
        let mut mesh_staging = Vec::new();

        for item in descriptor.resources {
            match item {
                ResourceCreateDescriptor::Mesh { id, mesh } => {
                    let vertex_data_len = match &mesh.vertex_streams[0].data {
                        VertexStreamData::Static(data) => data.len(),
                        VertexStreamData::External(_) => 0,
                    };
                    let vertex_size = vertex_data_len as vk::DeviceSize;
                    let index_size = mesh.index_data.len() as vk::DeviceSize;
                    validate_vertex_layout(&mesh.vertex_streams[0].layout, vertex_data_len)?;
                    let vertex_layout_id = self.intern_vertex_layout(&mesh.vertex_streams[0].layout)?;
                    let id = MeshId(id);
                    let (resource, vertex_staging, index_staging) =
                        self.create_mesh_allocation(&mesh, vertex_layout_id)?;
                    if let Some(vs) = vertex_staging {
                        mesh_staging.push(BufferStagingItem {
                            staging: vs,
                            id,
                            is_vertex: true,
                            size: vertex_size,
                        });
                    }
                    if let Some(is) = index_staging {
                        mesh_staging.push(BufferStagingItem {
                            staging: is,
                            id,
                            is_vertex: false,
                            size: index_size,
                        });
                    }
                    self.meshes.insert(id, resource);
                    handles.push(ResourceHandle::Mesh(id));
                }
                ResourceCreateDescriptor::Texture { id, texture } => {
                    let id = TextureId(id);
                    let (resource, staging) = self.create_texture_allocation(&texture)?;
                    texture_staging.push(TextureStagingItem {
                        buffer: staging,
                        id,
                        width: texture.width,
                        height: texture.height,
                    });
                    self.textures.insert(id, resource);
                    handles.push(ResourceHandle::Texture(id));
                }
                ResourceCreateDescriptor::Material { id, material } => {
                    let id = MaterialId(id);
                    self.materials
                        .insert(id, super::types::MaterialResource { descriptor: material });
                    handles.push(ResourceHandle::Material(id));
                }
                ResourceCreateDescriptor::Buffer { id, buffer: buf } => {
                    let id = BufferId(id);
                    let usage = map_buffer_usage(&buf);
                    let props = map_memory_location(buf.memory_location);
                    let buffer = RotexBuffer::new(
                        self.instance.raw(),
                        self.device.raw(),
                        buf.size.max(1),
                        usage,
                        props,
                    )?;
                    if let Some(data) = &buf.initial_data {
                        write_bytes(self.device.raw(), &buffer, data)?;
                    }
                    self.buffers.insert(id, super::types::BufferResource {
                        buffer,
                        size: buf.size,
                    });
                    handles.push(ResourceHandle::Buffer(id));
                }
                ResourceCreateDescriptor::BindGroupLayout { id, layout } => {
                    let id = BindGroupLayoutId(id);
                    let vk_layout = build_set_layout(self.device.raw(), &layout)?;
                    self.bind_group_layouts.insert(id, super::types::BindGroupLayoutResource {
                        layout: vk_layout,
                        desc: layout,
                    });
                    handles.push(ResourceHandle::BindGroupLayout(id));
                }
                ResourceCreateDescriptor::BindGroup { id, group: bg } => {
                    let id = BindGroupId(id);
                    let layout = self.bind_group_layouts.get(&bg.layout)
                        .ok_or(Error::fatal(ErrorKind::Unsupported("bind group layout not found")))?;
                    let sets = self.general_descriptor_pool.allocate_sets(
                        self.device.raw(),
                        &[layout.layout.handle()],
                    )?;
                    let set = sets.into_iter().next()
                        .ok_or(Error::fatal(ErrorKind::Unsupported("failed to allocate descriptor set")))?;
                    for entry in &bg.entries {
                        match entry {
                            BindGroupEntryDescriptor::Buffer { binding, buffer, offset, size } => {
                                let buf_res = self.buffers.get(buffer)
                                    .ok_or(Error::fatal(ErrorKind::Unsupported("buffer not found")))?;
                                let range = if *size == 0 { buf_res.size } else { *size };
                                let desc_type = layout.desc.entries.iter()
                                    .find(|e| e.binding == *binding)
                                    .map(|e| map_binding_type(e.ty))
                                    .unwrap_or(vk::DescriptorType::STORAGE_BUFFER);
                                set.write_buffer(
                                    self.device.raw(),
                                    *binding,
                                    &buf_res.buffer,
                                    *offset,
                                    range,
                                    desc_type,
                                );
                            }
                            BindGroupEntryDescriptor::Texture { binding, texture } => {
                                let tex_res = self
                                    .textures
                                    .get(texture)
                                    .ok_or(Error::fatal(ErrorKind::Unsupported(
                                        "texture not found for bind group",
                                    )))?;
                                set.write_image_sampler(
                                    self.device.raw(),
                                    *binding,
                                    tex_res.image.view(),
                                    self.fallback_sampler.handle(),
                                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                );
                            }
                        }
                    }
                    self.bind_groups.insert(id, super::types::BindGroupResource {
                        descriptor_set: set,
                    });
                    handles.push(ResourceHandle::BindGroup(id));
                }
                ResourceCreateDescriptor::ComputePipeline { id, pipeline } => {
                    let id = ComputePipelineId(id);
                    let result = self.create_compute_pipeline_resource(&pipeline)?;
                    self.compute_pipelines.insert(id, result);
                    handles.push(ResourceHandle::ComputePipeline(id));
                }
            }
        }

        if !texture_staging.is_empty() || !mesh_staging.is_empty() {
            self.upload_staging_batch(&texture_staging, &mesh_staging)?;
            for ts in &texture_staging {
                ts.buffer.destroy(self.device.raw());
            }
            for ms in &mesh_staging {
                ms.staging.destroy(self.device.raw());
            }
        }

        Ok(CreatedResources { handles })
    }

    pub fn update_resources(
        &mut self,
        descriptor: ResourceBatchUpdate,
    ) -> Result<(), Error> {
        let mut texture_staging = Vec::new();
        let mut mesh_staging = Vec::new();
        let mut deferred_textures: Vec<(TextureId, crate::backend::vulkan::RotexImage)> = Vec::new();
        let mut deferred_meshes: Vec<(MeshId, crate::backend::vulkan::RotexBuffer, crate::backend::vulkan::RotexBuffer)> = Vec::new();

        for item in descriptor.updates {
            match item {
                ResourceUpdateDescriptor::Mesh {
                    id,
                    vertex_streams,
                    index_data,
                    index_format,
                    index_count,
                } => {
                    let vertex_data_len = match &vertex_streams[0].data {
                        VertexStreamData::Static(data) => data.len(),
                        VertexStreamData::External(_) => 0,
                    };
                    let vertex_size = vertex_data_len as vk::DeviceSize;
                    let index_size = index_data.len() as vk::DeviceSize;
                    validate_vertex_layout(&vertex_streams[0].layout, vertex_data_len)?;
                    let vertex_layout_id = self.intern_vertex_layout(&vertex_streams[0].layout)?;
                    let (resource, vertex_staging, index_staging) = self.create_mesh_allocation(
                        &MeshDescriptor {
                            vertex_streams,
                            index_data,
                            index_format,
                            index_count,
                        },
                        vertex_layout_id,
                    )?;
                    let old = self
                        .meshes
                        .remove(&id)
                        .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?;
                    if let Some(vs) = vertex_staging {
                        mesh_staging.push(BufferStagingItem {
                            staging: vs,
                            id,
                            is_vertex: true,
                            size: vertex_size,
                        });
                    }
                    if let Some(is) = index_staging {
                        mesh_staging.push(BufferStagingItem {
                            staging: is,
                            id,
                            is_vertex: false,
                            size: index_size,
                        });
                    }
                    self.meshes.insert(id, resource);
                    deferred_meshes.push((id, old.vertex_buffer, old.index_buffer));
                }
                ResourceUpdateDescriptor::Texture { id, data } => {
                    let old_texture_descriptor = self
                        .textures
                        .get(&id)
                        .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?;
                    let mut updated_texture_descriptor = old_texture_descriptor.descriptor.clone();
                    updated_texture_descriptor.data = data;
                    let (updated_texture, staging) =
                        self.create_texture_allocation(&updated_texture_descriptor)?;
                    let previous = self
                        .textures
                        .insert(id, updated_texture)
                        .expect("texture must exist after get check");
                    texture_staging.push(TextureStagingItem {
                        buffer: staging,
                        id,
                        width: updated_texture_descriptor.width,
                        height: updated_texture_descriptor.height,
                    });
                    deferred_textures.push((id, previous.image));
                }
                ResourceUpdateDescriptor::Material {
                    id,
                    enable_depth,
                    texture,
                } => {
                    let material = self
                        .materials
                        .get_mut(&id)
                        .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?;
                    if let Some(enable_depth) = enable_depth {
                        material.descriptor.enable_depth = enable_depth;
                    }
                    if let Some(texture) = texture {
                        material.descriptor.texture = texture;
                    }
                    self.invalidate_material_pipelines(id);
                }
                _ => {}
            }
        }

        if !texture_staging.is_empty() || !mesh_staging.is_empty() {
            self.upload_staging_batch(&texture_staging, &mesh_staging)?;
            for ts in &texture_staging {
                ts.buffer.destroy(self.device.raw());
            }
            for ms in &mesh_staging {
                ms.staging.destroy(self.device.raw());
            }
        }

        let frame = self.current_frame_index as usize;
        for (_, image) in deferred_textures {
            self.deferred_delete
                .defer_after_frame(frame, crate::backend::vulkan::DeferredResource::Image(image));
        }
        for (_, vertex, index) in deferred_meshes {
            self.deferred_delete.defer_after_frame(
                frame,
                crate::backend::vulkan::DeferredResource::Mesh {
                    vertex,
                    index,
                },
            );
        }

        Ok(())
    }

    fn intern_vertex_layout(&mut self, layout: &VertexBufferLayout) -> Result<VertexLayoutId, Error> {
        let layout_id = compute_vertex_layout_id(layout);
        if let Some(existing) = self.vertex_layouts.get(&layout_id) {
            if existing != layout {
                return Err(Error::fatal(ErrorKind::Unsupported(
                    "Vertex layout ID collision detected",
                )));
            }
        } else {
            self.vertex_layouts.insert(layout_id, layout.clone());
        }
        Ok(layout_id)
    }
}

fn write_bytes(device: &Device, buffer: &RotexBuffer, data: &[u8]) -> Result<(), Error> {
    let ptr = buffer.map(device)? as *mut u8;
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
    buffer.unmap(device);
    Ok(())
}

fn map_index_type(index_format: IndexFormat) -> vk::IndexType {
    match index_format {
        IndexFormat::Uint16 => vk::IndexType::UINT16,
        IndexFormat::Uint32 => vk::IndexType::UINT32,
    }
}

fn index_format_size(index_format: IndexFormat) -> usize {
    match index_format {
        IndexFormat::Uint16 => 2,
        IndexFormat::Uint32 => 4,
    }
}

fn validate_vertex_layout(layout: &VertexBufferLayout, vertex_data_len: usize) -> Result<(), Error> {
    if layout.array_stride == 0 {
        return Err(Error::fatal(ErrorKind::Unsupported(
            "Vertex layout stride must be greater than zero",
        )));
    }
    if layout.array_stride > u32::MAX as u64 {
        return Err(Error::fatal(ErrorKind::Unsupported(
            "Vertex layout stride exceeds Vulkan limits",
        )));
    }
    if !vertex_data_len.is_multiple_of(layout.array_stride as usize) {
        return Err(Error::fatal(ErrorKind::Unsupported(
            "Vertex data size is not aligned with vertex stride",
        )));
    }

    let mut seen_locations = HashSet::new();
    for attribute in &layout.attributes {
        if !seen_locations.insert(attribute.location) {
            return Err(Error::fatal(ErrorKind::Unsupported(
                "Vertex layout has duplicate attribute location",
            )));
        }
        let format_size = vertex_format_size(attribute.format);
        if attribute.offset + format_size > layout.array_stride {
            return Err(Error::fatal(ErrorKind::Unsupported(
                "Vertex attribute exceeds stride bounds",
            )));
        }
        if attribute.offset > u32::MAX as u64 {
            return Err(Error::fatal(ErrorKind::Unsupported(
                "Vertex attribute offset exceeds Vulkan limits",
            )));
        }
    }
    Ok(())
}

fn vertex_format_size(format: VertexFormat) -> u64 {
    match format {
        VertexFormat::Float32 => 4,
        VertexFormat::Float32x2 => 8,
        VertexFormat::Float32x3 => 12,
        VertexFormat::Float32x4 => 16,
        VertexFormat::Uint32 => 4,
    }
}

fn vertex_format_tag(format: VertexFormat) -> u8 {
    match format {
        VertexFormat::Float32 => 1,
        VertexFormat::Float32x2 => 2,
        VertexFormat::Float32x3 => 3,
        VertexFormat::Float32x4 => 4,
        VertexFormat::Uint32 => 5,
    }
}

pub fn map_texture_format(format: TextureFormat) -> vk::Format {
    match format {
        TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
    }
}

fn expected_texture_bytes(desc: &TextureDescriptor) -> Option<usize> {
    (desc.width as usize)
        .checked_mul(desc.height as usize)?
        .checked_mul(4)
}

fn validate_texture_descriptor(desc: &TextureDescriptor) -> Result<(), Error> {
    if desc.width == 0 || desc.height == 0 {
        return Err(Error::fatal(ErrorKind::Unsupported(
            "Texture dimensions must be greater than zero",
        )));
    }
    let Some(expected_bytes) = expected_texture_bytes(desc) else {
        return Err(Error::fatal(ErrorKind::Unsupported(
            "Texture dimensions overflow expected byte size",
        )));
    };
    if desc.data.len() != expected_bytes {
        return Err(Error::fatal(ErrorKind::Unsupported(
            "Texture data size does not match width*height*4",
        )));
    }
    Ok(())
}

fn compute_vertex_layout_id(layout: &VertexBufferLayout) -> VertexLayoutId {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
        for byte in bytes {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    let mut hash = FNV_OFFSET;
    hash = hash_bytes(hash, &layout.array_stride.to_le_bytes());
    let step_tag = match layout.step_mode {
        rotex_types::VertexStepMode::Vertex => 0u8,
        rotex_types::VertexStepMode::Instance => 1u8,
    };
    hash = hash_bytes(hash, &[step_tag]);

    let mut attributes = layout.attributes.clone();
    attributes.sort_by_key(|attr| (attr.location, attr.offset, vertex_format_tag(attr.format)));
    hash = hash_bytes(hash, &(attributes.len() as u32).to_le_bytes());
    for attribute in attributes {
        hash = hash_bytes(hash, &attribute.location.to_le_bytes());
        hash = hash_bytes(hash, &[vertex_format_tag(attribute.format)]);
        hash = hash_bytes(hash, &attribute.offset.to_le_bytes());
    }

    VertexLayoutId(hash)
}

fn map_buffer_usage(desc: &rotex_types::resource::BufferDescriptor) -> vk::BufferUsageFlags {
    let mut flags = vk::BufferUsageFlags::TRANSFER_DST;
    let usages = desc.effective_usages();
    if usages.contains(BufferUsages::VERTEX) {
        flags |= vk::BufferUsageFlags::VERTEX_BUFFER;
    }
    if usages.contains(BufferUsages::INDEX) {
        flags |= vk::BufferUsageFlags::INDEX_BUFFER;
    }
    if usages.contains(BufferUsages::UNIFORM) {
        flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
    }
    if usages.contains(BufferUsages::STORAGE) {
        flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
    }
    if flags == vk::BufferUsageFlags::TRANSFER_DST {
        if desc.usage == BufferUsage::Uniform {
            flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        } else if desc.usage == BufferUsage::Storage {
            flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
        } else if desc.usage == BufferUsage::Vertex {
            flags |= vk::BufferUsageFlags::VERTEX_BUFFER;
        } else if desc.usage == BufferUsage::Index {
            flags |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
    }
    flags
}
