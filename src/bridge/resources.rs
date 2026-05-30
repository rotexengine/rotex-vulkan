use std::collections::HashSet;

use ash::vk;

use super::{VulkanBridge, types::VertexLayoutId};
use crate::backend::vulkan::{Device, ImageDescriptor, RotexBuffer, RotexImage, RotexSampler, SamplerDescriptor};
use crate::error::{Error, ErrorKind, vk_error};
use rotex_types::resource::{
    CreatedResources, IndexFormat, MaterialId, MeshDescriptor, MeshId, ResourceBatchCreate,
    ResourceBatchUpdate, ResourceCreateDescriptor, ResourceHandle, ResourceUpdateDescriptor,
    TextureDescriptor, TextureFormat, TextureId, VertexBufferLayout, VertexFormat,
};

impl VulkanBridge {
    pub(super) fn ensure_default_texture(&mut self) -> Result<&super::types::TextureResource, Error> {
        if self.default_texture.is_none() {
            let fallback_descriptor = TextureDescriptor {
                width: 1,
                height: 1,
                format: TextureFormat::Rgba8Unorm,
                data: vec![255, 255, 255, 255],
            };
            self.default_texture = Some(self.create_texture_resource(&fallback_descriptor)?);
        }
        Ok(self.default_texture.as_ref().expect("default texture initialized"))
    }

    pub(super) fn create_texture_resource(
        &mut self,
        desc: &TextureDescriptor,
    ) -> Result<super::types::TextureResource, Error> {
        validate_texture_descriptor(desc)?;
        let staging_buffer = RotexBuffer::new(
            self.instance.raw(),
            self.device.raw(),
            desc.data.len() as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        write_bytes(self.device.raw(), &staging_buffer, &desc.data)?;

        let mut image = RotexImage::new(
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
        let sampler = RotexSampler::new(
            self.device.raw(),
            SamplerDescriptor::default().with_filters(vk::Filter::LINEAR, vk::Filter::LINEAR),
        )?;

        if let Err(err) = self.upload_staging_texture(&staging_buffer, &mut image, desc.width, desc.height) {
            sampler.destroy(self.device.raw());
            image.destroy(self.device.raw());
            staging_buffer.destroy(self.device.raw());
            return Err(err);
        }
        staging_buffer.destroy(self.device.raw());

        let layouts = [self.texture_set_layout.handle()];
        let mut descriptor_sets = match self
            .texture_descriptor_pool
            .allocate_sets(self.device.raw(), &layouts)
        {
            Ok(sets) => sets,
            Err(err) => {
                sampler.destroy(self.device.raw());
                image.destroy(self.device.raw());
                return Err(err);
            }
        };
        let descriptor_set = descriptor_sets.pop().expect("one descriptor set");
        descriptor_set.write_image_sampler(
            self.device.raw(),
            0,
            image.view(),
            sampler.handle(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        Ok(super::types::TextureResource {
            descriptor: desc.clone(),
            image,
            sampler,
            descriptor_set,
        })
    }

    fn upload_staging_texture(
        &mut self,
        staging_buffer: &RotexBuffer,
        image: &mut RotexImage,
        width: u32,
        height: u32,
    ) -> Result<(), Error> {
        self.in_flight_fence.wait(self.device.raw(), u64::MAX)?;
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
        image.transition_layout(
            self.device.raw(),
            &self.command_buffer,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        self.command_buffer.copy_buffer_to_image(
            self.device.raw(),
            staging_buffer.handle(),
            image.handle(),
            width,
            height,
        );
        image.transition_layout(
            self.device.raw(),
            &self.command_buffer,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
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
        }
        .map_err(vk_error)?;
        self.in_flight_fence.wait(self.device.raw(), u64::MAX)?;
        Ok(())
    }

    pub(super) fn create_mesh_resource(
        &self,
        desc: &MeshDescriptor,
        vertex_layout_id: VertexLayoutId,
    ) -> Result<super::types::MeshResource, Error> {
        if desc.index_count == 0 {
            return Err(Error::fatal(ErrorKind::NoCompatibleDevice));
        }
        let vertex_size = desc.vertex_data.len() as vk::DeviceSize;
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
            vertex_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let index_buffer = RotexBuffer::new(
            self.instance.raw(),
            self.device.raw(),
            index_size,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        write_bytes(self.device.raw(), &vertex_buffer, &desc.vertex_data)?;
        write_bytes(self.device.raw(), &index_buffer, &desc.index_data)?;
        Ok(super::types::MeshResource {
            vertex_buffer,
            index_buffer,
            index_type,
            index_count: desc.index_count,
            vertex_layout_id,
        })
    }

    pub fn create_resources(
        &mut self,
        descriptor: ResourceBatchCreate,
    ) -> Result<CreatedResources, Error> {
        let mut handles = Vec::with_capacity(descriptor.resources.len());
        for item in descriptor.resources {
            match item {
                ResourceCreateDescriptor::Mesh(mesh) => {
                    validate_vertex_layout(&mesh.vertex_layout, mesh.vertex_data.len())?;
                    let vertex_layout_id = self.intern_vertex_layout(&mesh.vertex_layout)?;
                    let id = MeshId(self.next_mesh_id);
                    self.next_mesh_id += 1;
                    let resource = self.create_mesh_resource(&mesh, vertex_layout_id)?;
                    self.meshes.insert(id, resource);
                    handles.push(ResourceHandle::Mesh(id));
                }
                ResourceCreateDescriptor::Texture(texture) => {
                    let id = TextureId(self.next_texture_id);
                    self.next_texture_id += 1;
                    let resource = self.create_texture_resource(&texture)?;
                    self.textures.insert(id, resource);
                    handles.push(ResourceHandle::Texture(id));
                }
                ResourceCreateDescriptor::Material(material) => {
                    let id = MaterialId(self.next_material_id);
                    self.next_material_id += 1;
                    self.materials
                        .insert(id, super::types::MaterialResource { descriptor: material });
                    handles.push(ResourceHandle::Material(id));
                }
            }
        }
        Ok(CreatedResources { handles })
    }

    pub fn update_resources(
        &mut self,
        descriptor: ResourceBatchUpdate,
    ) -> Result<(), Error> {
        if descriptor.updates.iter().any(|update| {
            matches!(
                update,
                ResourceUpdateDescriptor::Mesh { .. }
                    | ResourceUpdateDescriptor::Texture { .. }
                    | ResourceUpdateDescriptor::Material { .. }
            )
        }) {
            self.in_flight_fence.wait(self.device.raw(), u64::MAX)?;
        }
        for item in descriptor.updates {
            match item {
                ResourceUpdateDescriptor::Mesh {
                    id,
                    vertex_data,
                    vertex_layout,
                    index_data,
                    index_format,
                    index_count,
                } => {
                    validate_vertex_layout(&vertex_layout, vertex_data.len())?;
                    let vertex_layout_id = self.intern_vertex_layout(&vertex_layout)?;
                    let resource = self.create_mesh_resource(
                        &MeshDescriptor {
                            vertex_data,
                            vertex_layout,
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
                    old.vertex_buffer.destroy(self.device.raw());
                    old.index_buffer.destroy(self.device.raw());
                    self.meshes.insert(id, resource);
                }
                ResourceUpdateDescriptor::Texture { id, data } => {
                    let old_texture_descriptor = self
                        .textures
                        .get(&id)
                        .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?;
                    let mut updated_texture_descriptor = old_texture_descriptor.descriptor.clone();
                    updated_texture_descriptor.data = data;
                    let updated_texture = self.create_texture_resource(&updated_texture_descriptor)?;
                    let previous = self
                        .textures
                        .insert(id, updated_texture)
                        .expect("texture must exist after get check");
                    previous.destroy(self.device.raw(), &self.texture_descriptor_pool);
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
            }
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

fn map_texture_format(format: TextureFormat) -> vk::Format {
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
