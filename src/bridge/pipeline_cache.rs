use ash::vk;

use super::{VulkanBridge, surface_not_attached_error};
use crate::backend::vulkan::{
    ColorBlendAttachmentState, ColorBlendState, DepthStencilState, GraphicsPipelineBuilder,
    GraphicsPipelineLayout, RasterizationState, ShaderModule, ShaderStageDescriptor,
    VertexInputDescriptor,
};
use crate::error::{Error, ErrorKind};
use rotex_types::resource::{MaterialDescriptor, MaterialId, VertexBufferLayout, VertexFormat};
use super::types::{DepthMode, MaterialPipelineKey, VertexLayoutId};

impl VulkanBridge {
    pub(super) fn create_pipeline_for_material(
        &self,
        material: &MaterialDescriptor,
        vertex_layout: &VertexBufferLayout,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
        depth_mode: DepthMode,
    ) -> Result<super::types::MaterialPipeline, Error> {
        let vert_words = spv_bytes_to_words(&material.vertex_shader_spv);
        let frag_words = spv_bytes_to_words(&material.fragment_shader_spv);
        let vert = ShaderModule::new(self.device.raw(), &vert_words)?;
        let frag = ShaderModule::new(self.device.raw(), &frag_words)?;
        let set_layouts = [self.texture_set_layout.handle()];
        let layout = GraphicsPipelineLayout::new(self.device.raw(), &set_layouts, &[])?;
        let pipeline = GraphicsPipelineBuilder::new()
            .with_shader_stage(ShaderStageDescriptor::new(vk::ShaderStageFlags::VERTEX, &vert))
            .with_shader_stage(ShaderStageDescriptor::new(vk::ShaderStageFlags::FRAGMENT, &frag))
            .with_color_blend_state(
                ColorBlendState::default().with_attachment(ColorBlendAttachmentState::default()),
            )
            .with_rasterization_state(
                RasterizationState::default().with_cull_mode(vk::CullModeFlags::NONE),
            )
            .with_depth_stencil_state(if depth_mode.is_enabled() {
                DepthStencilState::default()
                    .with_depth_test_enable(true)
                    .with_depth_write_enable(true)
                    .with_depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            } else {
                DepthStencilState::default()
            })
            .with_vertex_input_state(vertex_input_descriptor(vertex_layout)?)
            .with_render_pass(render_pass)
            .with_layout(layout.handle())
            .with_extent(extent.width, extent.height)
            .build(self.device.raw())?;
        vert.destroy(self.device.raw());
        frag.destroy(self.device.raw());
        Ok(super::types::MaterialPipeline { layout, pipeline })
    }

    pub(super) fn pipeline_handle_for(
        &mut self,
        material_id: MaterialId,
        vertex_layout_id: VertexLayoutId,
        depth_mode: DepthMode,
        render_pass: vk::RenderPass,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout), Error> {
        let extent = self
            .surface_state
            .as_ref()
            .ok_or(surface_not_attached_error())?
            .swapchain
            .raw()
            .extent();
        let pipeline_key = MaterialPipelineKey {
            material_id,
            vertex_layout_id,
            depth_mode,
        };

        if !self.material_pipelines.contains_key(&pipeline_key) {
            let material_desc = self
                .materials
                .get(&material_id)
                .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?
                .descriptor
                .clone();
            let vertex_layout = self
                .vertex_layouts
                .get(&vertex_layout_id)
                .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?
                .clone();
            let pipeline = self.create_pipeline_for_material(
                &material_desc,
                &vertex_layout,
                render_pass,
                extent,
                depth_mode,
            )?;
            self.material_pipelines.insert(pipeline_key, pipeline);
            self.pipelines_by_material
                .entry(material_id)
                .or_default()
                .insert(pipeline_key);
        }

        let pipeline = self
            .material_pipelines
            .get(&pipeline_key)
            .expect("pipeline must exist");
        Ok((pipeline.pipeline.handle(), pipeline.layout.handle()))
    }

    pub(super) fn invalidate_material_pipelines(&mut self, material_id: MaterialId) {
        let Some(keys) = self.pipelines_by_material.remove(&material_id) else {
            return;
        };
        for key in keys {
            if let Some(pipeline) = self.material_pipelines.remove(&key) {
                pipeline.pipeline.destroy(self.device.raw());
                pipeline.layout.destroy(self.device.raw());
            }
        }
    }

    pub(super) fn destroy_all_pipelines(&mut self) {
        for (_, pipeline) in self.material_pipelines.drain() {
            pipeline.pipeline.destroy(self.device.raw());
            pipeline.layout.destroy(self.device.raw());
        }
        self.pipelines_by_material.clear();
    }
}

fn vertex_input_descriptor(layout: &VertexBufferLayout) -> Result<VertexInputDescriptor, Error> {
    if layout.array_stride > u32::MAX as u64 {
        return Err(Error::fatal(ErrorKind::Unsupported(
            "Vertex layout stride exceeds Vulkan limits",
        )));
    }
    let mut descriptor = VertexInputDescriptor::default().with_binding(
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: layout.array_stride as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        },
    );

    for attribute in &layout.attributes {
        if attribute.offset > u32::MAX as u64 {
            return Err(Error::fatal(ErrorKind::Unsupported(
                "Vertex attribute offset exceeds Vulkan limits",
            )));
        }
        descriptor = descriptor.with_attribute(vk::VertexInputAttributeDescription {
            location: attribute.location,
            binding: 0,
            format: map_vertex_format(attribute.format)?,
            offset: attribute.offset as u32,
        });
    }

    Ok(descriptor)
}

fn map_vertex_format(format: VertexFormat) -> Result<vk::Format, Error> {
    let mapped = match format {
        VertexFormat::Float32 => vk::Format::R32_SFLOAT,
        VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
        VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
        VertexFormat::Uint32 => vk::Format::R32_UINT,
    };
    Ok(mapped)
}

pub(super) fn spv_bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
