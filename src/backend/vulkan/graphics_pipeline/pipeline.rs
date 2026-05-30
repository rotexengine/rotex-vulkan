use ash::vk;

use super::shader::ShaderStageDescriptor;
use super::state::{
    ColorBlendState, DepthStencilState, MultisampleState, RasterizationState, Viewport,
};
use super::vertex::VertexInputDescriptor;
use super::super::device::Device;
use crate::error::vk_error;
use crate::{Error, ErrorKind};

pub struct GraphicsPipeline {
    handle: vk::Pipeline,
}

impl GraphicsPipeline {
    pub fn new(
        device: &Device,
        create_info: &vk::GraphicsPipelineCreateInfo,
    ) -> Result<Self, Error> {
        let handle = unsafe {
            device.logical_device().create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[*create_info],
                None,
            )
        }
        .map_err(|(_, err)| vk_error(err))?
        .remove(0);

        Ok(Self { handle })
    }

    pub fn handle(&self) -> vk::Pipeline {
        self.handle
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.logical_device().destroy_pipeline(self.handle, None);
        }
    }
}

pub struct GraphicsPipelineBuilder<'a> {
    shader_stages: Vec<ShaderStageDescriptor<'a>>,
    render_pass: Option<vk::RenderPass>,
    layout: Option<vk::PipelineLayout>,
    vertex_input_state: VertexInputDescriptor,
    input_assembly_state: vk::PipelineInputAssemblyStateCreateInfo<'a>,
    viewport_state: Viewport,
    rasterization_state: RasterizationState,
    multisample_state: MultisampleState,
    color_blend_state: ColorBlendState,
    subpass: Option<u32>,
    depth_stencil_state: Option<DepthStencilState>,
}

impl<'a> GraphicsPipelineBuilder<'a> {
    pub fn new() -> Self {
        Self {
            shader_stages: Vec::new(),
            render_pass: None,
            layout: None,
            vertex_input_state: VertexInputDescriptor::default(),
            input_assembly_state: vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
            viewport_state: Viewport::default(),
            rasterization_state: RasterizationState::default(),
            multisample_state: MultisampleState::default(),
            color_blend_state: ColorBlendState::default(),
            subpass: None,
            depth_stencil_state: None,
        }
    }

    pub fn with_shader_stage(mut self, stage: ShaderStageDescriptor<'a>) -> Self {
        self.shader_stages.push(stage);
        self
    }

    pub fn with_render_pass(mut self, render_pass: vk::RenderPass) -> Self {
        self.render_pass = Some(render_pass);
        self
    }

    pub fn with_layout(mut self, layout: vk::PipelineLayout) -> Self {
        self.layout = Some(layout);
        self
    }

    pub fn with_vertex_input_state(mut self, state: VertexInputDescriptor) -> Self {
        self.vertex_input_state = state;
        self
    }

    pub fn with_input_assembly_state(
        mut self,
        state: vk::PipelineInputAssemblyStateCreateInfo<'a>,
    ) -> Self {
        self.input_assembly_state = state;
        self
    }

    pub fn with_viewport_state(mut self, viewport: Viewport) -> Self {
        self.viewport_state = viewport;
        self
    }

    pub fn with_rasterization_state(mut self, state: RasterizationState) -> Self {
        self.rasterization_state = state;
        self
    }

    pub fn with_multisample_state(mut self, state: MultisampleState) -> Self {
        self.multisample_state = state;
        self
    }

    pub fn with_color_blend_state(mut self, state: ColorBlendState) -> Self {
        self.color_blend_state = state;
        self
    }

    pub fn with_subpass(mut self, subpass: u32) -> Self {
        self.subpass = Some(subpass);
        self
    }

    pub fn with_extent(mut self, width: u32, height: u32) -> Self {
        self.viewport_state = self.viewport_state.with_extent(width, height);
        self
    }

    pub fn with_depth_stencil_state(mut self, state: DepthStencilState) -> Self {
        self.depth_stencil_state = Some(state);
        self
    }

    pub fn build(self, device: &Device) -> Result<GraphicsPipeline, Error> {
        if self.shader_stages.is_empty() {
            return Err(Error::fatal(ErrorKind::Vulkan(
                vk::Result::ERROR_INITIALIZATION_FAILED,
            )));
        }

        if self.layout.is_none() {
            return Err(Error::fatal(ErrorKind::Vulkan(
                vk::Result::ERROR_INITIALIZATION_FAILED,
            )));
        }

        if self.render_pass.is_none() {
            return Err(Error::fatal(ErrorKind::Vulkan(
                vk::Result::ERROR_INITIALIZATION_FAILED,
            )));
        }

        let vk_shader_stages: Vec<vk::PipelineShaderStageCreateInfo> = self
            .shader_stages
            .iter()
            .map(|stage| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(stage.stage)
                    .module(stage.module.handle)
                    .name(stage.entry_name)
            })
            .collect();

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&self.vertex_input_state.binding_descriptions)
            .vertex_attribute_descriptions(&self.vertex_input_state.attribute_descriptions)
            .flags(self.vertex_input_state.flags);

        let viewports = [self.viewport_state.to_vk_viewport()];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: self.viewport_state.width,
                height: self.viewport_state.height,
            },
        }];

        let viewport_info = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer_info = self.rasterization_state.to_vk_rasterization_state();

        let multisampler_info = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(self.multisample_state.sample_shading_enable)
            .rasterization_samples(self.multisample_state.rasterization_samples)
            .min_sample_shading(self.multisample_state.min_sample_shading)
            .alpha_to_coverage_enable(self.multisample_state.alpha_to_coverage_enable)
            .alpha_to_one_enable(self.multisample_state.alpha_to_one_enable)
            .flags(self.multisample_state.flags);

        let colourblend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = self
            .color_blend_state
            .attachments
            .iter()
            .map(|attachment| attachment.to_vk_color_blend_attachment_state())
            .collect();

        let colourblend_info = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(self.color_blend_state.logic_op_enable)
            .logic_op(self.color_blend_state.logic_op)
            .attachments(&colourblend_attachments)
            .blend_constants(self.color_blend_state.blend_constants)
            .flags(self.color_blend_state.flags);

        let subpass = self.subpass.unwrap_or(0);
        let depth_stencil_state = self
            .depth_stencil_state
            .unwrap_or_else(DepthStencilState::default);
        let depth_stencil_info = depth_stencil_state.to_vk_depth_stencil_state();

        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&vk_shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&self.input_assembly_state)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampler_info)
            .color_blend_state(&colourblend_info)
            .depth_stencil_state(&depth_stencil_info)
            .layout(self.layout.unwrap())
            .render_pass(self.render_pass.unwrap())
            .subpass(subpass);

        GraphicsPipeline::new(device, &create_info)
    }
}
