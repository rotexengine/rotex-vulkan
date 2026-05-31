use ash::vk;

/// Viewport and scissor extent for `VkPipelineViewportStateCreateInfo`.
pub(crate) struct Viewport {
    /// `VkViewport::x`.
    pub x: f32,
    /// `VkViewport::y`.
    pub y: f32,
    /// `VkViewport::width` and scissor `extent.width`.
    pub width: u32,
    /// `VkViewport::height` and scissor `extent.height`.
    pub height: u32,
    /// `VkViewport::minDepth`.
    pub min_depth: f32,
    /// `VkViewport::maxDepth`.
    pub max_depth: f32,
}

impl Viewport {
    /// Origin at (0, 0), zero extent, depth range [0, 1].
    pub fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 0,
            height: 0,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    /// Sets `x` and `y`.
    pub fn with_position(mut self, x: f32, y: f32) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    /// Sets `width` and `height`.
    pub fn with_extent(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Sets `min_depth` and `max_depth`.
    pub fn with_depth_range(mut self, min_depth: f32, max_depth: f32) -> Self {
        self.min_depth = min_depth;
        self.max_depth = max_depth;
        self
    }

    /// Converts to `VkViewport`.
    pub fn to_vk_viewport(&self) -> vk::Viewport {
        vk::Viewport {
            x: self.x,
            y: self.y,
            width: self.width as f32,
            height: self.height as f32,
            min_depth: self.min_depth,
            max_depth: self.max_depth,
        }
    }
}

/// Fixed-function rasterization state (`VkPipelineRasterizationStateCreateInfo`).
pub struct RasterizationState {
    /// `depthClampEnable`.
    pub depth_clamp_enable: bool,
    /// `rasterizerDiscardEnable`.
    pub rasterizer_discard_enable: bool,
    /// `polygonMode`.
    pub polygon_mode: vk::PolygonMode,
    /// `cullMode`.
    pub cull_mode: vk::CullModeFlags,
    /// `frontFace`.
    pub front_face: vk::FrontFace,
    /// `depthBiasEnable`.
    pub depth_bias_enable: bool,
    /// `depthBiasConstantFactor`.
    pub depth_bias_constant_factor: f32,
    /// `depthBiasClamp`.
    pub depth_bias_clamp: f32,
    /// `depthBiasSlopeFactor`.
    pub depth_bias_slope_factor: f32,
    /// `flags`.
    pub flags: vk::PipelineRasterizationStateCreateFlags,
    /// `lineWidth`.
    pub line_width: f32,
}

impl RasterizationState {
    /// Filled triangles, back-face cull, clockwise front, line width 1.
    pub fn default() -> Self {
        Self {
            depth_clamp_enable: false,
            rasterizer_discard_enable: false,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,
            depth_bias_enable: false,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            flags: vk::PipelineRasterizationStateCreateFlags::empty(),
            line_width: 1.0,
        }
    }

    /// Sets `depth_clamp_enable`.
    pub fn with_depth_clamp_enable(mut self, enable: bool) -> Self {
        self.depth_clamp_enable = enable;
        self
    }

    /// Sets `rasterizer_discard_enable`.
    pub fn with_rasterizer_discard_enable(mut self, enable: bool) -> Self {
        self.rasterizer_discard_enable = enable;
        self
    }

    /// Sets `polygon_mode`.
    pub fn with_polygon_mode(mut self, mode: vk::PolygonMode) -> Self {
        self.polygon_mode = mode;
        self
    }

    /// Sets `cull_mode`.
    pub fn with_cull_mode(mut self, mode: vk::CullModeFlags) -> Self {
        self.cull_mode = mode;
        self
    }

    /// Sets `front_face`.
    pub fn with_front_face(mut self, face: vk::FrontFace) -> Self {
        self.front_face = face;
        self
    }

    /// Sets `depth_bias_enable`.
    pub fn with_depth_bias_enable(mut self, enable: bool) -> Self {
        self.depth_bias_enable = enable;
        self
    }

    /// Sets depth bias constant factor, clamp, and slope factor.
    pub fn with_depth_bias(mut self, constant_factor: f32, clamp: f32, slope_factor: f32) -> Self {
        self.depth_bias_constant_factor = constant_factor;
        self.depth_bias_clamp = clamp;
        self.depth_bias_slope_factor = slope_factor;
        self
    }

    /// ORs `flags` into rasterization create flags.
    pub fn with_flags(mut self, flags: vk::PipelineRasterizationStateCreateFlags) -> Self {
        self.flags |= flags;
        self
    }

    /// Sets `line_width`.
    pub fn with_line_width(mut self, width: f32) -> Self {
        self.line_width = width;
        self
    }

    /// Converts to `VkPipelineRasterizationStateCreateInfo`.
    pub fn to_vk_rasterization_state(&self) -> vk::PipelineRasterizationStateCreateInfo<'_> {
        vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(self.depth_clamp_enable)
            .rasterizer_discard_enable(self.rasterizer_discard_enable)
            .polygon_mode(self.polygon_mode)
            .cull_mode(self.cull_mode)
            .front_face(self.front_face)
            .depth_bias_enable(self.depth_bias_enable)
            .depth_bias_constant_factor(self.depth_bias_constant_factor)
            .depth_bias_clamp(self.depth_bias_clamp)
            .depth_bias_slope_factor(self.depth_bias_slope_factor)
            .flags(self.flags)
            .line_width(self.line_width)
    }
}

/// Depth and stencil test state (`VkPipelineDepthStencilStateCreateInfo`).
pub struct DepthStencilState {
    /// `depthTestEnable`.
    pub depth_test_enable: bool,
    /// `depthWriteEnable`.
    pub depth_write_enable: bool,
    /// `depthCompareOp`.
    pub depth_compare_op: vk::CompareOp,
    /// `depthBoundsTestEnable`.
    pub depth_bounds_test_enable: bool,
    /// `stencilTestEnable`.
    pub stencil_test_enable: bool,
    /// `minDepthBounds`.
    pub min_depth_bounds: f32,
    /// `maxDepthBounds`.
    pub max_depth_bounds: f32,
}

impl DepthStencilState {
    /// Depth and stencil tests disabled; compare op `LESS` if enabled later.
    pub fn default() -> Self {
        Self {
            depth_test_enable: false,
            depth_write_enable: false,
            depth_compare_op: vk::CompareOp::LESS,
            depth_bounds_test_enable: false,
            stencil_test_enable: false,
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
        }
    }

    /// Sets `depth_test_enable`.
    pub fn with_depth_test_enable(mut self, enable: bool) -> Self {
        self.depth_test_enable = enable;
        self
    }

    /// Sets `depth_write_enable`.
    pub fn with_depth_write_enable(mut self, enable: bool) -> Self {
        self.depth_write_enable = enable;
        self
    }

    /// Sets `depth_compare_op`.
    pub fn with_depth_compare_op(mut self, op: vk::CompareOp) -> Self {
        self.depth_compare_op = op;
        self
    }

    /// Sets `depth_bounds_test_enable`.
    pub fn with_depth_bounds_test_enable(mut self, enable: bool) -> Self {
        self.depth_bounds_test_enable = enable;
        self
    }

    /// Sets `stencil_test_enable`.
    pub fn with_stencil_test_enable(mut self, enable: bool) -> Self {
        self.stencil_test_enable = enable;
        self
    }

    /// Sets `min_depth_bounds` and `max_depth_bounds`.
    pub fn with_depth_bounds(mut self, min: f32, max: f32) -> Self {
        self.min_depth_bounds = min;
        self.max_depth_bounds = max;
        self
    }

    /// Converts to `VkPipelineDepthStencilStateCreateInfo`.
    pub fn to_vk_depth_stencil_state(&self) -> vk::PipelineDepthStencilStateCreateInfo<'_> {
        vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(self.depth_test_enable)
            .depth_write_enable(self.depth_write_enable)
            .depth_compare_op(self.depth_compare_op)
            .depth_bounds_test_enable(self.depth_bounds_test_enable)
            .stencil_test_enable(self.stencil_test_enable)
            .min_depth_bounds(self.min_depth_bounds)
            .max_depth_bounds(self.max_depth_bounds)
    }
}

/// Multisample rasterization state (`VkPipelineMultisampleStateCreateInfo`).
pub(crate) struct MultisampleState {
    /// `sampleShadingEnable`.
    pub sample_shading_enable: bool,
    /// `rasterizationSamples`.
    pub rasterization_samples: vk::SampleCountFlags,
    /// `minSampleShading`.
    pub min_sample_shading: f32,
    /// `pSampleMask`.
    pub sample_mask: Option<Vec<u32>>,
    /// `alphaToCoverageEnable`.
    pub alpha_to_coverage_enable: bool,
    /// `alphaToOneEnable`.
    pub alpha_to_one_enable: bool,
    /// `flags`.
    pub flags: vk::PipelineMultisampleStateCreateFlags,
}

impl MultisampleState {
    /// Single-sample rasterization, no sample shading or alpha-to-coverage.
    pub fn default() -> Self {
        Self {
            sample_shading_enable: false,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            min_sample_shading: 1.0,
            sample_mask: None,
            alpha_to_coverage_enable: false,
            alpha_to_one_enable: false,
            flags: vk::PipelineMultisampleStateCreateFlags::empty(),
        }
    }

    /// Sets `sample_shading_enable`.
    pub fn with_sample_shading_enable(mut self, enable: bool) -> Self {
        self.sample_shading_enable = enable;
        self
    }

    /// Sets `rasterization_samples`.
    pub fn with_rasterization_samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.rasterization_samples = samples;
        self
    }

    /// Sets `min_sample_shading`.
    pub fn with_min_sample_shading(mut self, min_shading: f32) -> Self {
        self.min_sample_shading = min_shading;
        self
    }

    /// Sets `sample_mask`.
    pub fn with_sample_mask(mut self, mask: Vec<u32>) -> Self {
        self.sample_mask = Some(mask);
        self
    }

    /// Sets `alpha_to_coverage_enable`.
    pub fn with_alpha_to_coverage_enable(mut self, enable: bool) -> Self {
        self.alpha_to_coverage_enable = enable;
        self
    }

    /// Sets `alpha_to_one_enable`.
    pub fn with_alpha_to_one_enable(mut self, enable: bool) -> Self {
        self.alpha_to_one_enable = enable;
        self
    }

    /// ORs `flags` into multisample create flags.
    pub fn with_flags(mut self, flags: vk::PipelineMultisampleStateCreateFlags) -> Self {
        self.flags |= flags;
        self
    }
}

/// Per-attachment color blend state (`VkPipelineColorBlendAttachmentState`).
pub struct ColorBlendAttachmentState {
    /// `blendEnable`.
    pub blend_enable: bool,
    /// `srcColorBlendFactor`.
    pub src_color_blend_factor: vk::BlendFactor,
    /// `dstColorBlendFactor`.
    pub dst_color_blend_factor: vk::BlendFactor,
    /// `colorBlendOp`.
    pub color_blend_op: vk::BlendOp,
    /// `srcAlphaBlendFactor`.
    pub src_alpha_blend_factor: vk::BlendFactor,
    /// `dstAlphaBlendFactor`.
    pub dst_alpha_blend_factor: vk::BlendFactor,
    /// `alphaBlendOp`.
    pub alpha_blend_op: vk::BlendOp,
    /// `colorWriteMask`.
    pub color_write_mask: vk::ColorComponentFlags,
}

impl ColorBlendAttachmentState {
    /// Blending disabled; full RGBA write mask; factors `ONE` / `ZERO`, op `ADD`.
    pub fn default() -> Self {
        Self {
            blend_enable: false,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        }
    }

    /// Sets `blend_enable`.
    pub fn with_blend_enable(mut self, enable: bool) -> Self {
        self.blend_enable = enable;
        self
    }

    /// Sets `src_color_blend_factor`.
    pub fn with_src_color_blend_factor(mut self, factor: vk::BlendFactor) -> Self {
        self.src_color_blend_factor = factor;
        self
    }

    /// Sets `dst_color_blend_factor`.
    pub fn with_dst_color_blend_factor(mut self, factor: vk::BlendFactor) -> Self {
        self.dst_color_blend_factor = factor;
        self
    }

    /// Sets `color_blend_op`.
    pub fn with_color_blend_op(mut self, op: vk::BlendOp) -> Self {
        self.color_blend_op = op;
        self
    }

    /// Sets `src_alpha_blend_factor`.
    pub fn with_src_alpha_blend_factor(mut self, factor: vk::BlendFactor) -> Self {
        self.src_alpha_blend_factor = factor;
        self
    }

    /// Sets `dst_alpha_blend_factor`.
    pub fn with_dst_alpha_blend_factor(mut self, factor: vk::BlendFactor) -> Self {
        self.dst_alpha_blend_factor = factor;
        self
    }

    /// Sets `alpha_blend_op`.
    pub fn with_alpha_blend_op(mut self, op: vk::BlendOp) -> Self {
        self.alpha_blend_op = op;
        self
    }

    /// Sets `color_write_mask`.
    pub fn with_color_write_mask(mut self, mask: vk::ColorComponentFlags) -> Self {
        self.color_write_mask = mask;
        self
    }

    pub(crate) fn to_vk_color_blend_attachment_state(
        &self,
    ) -> vk::PipelineColorBlendAttachmentState {
        vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(self.blend_enable)
            .src_color_blend_factor(self.src_color_blend_factor)
            .dst_color_blend_factor(self.dst_color_blend_factor)
            .color_blend_op(self.color_blend_op)
            .src_alpha_blend_factor(self.src_alpha_blend_factor)
            .dst_alpha_blend_factor(self.dst_alpha_blend_factor)
            .alpha_blend_op(self.alpha_blend_op)
            .color_write_mask(self.color_write_mask)
    }
}

/// Global color blend state (`VkPipelineColorBlendStateCreateInfo`).
pub struct ColorBlendState {
    /// `logicOpEnable`.
    pub logic_op_enable: bool,
    /// `logicOp`.
    pub logic_op: vk::LogicOp,
    /// `pAttachments` (one entry per color attachment).
    pub attachments: Vec<ColorBlendAttachmentState>,
    /// `blendConstants`.
    pub blend_constants: [f32; 4],
    /// `flags`.
    pub flags: vk::PipelineColorBlendStateCreateFlags,
}

impl ColorBlendState {
    /// Logic op disabled, no attachments, zero blend constants.
    pub fn default() -> Self {
        Self {
            logic_op_enable: false,
            logic_op: vk::LogicOp::CLEAR,
            attachments: Vec::new(),
            blend_constants: [0.0; 4],
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
        }
    }

    /// Sets `logic_op_enable`.
    pub fn with_logic_op_enable(mut self, enable: bool) -> Self {
        self.logic_op_enable = enable;
        self
    }

    /// Sets `logic_op`.
    pub fn with_logic_op(mut self, op: vk::LogicOp) -> Self {
        self.logic_op = op;
        self
    }

    /// Appends a color blend attachment state.
    pub fn with_attachment(mut self, attachment: ColorBlendAttachmentState) -> Self {
        self.attachments.push(attachment);
        self
    }

    /// Sets `blend_constants`.
    pub fn with_blend_constants(mut self, constants: [f32; 4]) -> Self {
        self.blend_constants = constants;
        self
    }

    /// ORs `flags` into color blend create flags.
    pub fn with_flags(mut self, flags: vk::PipelineColorBlendStateCreateFlags) -> Self {
        self.flags |= flags;
        self
    }
}
