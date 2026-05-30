use ash::vk;

pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: u32,
    pub height: u32,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl Viewport {
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

    pub fn with_position(mut self, x: f32, y: f32) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    pub fn with_extent(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_depth_range(mut self, min_depth: f32, max_depth: f32) -> Self {
        self.min_depth = min_depth;
        self.max_depth = max_depth;
        self
    }

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

pub struct RasterizationState {
    pub depth_clamp_enable: bool,
    pub rasterizer_discard_enable: bool,
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub depth_bias_enable: bool,
    pub depth_bias_constant_factor: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope_factor: f32,
    pub flags: vk::PipelineRasterizationStateCreateFlags,
    pub line_width: f32,
}

impl RasterizationState {
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

    pub fn with_depth_clamp_enable(mut self, enable: bool) -> Self {
        self.depth_clamp_enable = enable;
        self
    }

    pub fn with_rasterizer_discard_enable(mut self, enable: bool) -> Self {
        self.rasterizer_discard_enable = enable;
        self
    }

    pub fn with_polygon_mode(mut self, mode: vk::PolygonMode) -> Self {
        self.polygon_mode = mode;
        self
    }

    pub fn with_cull_mode(mut self, mode: vk::CullModeFlags) -> Self {
        self.cull_mode = mode;
        self
    }

    pub fn with_front_face(mut self, face: vk::FrontFace) -> Self {
        self.front_face = face;
        self
    }

    pub fn with_depth_bias_enable(mut self, enable: bool) -> Self {
        self.depth_bias_enable = enable;
        self
    }

    pub fn with_depth_bias(mut self, constant_factor: f32, clamp: f32, slope_factor: f32) -> Self {
        self.depth_bias_constant_factor = constant_factor;
        self.depth_bias_clamp = clamp;
        self.depth_bias_slope_factor = slope_factor;
        self
    }

    pub fn with_flags(mut self, flags: vk::PipelineRasterizationStateCreateFlags) -> Self {
        self.flags |= flags;
        self
    }

    pub fn with_line_width(mut self, width: f32) -> Self {
        self.line_width = width;
        self
    }

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

pub struct DepthStencilState {
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: vk::CompareOp,
    pub depth_bounds_test_enable: bool,
    pub stencil_test_enable: bool,
    pub min_depth_bounds: f32,
    pub max_depth_bounds: f32,
}

impl DepthStencilState {
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

    pub fn with_depth_test_enable(mut self, enable: bool) -> Self {
        self.depth_test_enable = enable;
        self
    }

    pub fn with_depth_write_enable(mut self, enable: bool) -> Self {
        self.depth_write_enable = enable;
        self
    }

    pub fn with_depth_compare_op(mut self, op: vk::CompareOp) -> Self {
        self.depth_compare_op = op;
        self
    }

    pub fn with_depth_bounds_test_enable(mut self, enable: bool) -> Self {
        self.depth_bounds_test_enable = enable;
        self
    }

    pub fn with_stencil_test_enable(mut self, enable: bool) -> Self {
        self.stencil_test_enable = enable;
        self
    }

    pub fn with_depth_bounds(mut self, min: f32, max: f32) -> Self {
        self.min_depth_bounds = min;
        self.max_depth_bounds = max;
        self
    }

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

pub struct MultisampleState {
    pub sample_shading_enable: bool,
    pub rasterization_samples: vk::SampleCountFlags,
    pub min_sample_shading: f32,
    pub sample_mask: Option<Vec<u32>>,
    pub alpha_to_coverage_enable: bool,
    pub alpha_to_one_enable: bool,
    pub flags: vk::PipelineMultisampleStateCreateFlags,
}

impl MultisampleState {
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

    pub fn with_sample_shading_enable(mut self, enable: bool) -> Self {
        self.sample_shading_enable = enable;
        self
    }

    pub fn with_rasterization_samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.rasterization_samples = samples;
        self
    }

    pub fn with_min_sample_shading(mut self, min_shading: f32) -> Self {
        self.min_sample_shading = min_shading;
        self
    }

    pub fn with_sample_mask(mut self, mask: Vec<u32>) -> Self {
        self.sample_mask = Some(mask);
        self
    }

    pub fn with_alpha_to_coverage_enable(mut self, enable: bool) -> Self {
        self.alpha_to_coverage_enable = enable;
        self
    }

    pub fn with_alpha_to_one_enable(mut self, enable: bool) -> Self {
        self.alpha_to_one_enable = enable;
        self
    }

    pub fn with_flags(mut self, flags: vk::PipelineMultisampleStateCreateFlags) -> Self {
        self.flags |= flags;
        self
    }
}

pub struct ColorBlendAttachmentState {
    pub blend_enable: bool,
    pub src_color_blend_factor: vk::BlendFactor,
    pub dst_color_blend_factor: vk::BlendFactor,
    pub color_blend_op: vk::BlendOp,
    pub src_alpha_blend_factor: vk::BlendFactor,
    pub dst_alpha_blend_factor: vk::BlendFactor,
    pub alpha_blend_op: vk::BlendOp,
    pub color_write_mask: vk::ColorComponentFlags,
}

impl ColorBlendAttachmentState {
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

    pub fn with_blend_enable(mut self, enable: bool) -> Self {
        self.blend_enable = enable;
        self
    }

    pub fn with_src_color_blend_factor(mut self, factor: vk::BlendFactor) -> Self {
        self.src_color_blend_factor = factor;
        self
    }

    pub fn with_dst_color_blend_factor(mut self, factor: vk::BlendFactor) -> Self {
        self.dst_color_blend_factor = factor;
        self
    }

    pub fn with_color_blend_op(mut self, op: vk::BlendOp) -> Self {
        self.color_blend_op = op;
        self
    }

    pub fn with_src_alpha_blend_factor(mut self, factor: vk::BlendFactor) -> Self {
        self.src_alpha_blend_factor = factor;
        self
    }

    pub fn with_dst_alpha_blend_factor(mut self, factor: vk::BlendFactor) -> Self {
        self.dst_alpha_blend_factor = factor;
        self
    }

    pub fn with_alpha_blend_op(mut self, op: vk::BlendOp) -> Self {
        self.alpha_blend_op = op;
        self
    }

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

pub struct ColorBlendState {
    pub logic_op_enable: bool,
    pub logic_op: vk::LogicOp,
    pub attachments: Vec<ColorBlendAttachmentState>,
    pub blend_constants: [f32; 4],
    pub flags: vk::PipelineColorBlendStateCreateFlags,
}

impl ColorBlendState {
    pub fn default() -> Self {
        Self {
            logic_op_enable: false,
            logic_op: vk::LogicOp::CLEAR,
            attachments: Vec::new(),
            blend_constants: [0.0; 4],
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
        }
    }

    pub fn with_logic_op_enable(mut self, enable: bool) -> Self {
        self.logic_op_enable = enable;
        self
    }

    pub fn with_logic_op(mut self, op: vk::LogicOp) -> Self {
        self.logic_op = op;
        self
    }

    pub fn with_attachment(mut self, attachment: ColorBlendAttachmentState) -> Self {
        self.attachments.push(attachment);
        self
    }

    pub fn with_blend_constants(mut self, constants: [f32; 4]) -> Self {
        self.blend_constants = constants;
        self
    }

    pub fn with_flags(mut self, flags: vk::PipelineColorBlendStateCreateFlags) -> Self {
        self.flags |= flags;
        self
    }
}
