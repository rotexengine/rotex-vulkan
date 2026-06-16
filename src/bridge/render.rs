use ash::vk;

use crate::backend::vulkan::{
    Device, Framebuffer, FramebufferBuilder, ImageDescriptor, RenderPass, RenderPassBuilder,
    RotexImage, Swapchain, SubpassBlueprint,
};
use crate::core::Instance;
use crate::error::{Error, ErrorKind, vk_error};

pub struct RenderPassConfig {
    pub color_load: vk::AttachmentLoadOp,
    pub color_initial_layout: vk::ImageLayout,
    pub color_final_layout: vk::ImageLayout,
    pub depth_load: Option<vk::AttachmentLoadOp>,
    pub depth_store: vk::AttachmentStoreOp,
    pub depth_initial_layout: vk::ImageLayout,
}

pub(super) fn create_render_pass(
    device: &Device,
    format: vk::Format,
    depth_format: Option<vk::Format>,
    config: RenderPassConfig,
) -> Result<RenderPass, Error> {
    let mut builder = RenderPassBuilder::new().with_attachment(
        vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(config.color_load)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(config.color_initial_layout)
            .final_layout(config.color_final_layout),
    );

    let subpass = if let Some(depth_format) = depth_format {
        let depth_load = config.depth_load.unwrap_or(vk::AttachmentLoadOp::CLEAR);
        builder = builder.with_attachment(
            vk::AttachmentDescription::default()
                .format(depth_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(depth_load)
                .store_op(config.depth_store)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(config.depth_initial_layout)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        );
        SubpassBlueprint {
            color_attachments: vec![0],
            depth_attachment: Some(1),
        }
    } else {
        SubpassBlueprint {
            color_attachments: vec![0],
            depth_attachment: None,
        }
    };

    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );

    builder
        .with_subpass(subpass)
        .with_dependency(dependency)
        .build(device)
        .map_err(vk_error)
}

pub(super) fn build_texture_framebuffer(
    device: &Device,
    render_pass: vk::RenderPass,
    color_view: vk::ImageView,
    depth_image: Option<&RotexImage>,
    width: u32,
    height: u32,
) -> Result<Vec<Framebuffer>, Error> {
    let mut builder = FramebufferBuilder::new().with_attachment(color_view);
    if let Some(depth) = depth_image {
        builder = builder.with_attachment(depth.view());
    }
    Ok(vec![builder
        .with_extent(width, height)
        .build(device, render_pass)?])
}

pub(super) fn build_framebuffers(
    device: &Device,
    swapchain: &Swapchain,
    render_pass: vk::RenderPass,
    depth_image: Option<&RotexImage>,
) -> Result<Vec<Framebuffer>, Error> {
    swapchain
        .image_views()
        .iter()
        .map(|view| {
            let mut builder = FramebufferBuilder::new().with_attachment(*view);
            if let Some(depth_image) = depth_image {
                builder = builder.with_attachment(depth_image.view());
            }
            builder
                .with_extent(swapchain.extent().width, swapchain.extent().height)
                .build(device, render_pass)
        })
        .collect()
}

pub(super) fn create_depth_image(
    instance: &Instance,
    device: &Device,
    extent: vk::Extent2D,
    format: vk::Format,
) -> Result<RotexImage, Error> {
    RotexImage::new(
        instance,
        device,
        ImageDescriptor::default(
            format,
            vk::Extent3D {
                width: extent.width.max(1),
                height: extent.height.max(1),
                depth: 1,
            },
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ),
    )
}

pub(super) fn find_depth_format(instance: &Instance, device: &Device) -> Result<vk::Format, Error> {
    let candidates = [
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];
    for format in candidates {
        let props = unsafe {
            instance
                .instance()
                .get_physical_device_format_properties(device.physical_device(), format)
        };
        if props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            return Ok(format);
        }
    }
    Err(Error::fatal(ErrorKind::NoCompatibleDevice))
}

pub(super) fn is_swapchain_outdated(err: &Error) -> bool {
    matches!(
        err.vk_result_code(),
        Some(code)
            if code == vk::Result::ERROR_OUT_OF_DATE_KHR.as_raw()
                || code == vk::Result::SUBOPTIMAL_KHR.as_raw()
    )
}
