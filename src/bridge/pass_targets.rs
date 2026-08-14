#![allow(dead_code)]
use std::collections::HashMap;

use ash::vk;

use super::render::{
    RenderPassConfig, build_framebuffers, build_texture_framebuffer, create_depth_image,
    create_render_pass, find_depth_format,
};
use super::types::RenderTargets;
use super::{VulkanBridge, surface_not_attached_error};
use crate::backend::vulkan::Device;
use crate::error::{Error, ErrorKind};
use rotex_types::{
    ColorAttachmentLoad, DepthAttachmentLoad, PassColorTarget, PassDescriptor, RhiCommand,
    TextureId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ColorTargetKey {
    Swapchain,
    Texture(TextureId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetPassRole {
    Intermediate,
    Terminal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct PassTargetKey {
    color_target: ColorTargetKey,
    color_load: ColorAttachmentLoad,
    depth_load: DepthAttachmentLoad,
    uses_depth: bool,
    target_role: TargetPassRole,
}

pub(super) struct ResolvedPassTarget {
    pub render_pass: vk::RenderPass,
    pub framebuffer_index: usize,
    pub key: PassTargetKey,
}

pub(super) struct PassTargetCache {
    entries: HashMap<PassTargetKey, RenderTargets>,
    depth_format: Option<vk::Format>,
}

impl PassTargetCache {
    pub(super) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            depth_format: None,
        }
    }

    pub(super) fn clear_swapchain_entries(&mut self, device: &Device) {
        let keys: Vec<_> = self
            .entries
            .keys()
            .filter(|key| key.color_target == ColorTargetKey::Swapchain)
            .copied()
            .collect();
        for key in keys {
            if let Some(targets) = self.entries.remove(&key) {
                targets.destroy(device);
            }
        }
    }

    #[allow(dead_code)]
    pub(super) fn invalidate_texture(&mut self, device: &Device, texture_id: TextureId) {
        let keys: Vec<_> = self
            .entries
            .keys()
            .filter(|key| key.color_target == ColorTargetKey::Texture(texture_id))
            .copied()
            .collect();
        for key in keys {
            if let Some(targets) = self.entries.remove(&key) {
                targets.destroy(device);
            }
        }
    }

    pub(super) fn destroy(&mut self, device: &Device) {
        for (_, targets) in self.entries.drain() {
            targets.destroy(device);
        }
    }
}

impl VulkanBridge {
    pub(super) fn resolve_pass_targets(
        &mut self,
        pass: &PassDescriptor,
        remaining_commands: &[RhiCommand],
        uses_depth: bool,
        image_index: u32,
    ) -> Result<ResolvedPassTarget, Error> {
        let depth_load = effective_depth_load(pass, uses_depth);
        let target_role = target_pass_role(remaining_commands, pass.color_target);
        let color_target_key = match pass.color_target {
            PassColorTarget::Swapchain => ColorTargetKey::Swapchain,
            PassColorTarget::Texture(id) => ColorTargetKey::Texture(id),
        };
        let key = PassTargetKey {
            color_target: color_target_key,
            color_load: pass.color_load,
            depth_load,
            uses_depth,
            target_role,
        };

        if !self.pass_target_cache.entries.contains_key(&key) {
            let targets =
                self.create_pass_targets(pass, remaining_commands, uses_depth, depth_load)?;
            self.pass_target_cache.entries.insert(key, targets);
        }

        let targets = self
            .pass_target_cache
            .entries
            .get(&key)
            .expect("pass target inserted");
        let framebuffer_index = match pass.color_target {
            PassColorTarget::Swapchain => image_index as usize,
            PassColorTarget::Texture(_) => 0,
        };
        if framebuffer_index >= targets.framebuffers.len() {
            return Err(Error::fatal(ErrorKind::NoCompatibleDevice));
        }

        Ok(ResolvedPassTarget {
            render_pass: targets.render_pass.handle(),
            framebuffer_index,
            key,
        })
    }

    pub(super) fn pass_targets_for_key(
        &self,
        key: &PassTargetKey,
    ) -> Result<&RenderTargets, Error> {
        self.pass_target_cache
            .entries
            .get(key)
            .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))
    }

    fn create_pass_targets(
        &mut self,
        pass: &PassDescriptor,
        remaining_commands: &[RhiCommand],
        uses_depth: bool,
        depth_load: DepthAttachmentLoad,
    ) -> Result<RenderTargets, Error> {
        let instance = self.instance.raw();
        let device = self.device.raw();
        let target_role = target_pass_role(remaining_commands, pass.color_target);
        let depth_stored = depth_needs_store(remaining_commands, pass.color_target, uses_depth);

        if self.pass_target_cache.depth_format.is_none() && uses_depth {
            self.pass_target_cache.depth_format = Some(find_depth_format(instance, device)?);
        }
        let depth_format = if uses_depth {
            self.pass_target_cache.depth_format
        } else {
            None
        };

        let color_final_layout = color_final_layout(pass.color_target, target_role);
        let vk_color_load = map_color_load(pass.color_load);
        let config = RenderPassConfig {
            color_load: vk_color_load,
            color_initial_layout: color_initial_layout(vk_color_load),
            color_final_layout,
            depth_load: depth_format.map(|_| map_depth_load(depth_load)),
            depth_store: if depth_stored {
                vk::AttachmentStoreOp::STORE
            } else {
                vk::AttachmentStoreOp::DONT_CARE
            },
            depth_initial_layout: depth_format
                .map(|_| depth_initial_layout(map_depth_load(depth_load)))
                .unwrap_or(vk::ImageLayout::UNDEFINED),
        };

        match pass.color_target {
            PassColorTarget::Swapchain => {
                let state = self
                    .surface_state
                    .as_ref()
                    .ok_or(surface_not_attached_error())?;
                let swapchain = state.swapchain.raw();
                let render_pass =
                    create_render_pass(device, swapchain.format(), depth_format, config)?;
                let depth_image = match depth_format {
                    Some(format) => Some(create_depth_image(
                        instance,
                        device,
                        swapchain.extent(),
                        format,
                    )?),
                    None => None,
                };
                let framebuffers = build_framebuffers(
                    device,
                    swapchain,
                    render_pass.handle(),
                    depth_image.as_ref(),
                )?;
                Ok(RenderTargets {
                    render_pass,
                    framebuffers,
                    depth_image,
                })
            }
            PassColorTarget::Texture(texture_id) => {
                let texture = self
                    .textures
                    .get(&texture_id)
                    .ok_or(Error::fatal(ErrorKind::NoCompatibleDevice))?;
                let format = super::resources::map_texture_format(texture.descriptor.format);
                let render_pass = create_render_pass(device, format, depth_format, config)?;
                let extent = vk::Extent2D {
                    width: texture.descriptor.width.max(1),
                    height: texture.descriptor.height.max(1),
                };
                let depth_image = match depth_format {
                    Some(format) => Some(create_depth_image(instance, device, extent, format)?),
                    None => None,
                };
                let framebuffers = build_texture_framebuffer(
                    device,
                    render_pass.handle(),
                    texture.image.view(),
                    depth_image.as_ref(),
                    extent.width,
                    extent.height,
                )?;
                Ok(RenderTargets {
                    render_pass,
                    framebuffers,
                    depth_image,
                })
            }
        }
    }

    pub(super) fn clear_pass_target_cache(&mut self) {
        self.pass_target_cache
            .clear_swapchain_entries(self.device.raw());
    }

    pub(super) fn destroy_pass_target_cache(&mut self) {
        self.pass_target_cache.destroy(self.device.raw());
    }
}

fn effective_depth_load(pass: &PassDescriptor, uses_depth: bool) -> DepthAttachmentLoad {
    if !uses_depth {
        return DepthAttachmentLoad::None;
    }
    if pass.uses_depth_attachment() {
        pass.depth_load
    } else {
        DepthAttachmentLoad::Clear
    }
}

fn graphics_passes<'a>(
    commands: &'a [RhiCommand],
) -> impl Iterator<Item = &'a PassDescriptor> + 'a {
    commands.iter().filter_map(|command| match command {
        RhiCommand::BeginRenderPass { pass, .. } => Some(pass),
        _ => None,
    })
}

fn target_pass_role(remaining_commands: &[RhiCommand], target: PassColorTarget) -> TargetPassRole {
    let has_later = graphics_passes(remaining_commands).any(|later| later.color_target == target);
    if has_later {
        TargetPassRole::Intermediate
    } else {
        TargetPassRole::Terminal
    }
}

fn depth_needs_store(
    remaining_commands: &[RhiCommand],
    target: PassColorTarget,
    uses_depth: bool,
) -> bool {
    if !uses_depth {
        return false;
    }
    graphics_passes(remaining_commands).any(|later| {
        later.color_target == target
            && (later.uses_depth_attachment()
                || matches!(later.depth_load, DepthAttachmentLoad::Load))
    })
}

fn color_final_layout(target: PassColorTarget, role: TargetPassRole) -> vk::ImageLayout {
    match (target, role) {
        (PassColorTarget::Swapchain, TargetPassRole::Terminal) => vk::ImageLayout::PRESENT_SRC_KHR,
        (PassColorTarget::Swapchain, TargetPassRole::Intermediate) => {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        }
        (PassColorTarget::Texture(_), TargetPassRole::Terminal) => {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        }
        (PassColorTarget::Texture(_), TargetPassRole::Intermediate) => {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        }
    }
}

fn map_color_load(load: ColorAttachmentLoad) -> vk::AttachmentLoadOp {
    match load {
        ColorAttachmentLoad::Clear => vk::AttachmentLoadOp::CLEAR,
        ColorAttachmentLoad::Load => vk::AttachmentLoadOp::LOAD,
    }
}

fn map_depth_load(load: DepthAttachmentLoad) -> vk::AttachmentLoadOp {
    match load {
        DepthAttachmentLoad::Clear => vk::AttachmentLoadOp::CLEAR,
        DepthAttachmentLoad::Load => vk::AttachmentLoadOp::LOAD,
        DepthAttachmentLoad::None => vk::AttachmentLoadOp::DONT_CARE,
    }
}

fn color_initial_layout(load: vk::AttachmentLoadOp) -> vk::ImageLayout {
    match load {
        vk::AttachmentLoadOp::LOAD => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        _ => vk::ImageLayout::UNDEFINED,
    }
}

fn depth_initial_layout(load: vk::AttachmentLoadOp) -> vk::ImageLayout {
    match load {
        vk::AttachmentLoadOp::LOAD => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        _ => vk::ImageLayout::UNDEFINED,
    }
}
