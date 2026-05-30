use ash::vk;

use super::command::CommandBuffer;
use super::device::Device;
use crate::core::Instance;
use crate::error::vk_error;
use crate::{Error, ErrorKind};

pub struct ImageDescriptor {
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub usage: vk::ImageUsageFlags,
    pub properties: vk::MemoryPropertyFlags,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub image_type: vk::ImageType,
    pub view_type: vk::ImageViewType,
    pub tiling: vk::ImageTiling,
    pub samples: vk::SampleCountFlags,
}

impl ImageDescriptor {
    pub fn default(
        format: vk::Format,
        extent: vk::Extent3D,
        usage: vk::ImageUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Self {
        Self {
            format,
            extent,
            usage,
            properties,
            mip_levels: 1,
            array_layers: 1,
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }

    pub fn with_mip_levels(mut self, levels: u32) -> Self {
        self.mip_levels = levels;
        self
    }

    pub fn with_array_layers(mut self, layers: u32, view_type: vk::ImageViewType) -> Self {
        self.array_layers = layers;
        self.view_type = view_type;
        self
    }
}

pub struct RotexImage {
    image_handle: vk::Image,
    device_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    current_layout: std::cell::Cell<vk::ImageLayout>,
    aspect_mask: vk::ImageAspectFlags,
}

impl RotexImage {
    pub fn new(instance: &Instance, device: &Device, desc: ImageDescriptor) -> Result<Self, Error> {
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(desc.image_type)
            .format(desc.format)
            .extent(desc.extent)
            .mip_levels(desc.mip_levels)
            .array_layers(desc.array_layers)
            .samples(desc.samples)
            .tiling(desc.tiling)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image_handle = unsafe { device.logical_device().create_image(&image_create_info, None) }
            .map_err(vk_error)?;

        let mem_requirements = unsafe {
            device
                .logical_device()
                .get_image_memory_requirements(image_handle)
        };

        let memory_type_index = device.find_memory_type(
            instance,
            mem_requirements.memory_type_bits,
            desc.properties,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        let device_memory = unsafe { device.logical_device().allocate_memory(&alloc_info, None) }
            .map_err(vk_error)?;

        unsafe {
            device
                .logical_device()
                .bind_image_memory(image_handle, device_memory, 0)
        }
        .map_err(vk_error)?;

        let aspect_mask = Self::infer_aspect_mask(desc.format);

        let view_create_info = vk::ImageViewCreateInfo::default()
            .image(image_handle)
            .view_type(desc.view_type)
            .format(desc.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: desc.mip_levels,
                base_array_layer: 0,
                layer_count: desc.array_layers,
            });

        let image_view = unsafe { device.logical_device().create_image_view(&view_create_info, None) }
            .map_err(vk_error)?;

        Ok(Self {
            image_handle,
            device_memory,
            image_view,
            current_layout: std::cell::Cell::new(vk::ImageLayout::UNDEFINED),
            aspect_mask,
        })
    }

    fn infer_aspect_mask(format: vk::Format) -> vk::ImageAspectFlags {
        let is_depth = matches!(
            format,
            vk::Format::D32_SFLOAT
                | vk::Format::D32_SFLOAT_S8_UINT
                | vk::Format::D24_UNORM_S8_UINT
                | vk::Format::D16_UNORM
                | vk::Format::D16_UNORM_S8_UINT
        );

        let is_stencil = matches!(
            format,
            vk::Format::D32_SFLOAT_S8_UINT
                | vk::Format::D24_UNORM_S8_UINT
                | vk::Format::D16_UNORM_S8_UINT
                | vk::Format::S8_UINT
        );

        if is_depth && is_stencil {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        } else if is_depth {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        }
    }

    pub fn transition_layout(
        &self,
        device: &Device,
        command_buffer: &CommandBuffer,
        new_layout: vk::ImageLayout,
    ) {
        let old_layout = self.current_layout.get();

        if old_layout == new_layout {
            return;
        }

        command_buffer.transition_image_layout(
            device,
            self.image_handle,
            old_layout,
            new_layout,
            self.aspect_mask,
        );

        self.current_layout.set(new_layout);
    }

    pub fn handle(&self) -> vk::Image {
        self.image_handle
    }

    pub fn view(&self) -> vk::ImageView {
        self.image_view
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device
                .logical_device()
                .destroy_image_view(self.image_view, None);
            device
                .logical_device()
                .destroy_image(self.image_handle, None);
            device
                .logical_device()
                .free_memory(self.device_memory, None);
        }
    }
}

pub struct SamplerDescriptor {
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub anisotropy_enable: bool,
    pub max_anisotropy: f32,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
    pub border_color: vk::BorderColor,
    pub unnormalized_coordinates: bool,
    pub compare_enable: bool,
    pub mipmap_mode: vk::SamplerMipmapMode,
}

impl SamplerDescriptor {
    pub fn default() -> Self {
        Self {
            mag_filter: vk::Filter::NEAREST,
            min_filter: vk::Filter::NEAREST,
            anisotropy_enable: false,
            max_anisotropy: 1.0,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: false,
            compare_enable: false,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
        }
    }

    pub fn with_address_modes(
        mut self,
        u: vk::SamplerAddressMode,
        v: vk::SamplerAddressMode,
        w: vk::SamplerAddressMode,
    ) -> Self {
        self.address_mode_u = u;
        self.address_mode_v = v;
        self.address_mode_w = w;
        self
    }

    pub fn with_anisotropy(mut self, enable: bool, max_anisotropy: f32) -> Self {
        self.anisotropy_enable = enable;
        self.max_anisotropy = max_anisotropy;
        self
    }

    pub fn with_filters(mut self, mag_filter: vk::Filter, min_filter: vk::Filter) -> Self {
        self.mag_filter = mag_filter;
        self.min_filter = min_filter;
        self
    }

    pub fn with_border_color(mut self, border_color: vk::BorderColor) -> Self {
        self.border_color = border_color;
        self
    }

    pub fn with_unnormalized_coordinates(mut self, unnormalized: bool) -> Self {
        self.unnormalized_coordinates = unnormalized;
        self
    }

    pub fn with_compare_enable(mut self, compare_enable: bool) -> Self {
        self.compare_enable = compare_enable;
        self
    }

    pub fn with_mipmap_mode(mut self, mipmap_mode: vk::SamplerMipmapMode) -> Self {
        self.mipmap_mode = mipmap_mode;
        self
    }
}

pub struct RotexSampler {
    handle: vk::Sampler,
}

impl RotexSampler {
    pub fn new(device: &Device, descriptor: SamplerDescriptor) -> Result<Self, Error> {
        let create_info = vk::SamplerCreateInfo::default()
            .mag_filter(descriptor.mag_filter)
            .min_filter(descriptor.min_filter)
            .address_mode_u(descriptor.address_mode_u)
            .address_mode_v(descriptor.address_mode_v)
            .address_mode_w(descriptor.address_mode_w)
            .anisotropy_enable(descriptor.anisotropy_enable)
            .max_anisotropy(descriptor.max_anisotropy)
            .border_color(descriptor.border_color)
            .unnormalized_coordinates(descriptor.unnormalized_coordinates)
            .compare_enable(descriptor.compare_enable)
            .mipmap_mode(descriptor.mipmap_mode);

        let handle = unsafe { device.logical_device().create_sampler(&create_info, None) }
            .map_err(ErrorKind::Vulkan)
            .map_err(Error::fatal)?;

        Ok(Self { handle })
    }

    pub fn handle(&self) -> vk::Sampler {
        self.handle
    }

    pub fn destroy(&self, device: &Device) {
        unsafe { device.logical_device().destroy_sampler(self.handle, None) };
    }
}
