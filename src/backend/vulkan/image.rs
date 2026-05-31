use ash::vk;

use super::command::CommandBuffer;
use super::device::Device;
use crate::core::Instance;
use crate::error::vk_error;
use crate::{Error, ErrorKind};

/// Parameters for creating a [`RotexImage`] and its view.
pub struct ImageDescriptor {
    /// `VkImageCreateInfo::format`.
    pub format: vk::Format,
    /// `VkImageCreateInfo::extent`.
    pub extent: vk::Extent3D,
    /// `VkImageCreateInfo::usage`.
    pub usage: vk::ImageUsageFlags,
    /// Memory type flags used when allocating device memory.
    pub properties: vk::MemoryPropertyFlags,
    /// `VkImageCreateInfo::mipLevels`.
    pub mip_levels: u32,
    /// `VkImageCreateInfo::arrayLayers`.
    pub array_layers: u32,
    /// `VkImageCreateInfo::imageType`.
    pub image_type: vk::ImageType,
    /// `VkImageViewCreateInfo::viewType`.
    pub view_type: vk::ImageViewType,
    /// `VkImageCreateInfo::tiling`.
    pub tiling: vk::ImageTiling,
    /// `VkImageCreateInfo::samples`.
    pub samples: vk::SampleCountFlags,
}

impl ImageDescriptor {
    /// Builds a descriptor with single mip/layer, 2D optimal image, and one sample.
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

    /// Sets `mip_levels` on the descriptor.
    pub fn with_mip_levels(mut self, levels: u32) -> Self {
        self.mip_levels = levels;
        self
    }

    /// Sets `array_layers` and `view_type` on the descriptor.
    pub fn with_array_layers(mut self, layers: u32, view_type: vk::ImageViewType) -> Self {
        self.array_layers = layers;
        self.view_type = view_type;
        self
    }
}

/// Device image with bound memory, view, and tracked layout.
pub struct RotexImage {
    image_handle: vk::Image,
    device_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    current_layout: std::cell::Cell<vk::ImageLayout>,
    aspect_mask: vk::ImageAspectFlags,
}

impl RotexImage {
    /// Creates an image, allocates and binds memory, and creates an image view.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if image creation, memory allocation, binding, or view creation fails.
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

    /// Records a layout transition on `command_buffer` and updates the tracked layout.
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

    /// Returns the `VkImage` handle.
    pub fn handle(&self) -> vk::Image {
        self.image_handle
    }

    /// Returns the `VkImageView` handle.
    pub fn view(&self) -> vk::ImageView {
        self.image_view
    }

    /// Destroys the image view, image, and frees bound device memory.
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

/// Parameters for creating a [`RotexSampler`].
pub struct SamplerDescriptor {
    /// `VkSamplerCreateInfo::magFilter`.
    pub mag_filter: vk::Filter,
    /// `VkSamplerCreateInfo::minFilter`.
    pub min_filter: vk::Filter,
    /// `VkSamplerCreateInfo::anisotropyEnable`.
    pub anisotropy_enable: bool,
    /// `VkSamplerCreateInfo::maxAnisotropy`.
    pub max_anisotropy: f32,
    /// `VkSamplerCreateInfo::addressModeU`.
    pub address_mode_u: vk::SamplerAddressMode,
    /// `VkSamplerCreateInfo::addressModeV`.
    pub address_mode_v: vk::SamplerAddressMode,
    /// `VkSamplerCreateInfo::addressModeW`.
    pub address_mode_w: vk::SamplerAddressMode,
    /// `VkSamplerCreateInfo::borderColor`.
    pub border_color: vk::BorderColor,
    /// `VkSamplerCreateInfo::unnormalizedCoordinates`.
    pub unnormalized_coordinates: bool,
    /// `VkSamplerCreateInfo::compareEnable`.
    pub compare_enable: bool,
    /// `VkSamplerCreateInfo::mipmapMode`.
    pub mipmap_mode: vk::SamplerMipmapMode,
}

impl SamplerDescriptor {
    /// Default sampler: nearest filtering, clamp-to-edge, no anisotropy.
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

    /// Sets `address_mode_u`, `address_mode_v`, and `address_mode_w`.
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

    /// Sets `anisotropy_enable` and `max_anisotropy`.
    pub fn with_anisotropy(mut self, enable: bool, max_anisotropy: f32) -> Self {
        self.anisotropy_enable = enable;
        self.max_anisotropy = max_anisotropy;
        self
    }

    /// Sets `mag_filter` and `min_filter`.
    pub fn with_filters(mut self, mag_filter: vk::Filter, min_filter: vk::Filter) -> Self {
        self.mag_filter = mag_filter;
        self.min_filter = min_filter;
        self
    }

    /// Sets `border_color`.
    pub fn with_border_color(mut self, border_color: vk::BorderColor) -> Self {
        self.border_color = border_color;
        self
    }

    /// Sets `unnormalized_coordinates`.
    pub fn with_unnormalized_coordinates(mut self, unnormalized: bool) -> Self {
        self.unnormalized_coordinates = unnormalized;
        self
    }

    /// Sets `compare_enable`.
    pub fn with_compare_enable(mut self, compare_enable: bool) -> Self {
        self.compare_enable = compare_enable;
        self
    }

    /// Sets `mipmap_mode`.
    pub fn with_mipmap_mode(mut self, mipmap_mode: vk::SamplerMipmapMode) -> Self {
        self.mipmap_mode = mipmap_mode;
        self
    }
}

/// Vulkan sampler wrapper.
pub struct RotexSampler {
    handle: vk::Sampler,
}

impl RotexSampler {
    /// Creates a sampler from `descriptor`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `vkCreateSampler` fails.
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

    /// Returns the `VkSampler` handle.
    pub fn handle(&self) -> vk::Sampler {
        self.handle
    }

    /// Destroys the sampler.
    pub fn destroy(&self, device: &Device) {
        unsafe { device.logical_device().destroy_sampler(self.handle, None) };
    }
}
