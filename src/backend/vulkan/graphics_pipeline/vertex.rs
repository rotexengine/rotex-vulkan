use ash::vk;

/// Vertex input bindings and attributes for `VkPipelineVertexInputStateCreateInfo`.
pub struct VertexInputDescriptor {
    /// `pVertexBindingDescriptions`.
    pub binding_descriptions: Vec<vk::VertexInputBindingDescription>,
    /// `pVertexAttributeDescriptions`.
    pub attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
    /// `flags` on `VkPipelineVertexInputStateCreateInfo`.
    pub flags: vk::PipelineVertexInputStateCreateFlags,
}

/// Types that supply a fixed vertex input layout for pipeline creation.
pub trait Vertex {
    /// Returns the vertex input descriptor for this type.
    fn descriptor() -> VertexInputDescriptor;
}

impl VertexInputDescriptor {
    /// Empty vertex input with no bindings or attributes.
    pub fn default() -> Self {
        Self {
            binding_descriptions: Vec::new(),
            attribute_descriptions: Vec::new(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
        }
    }

    /// Appends a `VkVertexInputBindingDescription`.
    pub fn with_binding(mut self, description: vk::VertexInputBindingDescription) -> Self {
        self.binding_descriptions.push(description);
        self
    }

    /// Appends a `VkVertexInputAttributeDescription`.
    pub fn with_attribute(mut self, description: vk::VertexInputAttributeDescription) -> Self {
        self.attribute_descriptions.push(description);
        self
    }

    /// ORs `flags` into the vertex input create flags.
    pub fn with_flags(mut self, flags: vk::PipelineVertexInputStateCreateFlags) -> Self {
        self.flags |= flags;
        self
    }
}
