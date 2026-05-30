use ash::vk;

pub struct VertexInputDescriptor {
    pub binding_descriptions: Vec<vk::VertexInputBindingDescription>,
    pub attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
    pub flags: vk::PipelineVertexInputStateCreateFlags,
}

pub trait Vertex {
    fn descriptor() -> VertexInputDescriptor;
}

impl VertexInputDescriptor {
    pub fn default() -> Self {
        Self {
            binding_descriptions: Vec::new(),
            attribute_descriptions: Vec::new(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
        }
    }

    pub fn with_binding(mut self, description: vk::VertexInputBindingDescription) -> Self {
        self.binding_descriptions.push(description);
        self
    }

    pub fn with_attribute(mut self, description: vk::VertexInputAttributeDescription) -> Self {
        self.attribute_descriptions.push(description);
        self
    }

    pub fn with_flags(mut self, flags: vk::PipelineVertexInputStateCreateFlags) -> Self {
        self.flags |= flags;
        self
    }
}
