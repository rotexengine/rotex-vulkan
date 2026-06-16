use super::buffer::RotexBuffer;
use super::device::Device;

pub enum DeferredResource {
    Buffer(RotexBuffer),
}

pub struct DeferredDeleteQueue {
    pending: Vec<(usize, DeferredResource)>,
}

impl DeferredDeleteQueue {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    pub fn defer_after_frame(&mut self, frame_index: usize, resource: DeferredResource) {
        self.pending.push((frame_index, resource));
    }

    pub fn process_frame(&mut self, device: &Device, completed_frame: usize) {
        let mut index = 0;
        while index < self.pending.len() {
            if self.pending[index].0 == completed_frame {
                let (_, resource) = self.pending.swap_remove(index);
                match resource {
                    DeferredResource::Buffer(buffer) => buffer.destroy(device),
                }
            } else {
                index += 1;
            }
        }
    }

    pub fn destroy_all(&mut self, device: &Device) {
        for (_, resource) in self.pending.drain(..) {
            match resource {
                DeferredResource::Buffer(buffer) => buffer.destroy(device),
            }
        }
    }
}
