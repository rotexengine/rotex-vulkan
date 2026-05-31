//! Vulkan backend for `rotex_types`.
//!
//! Use [`VulkanBridge`] for the high-level engine integration, or [`vulkan`] for
//! lower-level Vulkan wrappers.

/// Vulkan backend modules ([`vulkan`]).
pub mod backend;
/// [`VulkanBridge`] and rotex integration.
pub mod bridge;
/// Vulkan instance creation and debug utilities.
pub mod core;
/// Backend error types and Vulkan result mapping.
pub mod error;

/// Low-level Vulkan wrappers (buffers, pipelines, swapchain, etc.).
pub use backend::vulkan;
pub use bridge::VulkanBridge;
pub use error::{Error, ErrorKind, Severity};
/// Shared rotex frontend types (re-exported).
pub use rotex_types;
