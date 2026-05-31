//! Error types returned by the Vulkan backend.

use std::fmt::{Display, Formatter};

use ash::vk;

/// Error severity reported with [`Error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Non-fatal informational condition.
    Info,
    /// Recoverable or degraded behavior.
    Warning,
    /// Operation failed but the process may continue.
    Recoverable,
    /// Unrecoverable failure.
    Fatal,
}

/// Specific failure category for [`Error`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    /// Vulkan API returned a non-success [`vk::Result`].
    Vulkan(vk::Result),
    /// Requested operation or configuration is not supported.
    Unsupported(&'static str),
    /// No physical device satisfied the device descriptor.
    NoCompatibleDevice,
}

/// Backend error with [`ErrorKind`] and [`Severity`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// Failure category.
    pub kind: ErrorKind,
    /// How severely the failure should be treated.
    pub severity: Severity,
}

impl Error {
    /// Builds a [`Severity::Fatal`] error for `kind`.
    pub fn fatal(kind: ErrorKind) -> Self {
        Self {
            kind,
            severity: Severity::Fatal,
        }
    }

    /// Returns the raw Vulkan result code when `kind` is [`ErrorKind::Vulkan`].
    pub fn vk_result_code(&self) -> Option<i32> {
        match self.kind {
            ErrorKind::Vulkan(code) => Some(code.as_raw()),
            _ => None,
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            ErrorKind::Vulkan(code) => write!(f, "Vulkan error: {code:?} ({})", code.as_raw()),
            ErrorKind::Unsupported(message) => write!(f, "Unsupported: {message}"),
            ErrorKind::NoCompatibleDevice => write!(f, "No compatible Vulkan device found"),
        }
    }
}

impl std::error::Error for Error {}

/// Maps a Vulkan `result` into a fatal [`Error`].
pub fn vk_error(result: vk::Result) -> Error {
    Error::fatal(ErrorKind::Vulkan(result))
}
