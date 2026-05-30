use std::fmt::{Display, Formatter};

use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Info,
    Warning,
    Recoverable,
    Fatal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    Vulkan(vk::Result),
    Unsupported(&'static str),
    NoCompatibleDevice,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub severity: Severity,
}

impl Error {
    pub fn fatal(kind: ErrorKind) -> Self {
        Self {
            kind,
            severity: Severity::Fatal,
        }
    }

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

pub fn vk_error(result: vk::Result) -> Error {
    Error::fatal(ErrorKind::Vulkan(result))
}
