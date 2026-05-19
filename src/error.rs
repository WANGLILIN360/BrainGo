//! Unified error type for BrainDB.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, BrainDBError>;

#[derive(Error, Debug)]
pub enum BrainDBError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("postcard (de)serialization error: {0}")]
    Postcard(#[from] postcard::Error),

    #[error("invalid file: {0}")]
    InvalidFile(String),

    #[error("invalid magic: expected {expected:?}, got {actual:?}")]
    InvalidMagic { expected: [u8; 4], actual: [u8; 4] },

    #[error("unsupported file version: {0}")]
    UnsupportedVersion(u16),

    #[error("validation error: {0}")]
    Validation(String),

    #[error("snapshot mismatch: {0}")]
    SnapshotMismatch(String),

    #[error("not implemented: {0}")]
    NotImplemented(&'static str),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("spreadsheet error: {0}")]
    Spreadsheet(String),
}

/// Convenience macro mirroring anyhow::ensure! that returns `BrainDBError::Validation`.
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            return Err($crate::error::BrainDBError::Validation(format!($($arg)*)));
        }
    };
}
