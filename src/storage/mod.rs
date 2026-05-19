//! Persistence layer — `.braindb` v2 binary format, mmap engine, snapshots.

pub mod format;
pub mod builder;
pub mod mmap_db;
pub mod wal;
pub mod dynamic_csr;
pub mod shard;
pub mod compress;
pub mod distributed;
pub mod loader;
