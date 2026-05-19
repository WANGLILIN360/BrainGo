//! Data loaders. Concrete implementations land in M4 (BAAIWorm, generic
//! connectome CSV/JSON, NeuroML/SWC).

pub mod baaiworm;
pub mod connectome;
pub mod neuroml;

pub use baaiworm::BAAIWormLoader;
