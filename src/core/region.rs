//! Brain regions & long-range pathways.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::core::glia::GliaParams;
use crate::core::neuromodulator::ModulationLevel;

/// A named brain region (mmap-friendly Pod, 112 B with explicit padding).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, Serialize, Deserialize)]
pub struct BrainRegion {
    pub id: u32,
    pub _pad_id: u32,         // explicit pad → u64 alignment for name_hash

    pub name_hash: u64,
    pub first_neuron: u64,

    pub neuron_count: u32,
    pub _pad0: u32,

    pub cx: f32,
    pub cy: f32,
    pub cz: f32,
    pub _pad_xyz: u32,

    pub modulation: ModulationLevel, // 16 B
    pub glia: GliaParams,            // 32 B

    pub sensory_start: u32,
    pub sensory_end: u32,
    pub motor_start: u32,
    pub motor_end: u32,

    pub _pad_tail: [u32; 2], // trailing pad to keep struct size a multiple of 8
}
const _: () = assert!(std::mem::size_of::<BrainRegion>() % 8 == 0);

/// White-matter pathway between regions (24 B).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, Serialize, Deserialize)]
pub struct LongRangePathway {
    pub source_region: u32,
    pub target_region: u32,

    pub pathway_type: u8, // 0=excitatory, 1=inhibitory, 2=mixed
    pub _pad: [u8; 3],

    pub fiber_count: u32,
    pub conduction_speed: f32, // m/s
    pub mean_delay_ms: f32,
}
