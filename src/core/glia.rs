//! Glia parameters (region-level, simplified).

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, Serialize, Deserialize)]
pub struct GliaParams {
    // Astrocytes
    pub clearance_rate_glut: f32, // 1/ms
    pub clearance_rate_gaba: f32,
    pub ca_wave_threshold: f32,   // μM
    pub ca_wave_speed: f32,       // μm/ms

    // Oligodendrocytes
    pub myelin_gain: f32,

    // Microglia
    pub prune_threshold: f32,
    pub prune_rate: f32,
    pub _pad: f32,
}
