//! Neuron attributes & runtime state — AoS layout, exactly 64 B each.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use static_assertions::const_assert_eq;

// ── Flag bits for `NeuronAttr.flags` ───────────────────────────────────────
pub const NEURON_ALIVE: u8         = 0b0000_0001;
pub const NEURON_LESIONED: u8      = 0b0000_0010;
pub const NEURON_RESERVED_SLOT: u8 = 0b0000_0100;

/// Static neuron attributes (mmap-friendly, exactly 64 B).
///
/// Layout (v2.4):
/// ```text
///  0  id              u64
///  8  first_comp_id   u64
/// 16  neuron_type     u32
/// 20  region_id       u32
/// 24  n_compartment   u32
/// 28  cm              f32
/// 32  g_leak          f32
/// 36  e_leak          f32
/// 40  x               f32
/// 44  y               f32
/// 48  z               f32
/// 52  flags           u8
/// 53  _pad            [u8; 3]
/// 56  _reserved       u64
/// 64 ─ end
/// ```
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize)]
pub struct NeuronAttr {
    pub id: u64,
    pub first_comp_id: u64,

    pub neuron_type: u32,
    pub region_id: u32,
    pub n_compartment: u32,

    pub cm: f32,
    pub g_leak: f32,
    pub e_leak: f32,

    pub x: f32,
    pub y: f32,
    pub z: f32,

    pub flags: u8,
    pub _pad: [u8; 3],
    pub _reserved: u64,
}
const_assert_eq!(std::mem::size_of::<NeuronAttr>(), 64);
const_assert_eq!(std::mem::align_of::<NeuronAttr>(), 64);

impl Default for NeuronAttr {
    fn default() -> Self {
        Self {
            id: 0,
            first_comp_id: u64::MAX,
            neuron_type: 0,
            region_id: 0,
            n_compartment: 0,
            cm: 100.0,    // pF
            g_leak: 10.0, // nS
            e_leak: -70.0, // mV
            x: 0.0,
            y: 0.0,
            z: 0.0,
            flags: NEURON_ALIVE,
            _pad: [0; 3],
            _reserved: 0,
        }
    }
}

/// Runtime neuron state (exactly 64 B).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize)]
pub struct NeuronState {
    pub last_spike_tick: u64, // u64::MAX = never spiked

    pub v_mem: f32,
    pub u: f32,
    pub i_total: f32,
    pub i_ext: f32,

    pub cai: f32,
    pub stdp_trace: f32,
    pub adapt_w: f32,
    pub v_mem_soma: f32,

    pub i_syn: f32,
    pub i_gap: f32,
    pub spike_count: u32,

    pub _reserved: [u32; 3],
}
const_assert_eq!(std::mem::size_of::<NeuronState>(), 64);
const_assert_eq!(std::mem::align_of::<NeuronState>(), 64);

impl Default for NeuronState {
    fn default() -> Self {
        Self {
            last_spike_tick: u64::MAX,
            v_mem: -70.0,
            u: 0.0,
            i_total: 0.0,
            i_ext: 0.0,
            cai: 0.05, // μM (resting)
            stdp_trace: 0.0,
            adapt_w: 0.0,
            v_mem_soma: -70.0,
            i_syn: 0.0,
            i_gap: 0.0,
            spike_count: 0,
            _reserved: [0; 3],
        }
    }
}
