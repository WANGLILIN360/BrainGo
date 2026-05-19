//! Chemical synapse — attributes (mmap-static) + state (dynamic).

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use static_assertions::const_assert_eq;

// ── Synapse mode ──────────────────────────────────────────────────────────
pub const SYN_MODE_EVENT_DRIVEN: u8 = 0;
pub const SYN_MODE_CONTINUOUS:   u8 = 1;

// ── Synapse type ──────────────────────────────────────────────────────────
pub const SYN_EXCITATORY: u8 = 0;
pub const SYN_INHIBITORY: u8 = 1;
pub const SYN_MODULATORY: u8 = 2;

// ── Receptor type (index into ReceptorParams table) ───────────────────────
pub const RECEPTOR_AMPA:  u8 = 0;
pub const RECEPTOR_NMDA:  u8 = 1;
pub const RECEPTOR_GABAA: u8 = 2;
pub const RECEPTOR_GABAB: u8 = 3;
pub const RECEPTOR_MIXED: u8 = 4;

/// Static synapse attributes (exactly 32 B).
///
/// Note: the design document's textual layout (§3.4) ends with `_pad: f32`,
/// which overshoots 32 B. The packing below preserves the same fields and
/// the documented total size (32 B) by using a 3-byte pad after
/// `receptor_type` to align `u_se`.
///
/// Layout:
/// ```text
///  0  post_neuron    u32
///  4  post_comp      u16
///  6  pre_comp       u16
///  8  base_weight    f32
/// 12  delay_ticks    u16
/// 14  syn_type       u8
/// 15  syn_mode       u8
/// 16  receptor_type  u8
/// 17  _pad0          [u8; 3]
/// 20  u_se           f32
/// 24  u_fac          f32
/// 28  tau_rec        f32
/// 32 ─ end
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize)]
pub struct SynapseAttr {
    pub post_neuron: u32,
    pub post_comp: u16,
    pub pre_comp: u16,

    pub base_weight: f32,
    pub delay_ticks: u16,
    pub syn_type: u8,
    pub syn_mode: u8,
    pub receptor_type: u8,
    pub _pad0: [u8; 3],

    pub u_se: f32,
    pub u_fac: f32,
    pub tau_rec: f32,
}
const_assert_eq!(std::mem::size_of::<SynapseAttr>(), 32);

impl Default for SynapseAttr {
    fn default() -> Self {
        Self {
            post_neuron: 0,
            post_comp: 0,
            pre_comp: 0,
            base_weight: 0.0,
            delay_ticks: 1,
            syn_type: SYN_EXCITATORY,
            syn_mode: SYN_MODE_EVENT_DRIVEN,
            receptor_type: RECEPTOR_AMPA,
            _pad0: [0; 3],
            u_se: 0.5,
            u_fac: 0.0,
            tau_rec: 100.0,
        }
    }
}

/// Runtime synapse state (exactly 32 B).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize)]
pub struct SynapseState {
    /// Event-driven: rise-phase conductance (nS).
    /// Continuous mode: re-purposed as `s` (Sigmoid state variable).
    pub g_rise: f32,
    /// Event-driven: decay-phase conductance (nS). Unused in continuous mode.
    pub g_decay: f32,

    pub r: f32,
    pub u: f32,

    pub stdp_trace: f32,
    pub dw_accum: f32,

    pub weight: f32,

    pub is_active: u8,
    pub _pad: [u8; 3],
}
const_assert_eq!(std::mem::size_of::<SynapseState>(), 32);

impl Default for SynapseState {
    fn default() -> Self {
        Self {
            g_rise: 0.0,
            g_decay: 0.0,
            r: 1.0,
            u: 0.0,
            stdp_trace: 0.0,
            dw_accum: 0.0,
            weight: 0.0,
            is_active: 0,
            _pad: [0; 3],
        }
    }
}
