//! Gap junction (electrical synapse) — continuous voltage coupling.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use static_assertions::const_assert_eq;

/// Gap junction record (exactly 24 B).
///
/// Layout:
/// ```text
///  0  pre_neuron   u32
///  4  post_neuron  u32
///  8  pre_comp     u16
/// 10  post_comp    u16
/// 12  weight       f32   (coupling conductance, nS)
/// 16  _pad         u32
/// 20  _reserved    u32
/// 24 ─ end
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize)]
pub struct GapJunction {
    pub pre_neuron: u32,
    pub post_neuron: u32,
    pub pre_comp: u16,
    pub post_comp: u16,
    pub weight: f32,
    pub _pad: u32,
    pub _reserved: u32,
}
const_assert_eq!(std::mem::size_of::<GapJunction>(), 24);

impl Default for GapJunction {
    fn default() -> Self {
        Self {
            pre_neuron: 0,
            post_neuron: 0,
            pre_comp: 0,
            post_comp: 0,
            weight: 0.0,
            _pad: 0,
            _reserved: 0,
        }
    }
}
