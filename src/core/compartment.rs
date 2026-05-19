//! Compartment attributes & state (multi-compartment HH).

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use static_assertions::const_assert_eq;

/// Compartment type code (stored in `CompartmentAttr.comp_type`).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompType {
    Soma = 0,
    ApicalDend = 1,
    BasalDend = 2,
    Axon = 3,
    Other = 255,
}

/// Static compartment attributes (exactly 128 B).
///
/// Layout (v2.4):
/// ```text
///  0  id                u64
///  8  neuron_id         u64
/// 16  parent_comp_id    u64        (u64::MAX = root/soma)
/// 24  comp_type         u8
/// 25  _pad0             [u8; 3]
/// 28  ion_channel_set   u32
/// 32  length            f32
/// 36  diameter          f32
/// 40  cm                f32
/// 44  r_axial           f32
/// 48  x                 f32
/// 52  y                 f32
/// 56  z                 f32
/// 60  g_leak            f32
/// 64  e_leak            f32
/// 68  _pad1             u32        (alignment for [u64;7])
/// 72  _reserved         [u64; 7]
/// 128 ─ end
/// ```
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize)]
pub struct CompartmentAttr {
    pub id: u64,
    pub neuron_id: u64,
    pub parent_comp_id: u64,

    pub comp_type: u8,
    pub _pad0: [u8; 3],
    pub ion_channel_set: u32,

    pub length: f32,
    pub diameter: f32,
    pub cm: f32,
    pub r_axial: f32,

    pub x: f32,
    pub y: f32,
    pub z: f32,

    pub g_leak: f32,
    pub e_leak: f32,
    pub _pad1: u32,

    pub _reserved: [u64; 7],
}
const_assert_eq!(std::mem::size_of::<CompartmentAttr>(), 128);
const_assert_eq!(std::mem::align_of::<CompartmentAttr>(), 64);

impl Default for CompartmentAttr {
    fn default() -> Self {
        Self {
            id: 0,
            neuron_id: 0,
            parent_comp_id: u64::MAX,
            comp_type: CompType::Soma as u8,
            _pad0: [0; 3],
            ion_channel_set: u32::MAX,
            length: 10.0,
            diameter: 1.0,
            cm: 1.0,
            r_axial: 100.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            g_leak: 0.0003,
            e_leak: -70.0,
            _pad1: 0,
            _reserved: [0; 7],
        }
    }
}

/// Runtime compartment state (exactly 64 B).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize)]
pub struct CompartmentState {
    pub v_mem: f32,
    pub i_total: f32,
    pub i_ext: f32,
    pub cai: f32,

    pub m_na: f32,
    pub h_na: f32,
    pub m_k: f32,
    pub m_ca: f32,

    pub h_ca: f32,
    pub m_kca: f32,
    pub _pad: u32,
    pub _reserved1: u32,

    pub _reserved2: u64,
    pub _reserved3: u64,
}
const_assert_eq!(std::mem::size_of::<CompartmentState>(), 64);
const_assert_eq!(std::mem::align_of::<CompartmentState>(), 64);

impl Default for CompartmentState {
    fn default() -> Self {
        Self {
            v_mem: -70.0,
            i_total: 0.0,
            i_ext: 0.0,
            cai: 0.05,
            m_na: 0.0,
            h_na: 1.0,
            m_k: 0.0,
            m_ca: 0.0,
            h_ca: 1.0,
            m_kca: 0.0,
            _pad: 0,
            _reserved1: 0,
            _reserved2: 0,
            _reserved3: 0,
        }
    }
}
