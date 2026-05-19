//! Receptor parameters (per-receptor-type, global table).

use bytemuck::{Pod, Zeroable};
use static_assertions::const_assert_eq;

/// Receptor parameters — 32 B Pod record, kept in a small global table.
///
/// Layout:
/// ```text
///  0  tau_rise       f32
///  4  tau_decay      f32
///  8  e_rev          f32
/// 12  mg_conc        f32    (NMDA only; 0 for non-NMDA)
/// 16  v_threshold    f32    (continuous-mode Sigmoid threshold, mV)
/// 20  v_slope        f32    (continuous-mode Sigmoid slope, mV)
/// 24  k_rate         f32    (continuous-mode activation rate, 1/ms)
/// 28  _pad           u32
/// 32 ─ end
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ReceptorParams {
    pub tau_rise: f32,
    pub tau_decay: f32,
    pub e_rev: f32,
    pub mg_conc: f32,

    pub v_threshold: f32,
    pub v_slope: f32,
    pub k_rate: f32,
    pub _pad: u32,
}
const_assert_eq!(std::mem::size_of::<ReceptorParams>(), 32);

impl Default for ReceptorParams {
    fn default() -> Self {
        Self::ampa()
    }
}

impl ReceptorParams {
    pub fn ampa() -> Self {
        Self { tau_rise: 0.5,  tau_decay:  3.0,  e_rev:   0.0, mg_conc: 0.0,
               v_threshold: -20.0, v_slope: 5.0, k_rate: 0.1, _pad: 0 }
    }
    pub fn nmda() -> Self {
        Self { tau_rise: 2.0,  tau_decay: 100.0, e_rev:   0.0, mg_conc: 1.0,
               v_threshold: -20.0, v_slope: 5.0, k_rate: 0.01, _pad: 0 }
    }
    pub fn gaba_a() -> Self {
        Self { tau_rise: 0.5,  tau_decay:   7.0, e_rev: -70.0, mg_conc: 0.0,
               v_threshold: -20.0, v_slope: 5.0, k_rate: 0.1, _pad: 0 }
    }
    pub fn gaba_b() -> Self {
        Self { tau_rise: 10.0, tau_decay:  50.0, e_rev: -70.0, mg_conc: 0.0,
               v_threshold: -20.0, v_slope: 5.0, k_rate: 0.01, _pad: 0 }
    }
}
