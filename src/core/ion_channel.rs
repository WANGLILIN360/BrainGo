//! Ion channel mechanisms (pluggable, registered at runtime).
//!
//! These types contain `String`/`Vec` and are therefore **not POD**; they are
//! stored in the metadata section of `.braindb` via `postcard` serialization
//! rather than mmap'd directly.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Carrier ion of a channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IonType {
    Sodium,
    Potassium,
    Calcium,
    Chloride,
    NonSpecific,
}

/// Gate kinetics — voltage-dependent steady-state / time-constant function.
///
/// `CustomIndex(u32)` references a closure in [`GATE_FN_REGISTRY`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GateFn {
    /// `x_inf = 1 / (1 + exp((V - v_half) / slope))`.
    Boltzmann { v_half: f32, slope: f32 },
    BoltzmannTau {
        v_half: f32,
        slope: f32,
        tau_max: f32,
        tau_min: f32,
    },
    ExpGaussian { a: f32, b: f32, c: f32, d: f32, e: f32 },
    /// Index into the runtime registry; lookup yields `fn(V) -> x_inf`.
    CustomIndex(u32),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateVarDef {
    pub name: String,
    pub gate_fn: GateFn,
    pub initial: f32,
}

/// Definition of a single ion channel mechanism.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IonChannelDef {
    pub name: String,
    pub ion: IonType,
    /// Default reversal potential (mV); per-compartment override is possible
    /// via `ChannelConductance::e_rev_override`.
    pub e_rev: f32,
    pub gate_vars: Vec<GateVarDef>,
    pub ca_dependent: bool,
    /// If `ca_dependent`, the channel ID providing the Ca²⁺ source current.
    pub ca_source_channel: Option<u32>,
}

/// Conductance density of a single channel within a compartment.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ChannelConductance {
    pub channel_id: u32,
    pub g_max: f32,                    // S/cm²
    pub e_rev_override: Option<f32>,   // mV
}

/// Bundle of ion channels (referenced by `CompartmentAttr.ion_channel_set`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IonChannelSet {
    pub name: String,
    pub channels: Vec<ChannelConductance>,
}

impl IonChannelSet {
    pub fn empty(name: impl Into<String>) -> Self {
        Self { name: name.into(), channels: Vec::new() }
    }
}

/// Runtime registry for `GateFn::CustomIndex` — filled at startup.
///
/// The default registry is empty; downstream code may push custom kinetics
/// before simulation begins.
pub static GATE_FN_REGISTRY: LazyLock<HashMap<u32, fn(f32) -> f32>> =
    LazyLock::new(HashMap::new);
