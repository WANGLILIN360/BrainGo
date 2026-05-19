//! Neuron type parameter table — shared across neurons of the same type.
//!
//! Not POD (contains `String`, `HashMap`, `Vec`); stored in the postcard
//! metadata section of `.braindb`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuronTypeParams {
    pub type_id: u32,
    pub type_name: String,
    pub category: NeuronCategory,
    pub model: NeuronModel,

    /// Izhikevich parameters (used when `model == Izhikevich`).
    pub iz_params: IzhikevichParams,

    /// HH parameter overrides keyed by channel name (used when `model == HH`
    /// or `MultiCompartmentHH`).
    pub hh_overrides: HashMap<String, f32>,

    pub compartment_spec: CompartmentSpec,

    /// Names of ion channels referenced from this neuron type (looked up in
    /// the global `IonChannelDef` list).
    pub channels: Vec<String>,
}

impl Default for NeuronTypeParams {
    fn default() -> Self {
        Self {
            type_id: 0,
            type_name: String::new(),
            category: NeuronCategory::Interneuron,
            model: NeuronModel::Izhikevich,
            iz_params: IzhikevichParams::regular_spiking(),
            hh_overrides: HashMap::new(),
            compartment_spec: CompartmentSpec::Single,
            channels: Vec::new(),
        }
    }
}

impl NeuronTypeParams {
    /// Default initial membrane voltage for this neuron type.
    pub fn default_v_init(&self) -> f32 {
        match self.model {
            NeuronModel::Izhikevich => self.iz_params.c,
            NeuronModel::LIF
            | NeuronModel::HodgkinHuxley
            | NeuronModel::MultiCompartmentHH
            | NeuronModel::Graded => self.iz_params.v_rest,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronCategory {
    Sensory,
    Interneuron,
    Motor,
    Modulatory,
    Glia,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronModel {
    LIF,
    Izhikevich,
    HodgkinHuxley,
    MultiCompartmentHH,
    /// v2.4: graded (non-spiking) neurons, used by most *C. elegans* cells.
    Graded,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct IzhikevichParams {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub cm: f32,
    pub v_rest: f32,
}

impl IzhikevichParams {
    pub fn regular_spiking() -> Self {
        // Classical Izhikevich "RS" cortical pyramidal cell.
        Self { a: 0.02, b: 0.2, c: -65.0, d: 8.0, cm: 1.0, v_rest: -70.0 }
    }
    pub fn fast_spiking() -> Self {
        Self { a: 0.1, b: 0.2, c: -65.0, d: 2.0, cm: 1.0, v_rest: -70.0 }
    }
    pub fn intrinsically_bursting() -> Self {
        Self { a: 0.02, b: 0.2, c: -55.0, d: 4.0, cm: 1.0, v_rest: -70.0 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CompartmentSpec {
    /// Single compartment (point neuron).
    Single,
    TwoCompartment {
        dend_length: f32,
        dend_diam: f32,
        g_coupling: f32, // nS
    },
    /// Load morphology from a SWC/HOC file by path.
    MorphologyFile(String),
    /// Reference a registered template by name.
    Template(String),
}
