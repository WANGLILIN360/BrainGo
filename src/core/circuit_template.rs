//! Microcircuit / cortical-column templates. Non-POD; stored in metadata.

use serde::{Deserialize, Serialize};

use crate::core::neuron::NeuronAttr;
use crate::core::neuron_type::CompartmentSpec;
use crate::core::synapse::SynapseAttr;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitTemplate {
    pub name: String,
    pub neuron_types: Vec<TemplateNeuronType>,
    pub internal_connections: Vec<TemplateConnection>,
    pub input_ports: Vec<TemplatePort>,
    pub output_ports: Vec<TemplatePort>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemplateNeuronType {
    pub type_name: String,
    pub count: u32,
    pub neuron_attr: NeuronAttr,
    pub compartment_spec: CompartmentSpec,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemplateConnection {
    pub from_type: String,
    pub to_type: String,
    pub connection_rule: ConnectionRule,
    pub synapse_attr: SynapseAttr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConnectionRule {
    AllToAll { probability: f32 },
    OneToOne,
    /// Distance-dependent Gaussian connection probability.
    Gaussian { sigma: f32 },
    /// Each target receives a fixed number of random inputs.
    Convergent { n_inputs: u32 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemplatePort {
    pub name: String,
    pub type_name: String,
}
