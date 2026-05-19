//! BrainDB — full-scale brain network simulation database.
//!
//! This is the **M1+M2 scaffold** of the design described in
//! `braindb-design.md` v2.4. It implements:
//!
//! - Core POD data structures with compile-time size assertions
//!   (`NeuronAttr`, `NeuronState`, `CompartmentAttr`, `CompartmentState`,
//!   `SynapseAttr`, `SynapseState`, `GapJunction`, `BrainRegion`,
//!   `LongRangePathway`, `ReceptorParams`, …).
//! - Auxiliary non-POD descriptors (`IonChannelDef`, `IonChannelSet`,
//!   `NeuronTypeParams`, `CircuitTemplate`) serialized via postcard.
//! - The `.braindb` v2 binary file format with a 512-byte header,
//!   mmap-friendly static segments, and a postcard-serialized metadata
//!   section.
//! - `BrainDBBuilder` collecting entities in arbitrary order, sorting
//!   synapses by `pre_neuron` and emitting a CSR + meta-segment.
//! - `BrainDB::open` for zero-copy mmap loading and `save_snapshot` /
//!   `load_snapshot` for the dynamic-state `.snapshot` file.
//!
//! Later milestones (M3–M5) will plug in the simulation engine, dynamic
//! CSR, loaders, plasticity and Python bindings on top of this scaffold.

#![allow(clippy::needless_doctest_main)]

pub mod error;
pub mod core;
pub mod storage;
pub mod sim;
pub mod query;

#[cfg(feature = "python")]
pub mod pyo3_bindings;

pub use crate::error::{BrainDBError, Result};

pub use crate::core::neuron::{NeuronAttr, NeuronState, NEURON_ALIVE, NEURON_LESIONED, NEURON_RESERVED_SLOT};
pub use crate::core::compartment::{CompartmentAttr, CompartmentState, CompType};
pub use crate::core::synapse::{
    SynapseAttr, SynapseState, SYN_MODE_EVENT_DRIVEN, SYN_MODE_CONTINUOUS,
    SYN_EXCITATORY, SYN_INHIBITORY, SYN_MODULATORY,
    RECEPTOR_AMPA, RECEPTOR_NMDA, RECEPTOR_GABAA, RECEPTOR_GABAB, RECEPTOR_MIXED,
};
pub use crate::core::gap_junction::GapJunction;
pub use crate::core::ion_channel::{
    IonChannelDef, IonChannelSet, IonType, GateFn, GateVarDef, ChannelConductance,
    GATE_FN_REGISTRY,
};
pub use crate::core::receptor::ReceptorParams;
pub use crate::core::neuromodulator::{
    ModulationLevel, ModulationRule, DopamineEffect, SerotoninEffect, AchEffect, NeEffect,
};
pub use crate::core::glia::GliaParams;
pub use crate::core::region::{BrainRegion, LongRangePathway};
pub use crate::core::neuron_type::{
    NeuronTypeParams, NeuronModel, NeuronCategory, IzhikevichParams, CompartmentSpec,
};
pub use crate::core::circuit_template::{
    CircuitTemplate, TemplateNeuronType, TemplateConnection, TemplatePort, ConnectionRule,
};

pub use crate::storage::format::{Header, FILE_MAGIC, FILE_VERSION};
pub use crate::storage::builder::BrainDBBuilder;
pub use crate::storage::mmap_db::BrainDB;

pub use crate::sim::{
    BrainObserver, EventRing, PlasticityEvent, Simulation, SimulationConfig, SpikeLog,
    SynapticEvent,
};

pub use crate::query::connectivity::{bfs_downstream, bfs_upstream, strongest_path, Hit};
pub use crate::query::oscillation::region_mean_lfp;
pub use crate::query::region_query::{region_pathway_info, outgoing_pathways, incoming_pathways, pathways_between, region_connectivity_matrix, RegionPathwayInfo};
