//! Simulation engine (M3 — point-neuron path).
//!
//! See `braindb-design.md` §5: hybrid continuous + event-driven loop with
//! `accumulate_current` as the unified current-injection entry point.
//!
//! - Multi-compartment cable solver: deferred to M3.5.
//! - STDP trace decay: implemented; weight write-back: deferred to M5.
//! - Structural plasticity (Phase 7): deferred to M5.

pub mod event_ring;
pub mod observer;
pub mod config;
pub mod engine;
pub mod integrator;

pub use config::SimulationConfig;
pub use engine::{BodyEnvironment, Simulation};
pub use event_ring::{EventRing, SynapticEvent};
pub use integrator::Integrator;
pub use observer::{BrainObserver, PlasticityEvent, SpikeLog};
