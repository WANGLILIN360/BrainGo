//! `BrainObserver` — decouple simulation core from visualisation / logging.

use crate::core::neuron::NeuronState;

#[derive(Clone, Copy, Debug)]
pub enum PlasticityEvent {
    Sprout { pre: u64, post: u64, new_weight: f32 },
    Prune { syn_id: u64, old_weight: f32 },
    Stdp { syn_id: u64, dw: f32 },
}

/// Hooks invoked by the simulation loop. All methods have a default no-op
/// implementation so implementers only need to override the ones they care
/// about.
pub trait BrainObserver: Send + Sync {
    fn on_spike(&mut self, _neuron_id: u64, _tick: u64) {}
    fn on_step_done(&mut self, _tick: u64, _states: &[NeuronState]) {}
    fn on_synapse_change(&mut self, _syn_id: u64, _old_weight: f32, _new_weight: f32) {}
    fn on_plasticity_event(&mut self, _event: PlasticityEvent) {}
}

/// In-memory spike log — convenient for tests and small experiments.
#[derive(Default)]
pub struct SpikeLog {
    pub spikes: Vec<(u64, u64)>, // (neuron_id, tick)
}

impl BrainObserver for SpikeLog {
    fn on_spike(&mut self, neuron_id: u64, tick: u64) {
        self.spikes.push((neuron_id, tick));
    }
}

impl SpikeLog {
    pub fn count(&self, neuron_id: u64) -> usize {
        self.spikes.iter().filter(|(n, _)| *n == neuron_id).count()
    }
}
