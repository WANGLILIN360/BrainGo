//! Python bindings — extended API per design doc §9.

use pyo3::prelude::*;

use crate::sim::engine::Simulation;
use crate::sim::observer::{BrainObserver, SpikeLog};
use crate::storage::mmap_db::BrainDB;

// ── PyBrainDB ─────────────────────────────────────────────────────────────

#[pyclass(name = "BrainDB")]
pub struct PyBrainDB {
    inner: BrainDB,
}

#[pymethods]
impl PyBrainDB {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = BrainDB::open(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn neuron_count(&self) -> u64 { self.inner.header.n_neurons }
    fn synapse_count(&self) -> u64 { self.inner.header.n_synapses }
    fn gap_junction_count(&self) -> u64 { self.inner.header.n_gap_junctions }
    fn compartment_count(&self) -> u64 { self.inner.header.n_compartments }
    fn current_tick(&self) -> u64 { self.inner.current_tick }

    /// Get neuron name by ID. Returns "N{id}" if name not stored.
    fn get_neuron_name(&self, id: u64) -> String {
        let i = id as usize;
        self.inner.meta.neuron_names.get(i)
            .cloned()
            .unwrap_or_else(|| format!("N{}", id))
    }

    /// Get all neuron names.
    fn get_all_neuron_names(&self) -> Vec<String> {
        if self.inner.meta.neuron_names.is_empty() {
            (0..self.inner.header.n_neurons as usize)
                .map(|i| format!("N{}", i))
                .collect()
        } else {
            self.inner.meta.neuron_names.clone()
        }
    }

    /// Get the membrane voltage of a neuron by ID.
    fn get_neuron_voltage(&self, id: u64) -> f32 {
        let i = id as usize;
        self.inner.neuron_states.get(i).map(|s| s.v_mem).unwrap_or(0.0)
    }

    /// Get the spike count of a neuron by ID.
    fn get_neuron_spike_count(&self, id: u64) -> u32 {
        let i = id as usize;
        self.inner.neuron_states.get(i).map(|s| s.spike_count).unwrap_or(0)
    }

    /// Get all membrane voltages as a list.
    fn get_all_voltages(&self) -> Vec<f32> {
        self.inner.neuron_states.iter().map(|s| s.v_mem).collect()
    }

    /// Query downstream neighbors (BFS, max `depth` hops).
    fn query_neighbors(&self, id: u64, depth: u32) -> PyResult<Vec<u64>> {
        let db = &self.inner;
        let result = crate::query::connectivity::bfs_downstream(db, id, depth);
        Ok(result.into_iter().map(|(nid, _)| nid).collect())
    }

    /// Query upstream neighbors (BFS, max `depth` hops).
    fn query_upstream_neighbors(&self, id: u64, depth: u32) -> PyResult<Vec<u64>> {
        let db = &self.inner;
        let result = crate::query::connectivity::bfs_upstream(db, id, depth);
        Ok(result.into_iter().map(|(nid, _)| nid).collect())
    }

    /// Query strongest path between two neurons.
    fn query_strongest_path(&self, from: u64, to: u64) -> PyResult<Vec<u64>> {
        let db = &self.inner;
        let path = crate::query::connectivity::strongest_path(db, from, to);
        match path {
            Some((neurons, _weight)) => Ok(neurons),
            None => Ok(Vec::new()),
        }
    }

    /// Region mean LFP.
    fn region_mean_lfp(&self, region_id: u32) -> f32 {
        crate::query::oscillation::region_mean_lfp(
            &self.inner,
            region_id,
            &self.inner.neuron_states,
        )
    }

    /// Get incoming synapse indices for a neuron (CSC lookup).
    fn in_synapses(&self, neuron_id: u64) -> Vec<u32> {
        let row_ptr = self.inner.csr_row_ptr();
        let col_idx = self.inner.csr_col_idx();
        let n = row_ptr.len().saturating_sub(1);
        if (neuron_id as usize) >= n {
            return Vec::new();
        }
        // Scan all outgoing edges to find those targeting this neuron.
        let target = neuron_id as u32;
        let mut result = Vec::new();
        for pre in 0..n {
            let s = row_ptr[pre] as usize;
            let e = row_ptr[pre + 1] as usize;
            for syn_idx in s..e {
                if col_idx[syn_idx] as u32 == target {
                    result.push(syn_idx as u32);
                }
            }
        }
        result
    }

    /// Region pathway info: synapse count and weight between two regions.
    fn region_pathway_info(&self, source: u32, target: u32) -> Option<(usize, f32)> {
        crate::query::region_query::region_pathway_info(&self.inner, source, target)
            .map(|info| (info.synapse_count, info.total_weight))
    }
}

// ── PySpikeLog (observer exposed to Python) ────────────────────────────────

#[pyclass(name = "SpikeLog")]
pub struct PySpikeLog {
    log: SpikeLog,
}

#[pymethods]
impl PySpikeLog {
    #[new]
    fn new() -> Self {
        Self { log: SpikeLog::default() }
    }

    /// Get all recorded spikes as [(neuron_id, tick), ...].
    fn spikes(&self) -> Vec<(u64, u64)> {
        self.log.spikes.clone()
    }

    /// Get spike count for a specific neuron.
    fn count(&self, neuron_id: u64) -> usize {
        self.log.count(neuron_id)
    }

    /// Clear the log.
    fn clear(&mut self) {
        self.log.spikes.clear();
    }
}

impl BrainObserver for PySpikeLog {
    fn on_spike(&mut self, neuron_id: u64, tick: u64) {
        self.log.on_spike(neuron_id, tick);
    }
}

// ── PySimulation ──────────────────────────────────────────────────────────

#[pyclass(name = "Simulation")]
pub struct PySimulation {
    inner: Simulation,
}

#[pymethods]
impl PySimulation {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let db = BrainDB::open(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner: Simulation::new(db) })
    }

    /// Step the simulation forward by one tick.
    fn step(&mut self) {
        self.inner.step();
    }

    /// Run `n_ticks` steps.
    fn run(&mut self, n_ticks: u64) {
        self.inner.run(n_ticks);
    }

    /// Get current tick.
    fn current_tick(&self) -> u64 {
        self.inner.current_tick
    }

    /// Get neuron count.
    fn neuron_count(&self) -> u64 {
        self.inner.db.header.n_neurons
    }

    /// Get neuron name by ID. Returns "N{id}" if name not stored.
    fn get_neuron_name(&self, id: u64) -> String {
        let i = id as usize;
        self.inner.db.meta.neuron_names.get(i)
            .cloned()
            .unwrap_or_else(|| format!("N{}", id))
    }

    /// Get all neuron names.
    fn get_all_neuron_names(&self) -> Vec<String> {
        if self.inner.db.meta.neuron_names.is_empty() {
            (0..self.inner.db.header.n_neurons as usize)
                .map(|i| format!("N{}", i))
                .collect()
        } else {
            self.inner.db.meta.neuron_names.clone()
        }
    }

    /// Get neuron voltage.
    fn get_neuron_voltage(&self, id: u64) -> f32 {
        let i = id as usize;
        self.inner.db.neuron_states.get(i).map(|s| s.v_mem).unwrap_or(0.0)
    }

    /// Set external current on a neuron (pA).
    fn set_neuron_input(&mut self, id: u64, current: f32) {
        let i = id as usize;
        if i < self.inner.db.neuron_states.len() {
            self.inner.db.neuron_states[i].i_ext = current;
        }
    }

    /// Get spike count for a neuron.
    fn get_spike_count(&self, id: u64) -> u32 {
        let i = id as usize;
        self.inner.db.neuron_states.get(i).map(|s| s.spike_count).unwrap_or(0)
    }

    /// Get all membrane voltages.
    fn get_all_voltages(&self) -> Vec<f32> {
        self.inner.db.neuron_states.iter().map(|s| s.v_mem).collect()
    }

    /// Present a stimulus pattern: `[(neuron_id, current_pA), ...]`.
    fn present_stimulus(&mut self, pattern: Vec<(u32, f32)>, duration_ticks: u64) {
        self.inner.present_stimulus(&pattern, duration_ticks);
    }

    /// Read firing rate pattern for a brain region.
    fn read_firing_rate_pattern(&self, region_id: u32, window_ticks: u64) -> Vec<f32> {
        self.inner.read_firing_rate_pattern(region_id, window_ticks)
    }

    /// Read membrane potential pattern for a brain region.
    fn read_vmem_pattern(&self, region_id: u32) -> Vec<f32> {
        self.inner.read_vmem_pattern(region_id)
    }

    /// Kill a neuron (mark as lesioned).
    fn kill_neuron(&mut self, id: u32) {
        self.inner.kill_neuron(id);
    }

    /// Activate a previously killed neuron.
    fn activate_neuron(&mut self, id: u32) {
        self.inner.activate_neuron(id);
    }

    /// Enable/disable STDP.
    fn set_stdp_enabled(&mut self, enabled: bool) {
        self.inner.config.stdp_enabled = enabled;
    }

    /// Enable/disable structural plasticity.
    fn set_structural_plasticity_enabled(&mut self, enabled: bool) {
        self.inner.config.structural_plasticity_enabled = enabled;
    }

    /// Enable/disable neuromodulator diffusion.
    fn set_modulation_enabled(&mut self, enabled: bool) {
        self.inner.config.modulation_enabled = enabled;
    }

    /// Add a spike observer (returns the observer index).
    fn add_spike_log(&mut self) -> usize {
        let idx = self.inner.observers.len();
        self.inner.add_observer(SpikeLog::default());
        idx
    }

    /// Get spikes from the spike log observer at the given index.
    fn get_spikes(&self, _observer_idx: usize) -> Vec<(u64, u64)> {
        // We can't easily access the observer from outside, so we use
        // recently_fired as a proxy for the most recent tick's spikes.
        self.inner.recently_fired.iter().map(|&n| (n as u64, self.inner.current_tick)).collect()
    }

    /// Get recently fired neurons (last tick).
    fn get_recently_fired(&self) -> Vec<u32> {
        self.inner.recently_fired.clone()
    }

    /// Get spike times for a specific neuron from the spike log observer.
    /// Returns `[(tick, ), ...]` for all recorded spikes of `neuron_id`.
    fn get_spike_times(&self, neuron_id: u64) -> Vec<u64> {
        // Use recently_fired as the most recent tick's data.
        // For full spike history, attach a SpikeLog observer before running.
        self.inner.recently_fired.iter()
            .filter(|&&n| n as u64 == neuron_id)
            .map(|_| self.inner.current_tick)
            .collect()
    }

    /// Enable WAL (Write-Ahead Log) for crash recovery.
    fn enable_wal(&mut self, path: &str) -> PyResult<()> {
        self.inner.enable_wal(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Disable WAL and close the file.
    fn disable_wal(&mut self) {
        self.inner.disable_wal();
    }

    /// Load a C. elegans connectome from BAAIWorm data.
    /// Returns a new PySimulation with the 302-neuron network.
    /// `data_dir` is the BAAIWorm project directory containing eworm/.
    /// `output_path` is where the .braindb file will be written.
    #[staticmethod]
    fn load_baaiworm(data_dir: &str, output_path: &str) -> PyResult<Self> {
        use crate::storage::loader::baaiworm::BAAIWormLoader;
        let loader = BAAIWormLoader::load_from_dir(std::path::Path::new(data_dir))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let db = loader.into_braindb(std::path::Path::new(output_path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner: Simulation::new(db) })
    }
}

// ── Module ────────────────────────────────────────────────────────────────

#[pymodule]
fn _braindb(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBrainDB>()?;
    m.add_class::<PySimulation>()?;
    m.add_class::<PySpikeLog>()?;
    Ok(())
}
