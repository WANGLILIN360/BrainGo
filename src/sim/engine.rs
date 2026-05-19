//! Main simulation loop — hybrid continuous + event-driven dynamics.
//!
//! This implementation covers the **point-neuron path** of the design
//! document's 8-phase loop (see `braindb-design.md` §5):
//!
//! 1. Gap junctions (continuous voltage coupling)
//! 2. Drain delayed synaptic events into `g_rise` / `g_decay`
//! 3. Decay active event-driven chemical synapses & inject currents
//! 3b. Continuous-conductance synapses (BAAIWorm-style Sigmoid)
//! 3c. Clear expired stimuli
//! 4. Update point neurons (Izhikevich / LIF / Graded)
//! 5. STDP trace decay (full weight-update plasticity lands in M5)
//! 8. Observer notifications
//!
//! Multi-compartment HH cable solver (M3.5) is implemented for neurons with
//! `n_compartment > 1`: forward-Euler axial coupling + classical Na/K HH
//! kinetics. Structural plasticity (Phase 7) is still deferred to M5.
//! `Simulation.skipped_multicomp` now only counts malformed neurons whose
//! compartment range overruns the static segment.

use std::collections::VecDeque;

use bytemuck::Pod;
use memmap2::Mmap;

use crate::core::compartment::{CompartmentAttr, CompartmentState};
use crate::core::gap_junction::GapJunction;
use crate::core::ion_channel::{IonChannelDef, IonChannelSet, IonType};
use crate::core::neuron::{NeuronAttr, NeuronState};
use crate::core::neuromodulator::ModulationLevel;
use crate::core::neuron_type::{NeuronModel, NeuronTypeParams};
use crate::core::receptor::ReceptorParams;
use crate::core::synapse::{
    SynapseAttr, RECEPTOR_NMDA, SYN_MODE_CONTINUOUS, SYN_MODE_EVENT_DRIVEN,
    SYN_EXCITATORY,
};
use crate::storage::dynamic_csr::DynamicCSR;
use crate::sim::config::SimulationConfig;
use crate::sim::event_ring::{EventRing, SynapticEvent};
use crate::sim::observer::BrainObserver;
use crate::storage::format::{off, Header};
use crate::storage::mmap_db::BrainDB;

/// Closed-loop interface — connects brain simulation with external body/environment
/// (design doc §15.5). Used for C. elegans chemotaxis, locomotion, etc.
pub trait BodyEnvironment: Send + Sync {
    /// Each tick: return sensory input as `[(neuron_id, current_pA)]`.
    fn get_sensory_input(&mut self, tick: u64) -> Vec<(u32, f32)>;

    /// Each tick: receive motor neuron output as `[(neuron_id, v_mem)]`.
    fn set_motor_output(&mut self, tick: u64, motor_voltages: &[(u32, f32)]);

    /// Optional: advance the body/environment simulation by dt.
    fn step_body(&mut self, dt: f32);
}

/// Wraps a loaded [`BrainDB`] together with all transient simulation state.
pub struct Simulation {
    pub db: BrainDB,
    pub config: SimulationConfig,
    pub dt: f32,
    pub ring_size: usize,

    pub current_tick: u64,

    /// Pending events keyed by `(arrival_tick % ring_size)`.
    pub event_ring: EventRing,
    /// Indices into `syn_attrs` whose conductance is non-negligible right now.
    pub active_synapses: VecDeque<usize>,
    /// Pre-computed list of synapse indices using continuous-conductance mode.
    pub continuous_synapse_indices: Vec<usize>,
    /// Pre-synaptic neuron id, indexed by global synapse index (universal,
    /// also used for STDP and reverse-CSR construction).
    pub syn_pre_neuron: Vec<u32>,

    /// Reverse CSR — for each post-neuron, range into [`Self::rev_csr_syn_idx`]
    /// listing the global indices of its incoming synapses. Built once at
    /// `Simulation::new` from the forward CSR.
    pub rev_csr_row_ptr: Vec<u64>,
    /// Global synapse indices grouped by post-neuron.
    pub rev_csr_syn_idx: Vec<u32>,

    /// Neurons that spiked during the most recent `step()`.
    pub recently_fired: Vec<u32>,

    /// Neurons currently driven by an external stimulus.
    pub active_stimulus_neurons: Vec<u32>,
    /// Absolute tick at which the current stimulus expires.
    pub stimulus_end_tick: u64,

    pub observers: Vec<Box<dyn BrainObserver>>,

    /// Diagnostic counter for multi-compartment neurons whose declared
    /// compartment range is malformed and was skipped.
    pub skipped_multicomp: u64,

    /// Mutable copy of `NeuronAttr.flags` for every neuron — provides the
    /// alive / lesioned bits used by the simulation (the underlying mmap is
    /// read-only). Manipulated via [`Self::kill_neuron`] /
    /// [`Self::activate_neuron`]. This is the foundation of the eventual
    /// DynamicCSR (M5).
    pub neuron_flags: Vec<u8>,

    /// Mutable gap-junction weights (copy of mmap's `GapJunction.weight`
    /// fields). The mmap segment is read-only, so we maintain a separate
    /// vector. `kill_neuron` zeroes the weights of all gap junctions
    /// involving the dead neuron (design doc §16.2).
    pub gap_junction_weights: Vec<f32>,

    /// Mutable copy of `BrainRegion.modulation` for every region.
    /// Updated by Phase 6 (neuromodulator diffusion) each tick.
    pub region_modulation: Vec<ModulationLevel>,

    /// Dynamic CSR for runtime synapse insertion/deletion (structural plasticity).
    /// `None` until structural_plasticity_enabled is turned on.
    pub dynamic_csr: Option<DynamicCSR>,

    /// Optional WAL writer for crash recovery. When set, every state mutation
    /// (neuron/synapse/compartment/gap-junction/flags/modulation) is logged
    /// before being applied, enabling replay after a crash.
    pub wal: Option<crate::storage::wal::WalWriter>,

    /// How often (in ticks) to write a WAL checkpoint + flush.
    /// A checkpoint marks a known-good point; after recovery, the WAL is
    /// replayed from the last checkpoint forward.
    pub wal_checkpoint_interval: u64,
}

impl Simulation {
    pub fn new(db: BrainDB) -> Self {
        let dt = db.header.dt;
        let ring_size = db.header.ring_size as usize;
        let mut sim = Self {
            db,
            config: SimulationConfig::default(),
            dt,
            ring_size,
            current_tick: 0,
            event_ring: EventRing::new(ring_size.max(1)),
            active_synapses: VecDeque::new(),
            continuous_synapse_indices: Vec::new(),
            syn_pre_neuron: Vec::new(),
            rev_csr_row_ptr: Vec::new(),
            rev_csr_syn_idx: Vec::new(),
            recently_fired: Vec::new(),
            active_stimulus_neurons: Vec::new(),
            stimulus_end_tick: 0,
            observers: Vec::new(),
            skipped_multicomp: 0,
            neuron_flags: Vec::new(),
            gap_junction_weights: Vec::new(),
            region_modulation: Vec::new(),
            dynamic_csr: None,
            wal: None,
            wal_checkpoint_interval: 10000, // every 1 s at dt=0.1 ms
        };
        sim.precompute_aux_tables();
        // Initialise the runtime flags overlay from the static NeuronAttr.
        let attrs = sim.db.neuron_attrs();
        sim.neuron_flags = attrs.iter().map(|a| a.flags).collect();
        // Copy gap-junction weights from mmap into mutable vector.
        sim.gap_junction_weights = sim.db.gap_junctions().iter().map(|gj| gj.weight).collect();
        // Copy region modulation levels from mmap into mutable vector.
        sim.region_modulation = sim.db.regions().iter().map(|r| r.modulation).collect();
        sim
    }

    /// Mark `nid` as lesioned: clears `NEURON_ALIVE` and sets `NEURON_LESIONED`.
    /// Subsequent simulation steps will skip the neuron's update and ignore
    /// its gap-junction contributions. Existing synapses targeting this
    /// neuron continue to fire, but its outgoing synapses can no longer
    /// trigger because the neuron will not spike.
    ///
    /// v2.4: Also zeroes all gap-junction weights involving this neuron
    /// (design doc §16.2 — dead neuron's gap junctions must be severed).
    pub fn kill_neuron(&mut self, nid: u32) {
        let i = nid as usize;
        if i >= self.neuron_flags.len() {
            return;
        }
        self.neuron_flags[i] &= !crate::core::neuron::NEURON_ALIVE;
        self.neuron_flags[i] |= crate::core::neuron::NEURON_LESIONED;
        // Read e_leak before mutably borrowing the dynamic states.
        let e_leak = self.db.neuron_attrs()[i].e_leak;
        let st = &mut self.db.neuron_states[i];
        st.i_total = 0.0;
        st.i_syn = 0.0;
        st.i_gap = 0.0;
        st.i_ext = 0.0;
        st.v_mem = e_leak;

        // Sever all gap junctions involving this neuron.
        let gjs = self.db.gap_junctions();
        for (gi, gj) in gjs.iter().enumerate() {
            if gj.pre_neuron == nid || gj.post_neuron == nid {
                if gi < self.gap_junction_weights.len() {
                    self.gap_junction_weights[gi] = 0.0;
                    if let Some(ref mut wal) = self.wal {
                        let _ = wal.log_gap_junction_weight(gi as u32, 0.0);
                    }
                }
            }
        }
        // WAL: log neuron state + flags after mutation.
        if let Some(ref mut wal) = self.wal {
            let _ = wal.log_neuron_flags(nid, self.neuron_flags[i]);
            let _ = wal.log_neuron_state(nid, &self.db.neuron_states[i]);
        }
    }

    /// Re-activate a previously killed neuron.
    pub fn activate_neuron(&mut self, nid: u32) {
        let i = nid as usize;
        if i < self.neuron_flags.len() {
            self.neuron_flags[i] |= crate::core::neuron::NEURON_ALIVE;
            self.neuron_flags[i] &= !crate::core::neuron::NEURON_LESIONED;
            if let Some(ref mut wal) = self.wal {
                let _ = wal.log_neuron_flags(nid, self.neuron_flags[i]);
            }
        }
    }

    #[inline]
    pub fn is_alive(&self, nid: u32) -> bool {
        let i = nid as usize;
        i < self.neuron_flags.len()
            && (self.neuron_flags[i] & crate::core::neuron::NEURON_ALIVE) != 0
    }

    pub fn with_config(mut self, cfg: SimulationConfig) -> Self {
        self.config = cfg;
        self
    }

    /// Enable WAL (Write-Ahead Log) for crash recovery.
    /// Creates the WAL file at `path` and begins logging state mutations.
    pub fn enable_wal(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        let mut wal = crate::storage::wal::WalWriter::create(path)?;
        wal.set_tick(self.current_tick);
        self.wal = Some(wal);
        Ok(())
    }

    /// Disable WAL and close the file.
    pub fn disable_wal(&mut self) {
        if let Some(mut wal) = self.wal.take() {
            let _ = wal.flush();
        }
    }

    /// Pre-compute auxiliary tables: pre-neuron lookup, continuous-mode
    /// synapse list (BAAIWorm Sigmoid), and reverse CSR for STDP.
    fn precompute_aux_tables(&mut self) {
        let view = StaticView::new(&self.db.mmap, &self.db.header);
        let n_neurons = self.db.header.n_neurons as usize;
        let n_syn = self.db.header.n_synapses as usize;

        // Pre-neuron lookup + continuous-mode list (forward sweep).
        self.syn_pre_neuron = vec![0u32; n_syn];
        for pre in 0..n_neurons {
            for syn_idx in view.out_range(pre) {
                self.syn_pre_neuron[syn_idx] = pre as u32;
                if view.syn_attrs[syn_idx].syn_mode == SYN_MODE_CONTINUOUS {
                    self.continuous_synapse_indices.push(syn_idx);
                }
            }
        }

        // Reverse CSR — group synapse indices by post-neuron.
        // Two-pass counting sort: histogram → prefix sum → place.
        self.rev_csr_row_ptr = vec![0u64; n_neurons + 1];
        for s in view.syn_attrs.iter() {
            self.rev_csr_row_ptr[s.post_neuron as usize + 1] += 1;
        }
        for i in 1..=n_neurons {
            self.rev_csr_row_ptr[i] += self.rev_csr_row_ptr[i - 1];
        }
        self.rev_csr_syn_idx = vec![0u32; n_syn];
        let mut cursor = vec![0u64; n_neurons];
        for (syn_idx, s) in view.syn_attrs.iter().enumerate() {
            let post = s.post_neuron as usize;
            let pos = self.rev_csr_row_ptr[post] + cursor[post];
            self.rev_csr_syn_idx[pos as usize] = syn_idx as u32;
            cursor[post] += 1;
        }
    }

    /// Range of incoming synapse indices for `post_id` in [`Self::rev_csr_syn_idx`].
    #[inline]
    pub fn in_range(&self, post_id: usize) -> std::ops::Range<usize> {
        self.rev_csr_row_ptr[post_id] as usize..self.rev_csr_row_ptr[post_id + 1] as usize
    }

    pub fn add_observer<O: BrainObserver + 'static>(&mut self, obs: O) {
        self.observers.push(Box::new(obs));
    }

    // ── Stimulus / pattern I/O (§14.2) ───────────────────────────────────

    /// Inject `current` (pA) into `neuron_id` for `duration_ticks` ticks.
    /// Subsequent calls before expiry simply extend / overwrite the deadline.
    pub fn present_stimulus(&mut self, pattern: &[(u32, f32)], duration_ticks: u64) {
        for &(nid, current) in pattern {
            let i = nid as usize;
            if i < self.db.neuron_states.len() {
                self.db.neuron_states[i].i_ext = current;
                if !self.active_stimulus_neurons.contains(&nid) {
                    self.active_stimulus_neurons.push(nid);
                }
            }
        }
        self.stimulus_end_tick = self.current_tick + duration_ticks;
    }

    pub fn clear_stimulus(&mut self) {
        for &nid in &self.active_stimulus_neurons {
            let i = nid as usize;
            if i < self.db.neuron_states.len() {
                self.db.neuron_states[i].i_ext = 0.0;
            }
        }
        self.active_stimulus_neurons.clear();
    }

    /// Read instantaneous v_mem of every neuron in a contiguous range.
    pub fn read_vmem_range(&self, start: u32, end: u32) -> Vec<f32> {
        let s = start as usize;
        let e = (end as usize).min(self.db.neuron_states.len());
        self.db.neuron_states[s..e].iter().map(|st| st.v_mem).collect()
    }

    /// Spike-count snapshot of every neuron in a contiguous range.
    pub fn read_spike_counts(&self, start: u32, end: u32) -> Vec<u32> {
        let s = start as usize;
        let e = (end as usize).min(self.db.neuron_states.len());
        self.db.neuron_states[s..e].iter().map(|st| st.spike_count).collect()
    }

    /// Firing-rate pattern for a brain region (design doc §14.2).
    ///
    /// Returns Hz per neuron in the region, computed as
    /// `spike_count / (window_ticks * dt * 1e-3)`.
    /// Note: this gives the *cumulative* rate since simulation start.
    /// For a sliding-window rate, capture spike_count at two time points
    /// and compute the delta.
    pub fn read_firing_rate_pattern(&self, region_id: u32, window_ticks: u64) -> Vec<f32> {
        let regions = self.db.regions();
        if region_id as usize >= regions.len() || window_ticks == 0 {
            return Vec::new();
        }
        let region = &regions[region_id as usize];
        let first = region.first_neuron as usize;
        let n = region.neuron_count as usize;
        let window_s = window_ticks as f32 * self.dt * 1e-3; // seconds
        let mut rates = Vec::with_capacity(n);
        for i in first..first + n {
            if i < self.db.neuron_states.len() {
                let rate = self.db.neuron_states[i].spike_count as f32 / window_s;
                rates.push(rate);
            }
        }
        rates
    }

    /// Instantaneous membrane-potential pattern for a brain region (§14.2).
    pub fn read_vmem_pattern(&self, region_id: u32) -> Vec<f32> {
        let regions = self.db.regions();
        if region_id as usize >= regions.len() {
            return Vec::new();
        }
        let region = &regions[region_id as usize];
        let first = region.first_neuron as usize;
        let n = region.neuron_count as usize;
        let mut vmems = Vec::with_capacity(n);
        for i in first..first + n {
            if i < self.db.neuron_states.len() {
                vmems.push(self.db.neuron_states[i].v_mem);
            }
        }
        vmems
    }

    // ── Main loop ────────────────────────────────────────────────────────

    /// Run `n_ticks` steps.
    pub fn run(&mut self, n_ticks: u64) {
        for _ in 0..n_ticks {
            self.step();
        }
    }

    /// Closed-loop simulation step (design doc §15.5).
    ///
    /// 1. Get sensory input from the body/environment.
    /// 2. Run one brain simulation step.
    /// 3. Read motor neuron output back to the body.
    /// 4. Step the body/environment forward.
    pub fn step_closed_loop(&mut self, body: &mut dyn BodyEnvironment) {
        let tick = self.current_tick;
        // 1. Sensory input → i_ext on sensory neurons.
        let sensory = body.get_sensory_input(tick);
        for &(neuron_id, current) in &sensory {
            let i = neuron_id as usize;
            if i < self.db.neuron_states.len() {
                self.db.neuron_states[i].i_ext = current;
            }
        }
        // 2. Brain step.
        self.step();
        // 3. Read motor output.
        let regions = self.db.regions();
        let mut motor_output: Vec<(u32, f32)> = Vec::new();
        for region in regions {
            let start = region.motor_start as usize;
            let end = region.motor_end as usize;
            for i in start..end {
                if i < self.db.neuron_states.len() {
                    motor_output.push((i as u32, self.db.neuron_states[i].v_mem));
                }
            }
        }
        body.set_motor_output(tick, &motor_output);
        // 4. Step body.
        body.step_body(self.dt);
    }

    /// Single tick.
    pub fn step(&mut self) {
        let tick = self.current_tick;
        let dt = self.dt;

        self.recently_fired.clear();

        // Phase 1 — gap junctions (continuous coupling).
        self.phase1_gap_junctions();

        // Phase 2 — drain delay-ring events into rise/decay.
        self.phase2_drain_events(tick);

        // Phase 3 — event-driven conductance decay & current injection.
        self.phase3_event_synapses(dt);

        // Phase 3b — continuous-conductance synapses.
        self.phase3b_continuous_synapses(dt);

        // Phase 3c — expire stimuli.
        if self.current_tick >= self.stimulus_end_tick
            && !self.active_stimulus_neurons.is_empty()
        {
            self.clear_stimulus();
        }

        // Phase 4 — neurons.
        self.phase4_update_neurons(tick, dt);

        // Phase 5 — STDP trace decay + periodic weight write-back.
        if self.config.stdp_enabled {
            let decay = (-dt / self.config.stdp_tau).exp();
            for st in self.db.neuron_states.iter_mut() {
                st.stdp_trace *= decay;
            }
            if self.config.stdp_apply_every > 0
                && (self.current_tick + 1) % self.config.stdp_apply_every == 0
            {
                self.apply_dw_accum();
            }
        }

        // Phase 6 — neuromodulator diffusion (design doc §5.3).
        if self.config.modulation_enabled {
            self.update_neuromodulation(dt);
        }

        // Phase 7 — structural plasticity (low frequency, every 10000 tick = 1 s).
        if self.config.structural_plasticity_enabled && self.current_tick % 10000 == 0 {
            self.structural_plasticity();
        }

        // Phase 8 — observers.
        if !self.observers.is_empty() {
            let states = self.db.neuron_states.as_slice();
            for obs in self.observers.iter_mut() {
                obs.on_step_done(tick, states);
            }
        }

        // Phase 9 — WAL (Write-Ahead Log) for crash recovery.
        // Log all state mutations that occurred this tick so that after a
        // crash the simulation can be restored from the last snapshot + WAL.
        if let Some(ref mut wal) = self.wal {
            wal.set_tick(self.current_tick);
            // Log recently-spiked neurons (their state was mutated).
            for &nid in &self.recently_fired {
                let i = nid as usize;
                if i < self.db.neuron_states.len() {
                    let _ = wal.log_neuron_state(nid, &self.db.neuron_states[i]);
                }
            }
            // Log synapse weight changes from STDP batch apply.
            if self.config.stdp_enabled
                && self.config.stdp_apply_every > 0
                && (self.current_tick + 1) % self.config.stdp_apply_every == 0
            {
                for (idx, st) in self.db.syn_states.iter().enumerate() {
                    if st.dw_accum != 0.0 {
                        let _ = wal.log_synapse_state(idx as u32, st);
                    }
                }
            }
            // Log gap-junction weight changes (from kill_neuron).
            // (Logged at kill_neuron call site, not here.)
            // Log neuron flag changes (from kill/activate).
            // (Logged at kill_neuron/activate_neuron call sites, not here.)
            // Periodic checkpoint + flush.
            if self.wal_checkpoint_interval > 0
                && (self.current_tick + 1) % self.wal_checkpoint_interval == 0
            {
                let _ = wal.write_checkpoint();
                let _ = wal.flush();
            }
        }

        self.current_tick += 1;
    }

    // ── Phase implementations ────────────────────────────────────────────

    fn phase1_gap_junctions(&mut self) {
        let view = StaticView::new(&self.db.mmap, &self.db.header);
        // For now point-neurons only: accumulate into NeuronState.i_total.
        // Uses mutable gap_junction_weights (copy of mmap weights) so that
        // kill_neuron can sever gap junctions by zeroing the weight.
        for (gi, gj) in view.gap_junctions.iter().enumerate() {
            let pre = gj.pre_neuron as usize;
            let post = gj.post_neuron as usize;
            if pre >= self.db.neuron_states.len() || post >= self.db.neuron_states.len() {
                continue;
            }
            // Skip dead neurons (lesioned / never-activated reserved slots).
            if self.neuron_flags[pre] & crate::core::neuron::NEURON_ALIVE == 0
                || self.neuron_flags[post] & crate::core::neuron::NEURON_ALIVE == 0
            {
                continue;
            }
            let w = if gi < self.gap_junction_weights.len() {
                self.gap_junction_weights[gi]
            } else {
                gj.weight // fallback to mmap value
            };
            if w == 0.0 { continue; } // Severed gap junction.
            let v_pre = self.db.neuron_states[pre].v_mem;
            let v_post = self.db.neuron_states[post].v_mem;
            let i_gj = w * (v_pre - v_post); // nS·mV = pA
            self.db.neuron_states[post].i_total += i_gj;
            self.db.neuron_states[pre].i_total -= i_gj;

            // Debug split.
            self.db.neuron_states[post].i_gap += i_gj;
            self.db.neuron_states[pre].i_gap -= i_gj;
        }
    }

    fn phase2_drain_events(&mut self, tick: u64) {
        // Snapshot ring contents at this tick into a small vector so we can
        // mutate `active_synapses` and `syn_states` freely.
        let events: Vec<SynapticEvent> = {
            let drained = self.event_ring.drain_at(tick);
            drained.to_vec()
        };
        for ev in events {
            let i = ev.syn_id as usize;
            if i >= self.db.syn_states.len() {
                continue;
            }
            let st = &mut self.db.syn_states[i];
            st.g_rise += ev.delta_g;
            st.g_decay += ev.delta_g;
            if st.is_active == 0 {
                st.is_active = 1;
                self.active_synapses.push_back(i);
            }
        }
    }

    fn phase3_event_synapses(&mut self, dt: f32) {
        let view = StaticView::new(&self.db.mmap, &self.db.header);

        // Drain & rebuild active list to keep it tight.
        let estimated = self.active_synapses.len() * 11 / 10 + 64;
        let mut next_active: VecDeque<usize> = VecDeque::with_capacity(estimated);

        while let Some(i) = self.active_synapses.pop_front() {
            let attr = view.syn_attrs[i];
            if attr.syn_mode != SYN_MODE_EVENT_DRIVEN {
                // Continuous-mode synapses live in their own list.
                self.db.syn_states[i].is_active = 0;
                continue;
            }
            let receptor = view.receptors[attr.receptor_type as usize];

            // Decay rise / decay.
            let exp_r = (-dt / receptor.tau_rise.max(1e-6)).exp();
            let exp_d = (-dt / receptor.tau_decay.max(1e-6)).exp();
            let st = &mut self.db.syn_states[i];
            st.g_rise *= exp_r;
            st.g_decay *= exp_d;
            let g = st.g_rise - st.g_decay;

            if g.abs() < self.config.min_active_conductance {
                st.is_active = 0;
                continue;
            }
            next_active.push_back(i);

            // Short-term resource recovery.
            let tau_rec = attr.tau_rec.max(1e-3);
            st.r += dt / tau_rec * (1.0 - st.r);

            // Post-synaptic neuron voltage — respect post_comp for multi-comp.
            let post_n = attr.post_neuron as usize;
            if post_n >= self.db.neuron_states.len() {
                continue;
            }
            let (v_post, post_comp_idx) = if attr.post_comp > 0 {
                let post_first = view.neuron_attrs[post_n].first_comp_id as usize;
                let post_ncomp = view.neuron_attrs[post_n].n_compartment as usize;
                let comp_idx = post_first + attr.post_comp as usize;
                if comp_idx < post_first + post_ncomp && comp_idx < self.db.comp_states.len() {
                    (self.db.comp_states[comp_idx].v_mem, Some(comp_idx))
                } else {
                    (self.db.neuron_states[post_n].v_mem, None)
                }
            } else {
                (self.db.neuron_states[post_n].v_mem, None)
            };
            let mut i_syn = g * (v_post - receptor.e_rev);

            // NMDA Mg²⁺ block (Jahr–Stevens).
            if attr.receptor_type == RECEPTOR_NMDA {
                let mg_block = 1.0
                    / (1.0 + (receptor.mg_conc / 3.57) * (-0.062 * v_post).exp());
                i_syn *= mg_block;
            }

            // Inject current into the correct target (compartment or point neuron).
            if let Some(ci) = post_comp_idx {
                self.db.comp_states[ci].i_total += i_syn;
            } else {
                self.db.neuron_states[post_n].i_total += i_syn;
            }
            self.db.neuron_states[post_n].i_syn += i_syn;
        }
        self.active_synapses = next_active;
    }

    fn phase3b_continuous_synapses(&mut self, dt: f32) {
        let view = StaticView::new(&self.db.mmap, &self.db.header);
        for &i in self.continuous_synapse_indices.iter() {
            let attr = view.syn_attrs[i];
            let receptor = view.receptors[attr.receptor_type as usize];

            let pre_n = self.syn_pre_neuron[i] as usize;
            let post_n = attr.post_neuron as usize;
            if pre_n >= self.db.neuron_states.len() || post_n >= self.db.neuron_states.len() {
                continue;
            }

            // v_pre: for multi-compartment neurons, use the specific compartment.
            let v_pre = if attr.pre_comp > 0 {
                let pre_first = view.neuron_attrs[pre_n].first_comp_id as usize;
                let pre_ncomp = view.neuron_attrs[pre_n].n_compartment as usize;
                let comp_idx = pre_first + attr.pre_comp as usize;
                if comp_idx < pre_first + pre_ncomp && comp_idx < self.db.comp_states.len() {
                    self.db.comp_states[comp_idx].v_mem
                } else {
                    self.db.neuron_states[pre_n].v_mem
                }
            } else {
                self.db.neuron_states[pre_n].v_mem
            };

            // v_post: for multi-compartment neurons, use the specific compartment.
            let (v_post, post_comp_idx) = if attr.post_comp > 0 {
                let post_first = view.neuron_attrs[post_n].first_comp_id as usize;
                let post_ncomp = view.neuron_attrs[post_n].n_compartment as usize;
                let comp_idx = post_first + attr.post_comp as usize;
                if comp_idx < post_first + post_ncomp && comp_idx < self.db.comp_states.len() {
                    (self.db.comp_states[comp_idx].v_mem, Some(comp_idx))
                } else {
                    (self.db.neuron_states[post_n].v_mem, None)
                }
            } else {
                (self.db.neuron_states[post_n].v_mem, None)
            };

            // Sigmoid steady-state activation.
            let inf =
                1.0 / (1.0 + ((receptor.v_threshold - v_pre) / receptor.v_slope).exp());
            // Time-constant scales with (1 - inf) (BAAIWorm exc_syn_advance.mod).
            let tau = ((1.0 - inf) / receptor.k_rate.max(1e-6)).max(dt);

            let st = &mut self.db.syn_states[i];
            st.g_rise += (inf - st.g_rise) / tau * dt;
            let g = st.weight * st.g_rise;
            let mut i_syn = g * (v_post - receptor.e_rev);

            // NMDA Mg²⁺ block (Jahr–Stevens) for continuous-mode NMDA synapses.
            if attr.receptor_type == RECEPTOR_NMDA {
                let mg_block = 1.0
                    / (1.0 + (receptor.mg_conc / 3.57) * (-0.062 * v_post).exp());
                i_syn *= mg_block;
            }

            // Inject current into the correct target (compartment or point neuron).
            if let Some(ci) = post_comp_idx {
                self.db.comp_states[ci].i_total += i_syn;
            } else {
                self.db.neuron_states[post_n].i_total += i_syn;
            }
            self.db.neuron_states[post_n].i_syn += i_syn;
        }
    }

    fn phase4_update_neurons(&mut self, tick: u64, dt: f32) {
        // We need to: read attrs (static, via mmap), look up NeuronTypeParams
        // (in self.db.meta), mutate states, optionally enqueue spike events.
        let view = StaticView::new(&self.db.mmap, &self.db.header);
        let n = view.neuron_attrs.len();

        // We build spike events into a local vector to avoid borrowing
        // `self` mutably twice (once for states, once for event_ring).
        let mut spike_events: Vec<(u32, u64)> = Vec::new();

        if self.config.parallel_neuron_update && n > 256 {
            // ── Parallel path (rayon) ────────────────────────────────
            // Point neurons are updated in parallel; multi-compartment
            // neurons are deferred to a sequential pass because they
            // share comp_states (write-write race).
            use rayon::prelude::*;

            // Snapshot flags and attrs for read-only access.
            let flags = &self.neuron_flags;
            let attrs = view.neuron_attrs;
            let neuron_types = &self.db.meta.neuron_types;
            let spike_thresh = self.config.spike_threshold_mv;
            let adapt_w_rate = self.config.adapt_w_rate;
            let cai_tau = self.config.cai_tau;

            // Phase 4a: parallel point-neuron updates.
            // Collect indices of multi-compartment neurons for sequential pass.
            let mut multicomp_indices: Vec<usize> = Vec::new();

            // First pass: identify point vs multi-comp neurons.
            // HH model neurons always use the compartment-based solver
            // because gate variables (m_na, h_na, m_k, ...) live in
            // CompartmentState, not NeuronState.
            for i in 0..n {
                if flags[i] & crate::core::neuron::NEURON_ALIVE == 0 {
                    self.db.neuron_states[i].i_total = 0.0;
                    continue;
                }
                let tp_opt = neuron_types.get(attrs[i].neuron_type as usize);
                let model = tp_opt.map(|t| t.model).unwrap_or(NeuronModel::Izhikevich);
                let needs_comp = attrs[i].n_compartment > 1
                    || matches!(model, NeuronModel::HodgkinHuxley | NeuronModel::MultiCompartmentHH);
                if needs_comp {
                    multicomp_indices.push(i);
                }
            }

            // Parallel point-neuron update (Izhikevich / LIF / Graded only).
            // HH model neurons are handled in the sequential multicomp pass.
            let spiked_ids: Vec<u32> = self.db.neuron_states[..n]
                .par_iter_mut()
                .enumerate()
                .filter_map(|(i, st)| {
                    if flags[i] & crate::core::neuron::NEURON_ALIVE == 0 {
                        return None;
                    }
                    let attr = attrs[i];
                    if attr.n_compartment > 1 {
                        return None;
                    }
                    let tp_opt = neuron_types.get(attr.neuron_type as usize);
                    let model = tp_opt.map(|t| t.model).unwrap_or(NeuronModel::Izhikevich);
                    if matches!(model, NeuronModel::HodgkinHuxley | NeuronModel::MultiCompartmentHH) {
                        return None; // handled by multicomp solver
                    }
                    let spiked = update_point_neuron(
                        attr, tp_opt, st, dt, tick,
                        spike_thresh, adapt_w_rate, cai_tau,
                    );
                    if spiked { Some(i as u32) } else { None }
                })
                .collect();

            // Sequential multi-compartment update (shared comp_states).
            for &i in &multicomp_indices {
                let attr = attrs[i];
                let tp_opt = neuron_types.get(attr.neuron_type as usize);
                let first = attr.first_comp_id as usize;
                let n_comp = attr.n_compartment as usize;
                if first.checked_add(n_comp)
                    .map_or(true, |end| end > view.compartment_attrs.len())
                {
                    self.skipped_multicomp += 1;
                    continue;
                }
                let spiked = update_multicomp_neuron(
                    attr, tp_opt, view.compartment_attrs,
                    &mut self.db.comp_states,
                    &mut self.db.neuron_states[i],
                    dt, tick, spike_thresh,
                    &self.db.meta.ion_channels,
                    &self.db.meta.ion_channel_sets,
                );
                if spiked {
                    spike_events.push((i as u32, tick));
                }
            }

            // Merge parallel spikes.
            for nid in spiked_ids {
                self.recently_fired.push(nid);
                spike_events.push((nid, tick));
            }
        } else {
            // ── Sequential path (default) ─────────────────────────────
            for i in 0..n {
                let attr = view.neuron_attrs[i];
                if self.neuron_flags[i] & crate::core::neuron::NEURON_ALIVE == 0 {
                    self.db.neuron_states[i].i_total = 0.0;
                    continue;
                }
                let tp_opt = self.db.meta.neuron_types.get(attr.neuron_type as usize);
                let model = tp_opt.map(|t| t.model).unwrap_or(NeuronModel::Izhikevich);
                // HH model neurons always use the compartment-based solver
                // because gate variables live in CompartmentState.
                let needs_comp = attr.n_compartment > 1
                    || matches!(model, NeuronModel::HodgkinHuxley | NeuronModel::MultiCompartmentHH);
                let spiked = if needs_comp {
                    let first = attr.first_comp_id as usize;
                    let n_comp = attr.n_compartment.max(1) as usize;
                    if first.checked_add(n_comp)
                        .map_or(true, |end| end > view.compartment_attrs.len())
                    {
                        // Malformed neuron — skip safely.
                        self.skipped_multicomp += 1;
                        false
                    } else {
                        update_multicomp_neuron(
                            attr,
                            tp_opt,
                            view.compartment_attrs,
                            &mut self.db.comp_states,
                            &mut self.db.neuron_states[i],
                            dt,
                            tick,
                            self.config.spike_threshold_mv,
                            &self.db.meta.ion_channels,
                            &self.db.meta.ion_channel_sets,
                        )
                    }
                } else {
                    update_point_neuron(
                        attr,
                        tp_opt,
                        &mut self.db.neuron_states[i],
                        dt,
                        tick,
                        self.config.spike_threshold_mv,
                        self.config.adapt_w_rate,
                        self.config.cai_tau,
                    )
                };

                if spiked {
                    self.recently_fired.push(i as u32);
                    spike_events.push((i as u32, tick));
                }
            }
        }

        // Apply post-side LTP first (uses pre traces *before* they decay or
        // get bumped by their own emit_spike), then emit downstream events.
        if self.config.stdp_enabled {
            for &(nid, _) in &spike_events {
                self.apply_post_stdp_ltp(nid);
            }
        }
        for &(nid, t) in &spike_events {
            self.emit_spike(nid, t);
        }
    }

    fn emit_spike(&mut self, neuron_id: u32, tick: u64) {
        let view = StaticView::new(&self.db.mmap, &self.db.header);
        let range = view.out_range(neuron_id as usize);
        for syn_idx in range {
            let attr = view.syn_attrs[syn_idx];
            // Tsodyks–Markram update on the *pre-synaptic* spike (v2.4).
            let st = &mut self.db.syn_states[syn_idx];
            st.u += attr.u_se * (1.0 - st.u);
            let effective = st.weight * st.u * st.r;
            st.r -= st.u * st.r;
            if st.r < 0.0 { st.r = 0.0; }

            // Event-driven: deposit conductance at arrival tick.
            if attr.syn_mode == SYN_MODE_EVENT_DRIVEN {
                let arrival = tick + attr.delay_ticks as u64;
                debug_assert!(
                    (attr.delay_ticks as usize) < self.event_ring.size(),
                    "delay_ticks {} exceeds ring_size {}",
                    attr.delay_ticks,
                    self.event_ring.size()
                );
                self.event_ring.push(SynapticEvent {
                    tick: arrival,
                    syn_id: syn_idx as u32,
                    delta_g: effective,
                });
            }
            // Continuous mode does not use events; voltage drives release.

            // Pre-side STDP — LTD using post.stdp_trace (Song 2000).
            // v2.4: Dopamine modulation on LTD as well.
            if self.config.stdp_enabled {
                let post = attr.post_neuron as usize;
                if post < self.db.neuron_states.len() {
                    let post_trace = self.db.neuron_states[post].stdp_trace;
                    // Dopamine from pre-neuron's region (pre just fired).
                    let da = {
                        let region_id = self.db.neuron_attrs()[neuron_id as usize].region_id as usize;
                        if region_id < self.region_modulation.len() {
                            self.region_modulation[region_id].dopamine
                        } else {
                            0.0
                        }
                    };
                    let da_mod = 1.0 + da * 3.0;
                    self.db.syn_states[syn_idx].dw_accum -=
                        self.config.stdp_a_minus * post_trace * da_mod;
                }
            }
        }

        for obs in self.observers.iter_mut() {
            obs.on_spike(neuron_id as u64, tick);
        }
    }

    /// Post-side STDP — LTP on every incoming synapse, using pre.stdp_trace.
    /// Called for each neuron that just spiked, *before* its own trace is
    /// bumped to 1.0 inside [`update_point_neuron`].
    ///
    /// v2.4: Dopamine modulation — dw *= (1 + da * 3.0) from brain region.
    fn apply_post_stdp_ltp(&mut self, post_id: u32) {
        let s = self.rev_csr_row_ptr[post_id as usize] as usize;
        let e = self.rev_csr_row_ptr[post_id as usize + 1] as usize;

        // Dopamine level from the post-neuron's brain region.
        let da = if (post_id as usize) < self.db.neuron_attrs().len() {
            let region_id = self.db.neuron_attrs()[post_id as usize].region_id as usize;
            if region_id < self.region_modulation.len() {
                self.region_modulation[region_id].dopamine
            } else {
                0.0
            }
        } else {
            0.0
        };
        let da_mod = 1.0 + da * 3.0; // Dopamine amplifies LTP.

        for &syn_idx_u32 in &self.rev_csr_syn_idx[s..e] {
            let syn_idx = syn_idx_u32 as usize;
            let pre_n = self.syn_pre_neuron[syn_idx] as usize;
            let pre_trace = self.db.neuron_states[pre_n].stdp_trace;
            self.db.syn_states[syn_idx].dw_accum +=
                self.config.stdp_a_plus * pre_trace * da_mod;
        }
    }

    /// Move accumulated `dw_accum` into `weight`, clamp to `[0, max_syn_weight]`,
    /// reset `dw_accum`, and notify observers via `on_synapse_change`.
    fn apply_dw_accum(&mut self) {
        let max_w = self.config.max_syn_weight;
        let has_obs = !self.observers.is_empty();
        for (idx, st) in self.db.syn_states.iter_mut().enumerate() {
            if st.dw_accum == 0.0 {
                continue;
            }
            let old = st.weight;
            let mut w = old + st.dw_accum;
            if w < 0.0 { w = 0.0; }
            if w > max_w { w = max_w; }
            st.weight = w;
            st.dw_accum = 0.0;
            if has_obs && (w - old).abs() > 0.0 {
                for obs in self.observers.iter_mut() {
                    obs.on_synapse_change(idx as u64, old, w);
                    obs.on_plasticity_event(
                        crate::sim::observer::PlasticityEvent::Stdp {
                            syn_id: idx as u64,
                            dw: w - old,
                        },
                    );
                }
            }
        }
    }

    /// Phase 6 — neuromodulator diffusion & decay (design doc §5.3).
    ///
    /// Each region's modulator concentrations diffuse along `LongRangePathway`
    /// connections and decay toward zero. This is a simplified linear model:
    ///   - Diffusion: Δc_i = rate * (c_j - c_i) for each connected region j
    ///   - Decay: Δc_i = -decay_rate * c_i
    fn update_neuromodulation(&mut self, dt: f32) {
        let n_regions = self.region_modulation.len();
        if n_regions == 0 { return; }

        let diff_rate = self.config.modulation_diffusion_rate;
        let decay_rate = self.config.modulation_decay_rate;

        // Compute diffusion deltas (read from current state, write to deltas).
        let mut deltas = vec![ModulationLevel::default(); n_regions];
        let pathways = self.db.pathways();
        for pw in pathways {
            let src = pw.source_region as usize;
            let tgt = pw.target_region as usize;
            if src >= n_regions || tgt >= n_regions { continue; }
            // Diffuse from source → target (proportional to fiber_count).
            let scale = diff_rate * pw.fiber_count as f32 * dt;
            let s = &self.region_modulation[src];
            let t = &self.region_modulation[tgt];
            // Dopamine
            let dd = scale * (s.dopamine - t.dopamine);
            deltas[tgt].dopamine += dd;
            deltas[src].dopamine -= dd;
            // Serotonin
            let ds = scale * (s.serotonin - t.serotonin);
            deltas[tgt].serotonin += ds;
            deltas[src].serotonin -= ds;
            // Acetylcholine
            let da = scale * (s.acetylcholine - t.acetylcholine);
            deltas[tgt].acetylcholine += da;
            deltas[src].acetylcholine -= da;
            // Noradrenaline
            let dn = scale * (s.noradrenaline - t.noradrenaline);
            deltas[tgt].noradrenaline += dn;
            deltas[src].noradrenaline -= dn;
        }

        // Apply diffusion + decay.
        for (i, m) in self.region_modulation.iter_mut().enumerate() {
            m.dopamine += deltas[i].dopamine - decay_rate * m.dopamine * dt;
            m.serotonin += deltas[i].serotonin - decay_rate * m.serotonin * dt;
            m.acetylcholine += deltas[i].acetylcholine - decay_rate * m.acetylcholine * dt;
            m.noradrenaline += deltas[i].noradrenaline - decay_rate * m.noradrenaline * dt;
            // Clamp to non-negative.
            m.dopamine = m.dopamine.max(0.0);
            m.serotonin = m.serotonin.max(0.0);
            m.acetylcholine = m.acetylcholine.max(0.0);
            m.noradrenaline = m.noradrenaline.max(0.0);
        }
    }

    /// Phase 7 — structural plasticity (design doc §5.5).
    ///
    /// Simplified activity-dependent rule:
    ///   - **Sprout**: if two neurons in the same region have both fired within
    ///     the last `sp_window` ticks and there is no existing synapse between
    ///     them, create one with `sp_init_weight`.
    ///   - **Prune**: if a synapse's weight has decayed below `sp_prune_threshold`,
    ///     remove it.
    ///   - **Cap**: each neuron can have at most `sp_max_out_degree` outgoing
    ///     synapses (including the static ones).
    ///
    /// All mutations go through [`DynamicCSR`], which batches inserts/deletes
    /// and rebuilds the CSR when the delta area grows too large.
    fn structural_plasticity(&mut self) {
        // Lazily initialise DynamicCSR from the static base CSR.
        if self.dynamic_csr.is_none() {
            let view = StaticView::new(&self.db.mmap, &self.db.header);
            let n_neurons = view.neuron_attrs.len();
            let row_ptr = view.csr_row_ptr.to_vec();
            let col_idx = view.csr_col_idx.to_vec();
            let syn_attrs: Vec<SynapseAttr> = view.syn_attrs.to_vec();
            self.dynamic_csr = Some(DynamicCSR::new(
                row_ptr, col_idx, syn_attrs, n_neurons as u32,
            ));
        }

        let sp_window = self.config.sp_window;
        let sp_init_weight = self.config.sp_init_weight;
        let sp_prune_threshold = self.config.sp_prune_threshold;
        let sp_max_out = self.config.sp_max_out_degree;
        let tick = self.current_tick;

        let dcsr = self.dynamic_csr.as_mut().unwrap();
        let n_neurons = self.db.neuron_states.len();

        // ── Prune weak synapses (base CSR) ─────────────────────────────
        if sp_prune_threshold > 0.0 {
            for i in 0..self.db.syn_states.len() {
                if dcsr.is_deleted(i as u64) { continue; }
                if self.db.syn_states[i].weight.abs() < sp_prune_threshold {
                    dcsr.remove_synapse(i as u64);
                    for obs in self.observers.iter_mut() {
                        obs.on_plasticity_event(
                            crate::sim::observer::PlasticityEvent::Prune {
                                syn_id: i as u64,
                                old_weight: self.db.syn_states[i].weight,
                            },
                        );
                    }
                }
            }
        }

        // ── Sprout new synapses between co-active neurons ───────────────
        if sp_init_weight > 0.0 && sp_window > 0 {
            // Collect recently-active neurons (fired within sp_window).
            let active: Vec<u32> = (0..n_neurons)
                .filter(|&n| {
                    let lst = self.db.neuron_states[n].last_spike_tick;
                    lst != u64::MAX && tick.saturating_sub(lst) < sp_window
                })
                .map(|n| n as u32)
                .collect();

            let attrs = self.db.neuron_attrs();
            for &pre in &active {
                let pre_region = attrs.get(pre as usize).map(|a| a.region_id).unwrap_or(u32::MAX);
                // Check out-degree cap (base + delta).
                let base_out = dcsr.csr_out_range(pre).len();
                let delta_out = dcsr.delta_out_synapses(pre).count();
                if (base_out + delta_out) >= sp_max_out {
                    continue;
                }
                for &post in &active {
                    if pre == post { continue; }
                    let post_region = attrs.get(post as usize).map(|a| a.region_id).unwrap_or(u32::MAX);
                    if pre_region != post_region { continue; }
                    // Check if edge already exists in base CSR.
                    let exists_base = dcsr.csr_out_range(pre).any(|i| {
                        dcsr.col_idx.get(i).map(|&c| c as u32 == post).unwrap_or(false)
                            && !dcsr.is_deleted(i as u64)
                    });
                    // Check delta area.
                    let exists_delta = dcsr.delta_out_synapses(pre)
                        .any(|(_, a)| a.post_neuron == post);
                    if exists_base || exists_delta {
                        continue;
                    }
                    let attr = SynapseAttr {
                        post_neuron: post,
                        base_weight: sp_init_weight,
                        syn_type: SYN_EXCITATORY,
                        syn_mode: SYN_MODE_EVENT_DRIVEN,
                        delay_ticks: 1,
                        ..Default::default()
                    };
                    let _sid = dcsr.insert_synapse(pre, attr);
                    for obs in self.observers.iter_mut() {
                        obs.on_plasticity_event(
                            crate::sim::observer::PlasticityEvent::Sprout {
                                pre: pre as u64,
                                post: post as u64,
                                new_weight: sp_init_weight,
                            },
                        );
                    }
                    // Respect out-degree cap.
                    let new_out = base_out + dcsr.delta_out_synapses(pre).count();
                    if new_out >= sp_max_out {
                        break;
                    }
                }
            }
        }

        // Rebuild the DynamicCSR if the delta area has grown too large.
        if dcsr.should_rebuild() {
            dcsr.rebuild();
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Point-neuron integration (Izhikevich / LIF / Graded)
// ────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn update_point_neuron(
    attr: NeuronAttr,
    tp: Option<&NeuronTypeParams>,
    state: &mut NeuronState,
    dt: f32,
    tick: u64,
    spike_thresh: f32,
    adapt_w_rate: f32,
    cai_tau: f32,
) -> bool {
    let i_total = state.i_total + state.i_ext + state.adapt_w;

    let model = tp.map(|t| t.model).unwrap_or(NeuronModel::Izhikevich);

    let mut spiked = false;
    match model {
        NeuronModel::Izhikevich => {
            let iz = tp.map(|t| t.iz_params).unwrap_or_else(|| {
                crate::core::neuron_type::IzhikevichParams::regular_spiking()
            });
            // Normalise current to Izhikevich mV/ms scale via iz.cm.
            let i_norm = i_total / iz.cm.max(1e-6);
            let v = state.v_mem;
            let u = state.u;
            let dv = 0.04 * v * v + 5.0 * v + 140.0 - u + i_norm;
            let du = iz.a * (iz.b * v - u);
            state.v_mem += dv * dt;
            state.u += du * dt;

            if state.v_mem >= spike_thresh {
                state.v_mem = iz.c;
                state.u += iz.d;
                state.last_spike_tick = tick;
                state.spike_count = state.spike_count.saturating_add(1);
                state.stdp_trace = 1.0;
                state.i_total = 0.0;
                state.i_syn = 0.0;
                state.i_gap = 0.0;
                return true; // skip slow-time updates after reset
            }
        }
        NeuronModel::LIF => {
            let cm = attr.cm.max(1e-6);
            let g_leak = attr.g_leak;
            let v = state.v_mem;
            // dv = ( -g_leak*(v - e_leak) + i_total ) / cm
            let dv = (-g_leak * (v - attr.e_leak) + i_total) / cm;
            state.v_mem += dv * dt;
            if state.v_mem >= spike_thresh {
                state.v_mem = attr.e_leak; // reset to leak
                state.last_spike_tick = tick;
                state.spike_count = state.spike_count.saturating_add(1);
                state.stdp_trace = 1.0;
                state.i_total = 0.0;
                state.i_syn = 0.0;
                state.i_gap = 0.0;
                return true;
            }
        }
        NeuronModel::Graded => {
            // Non-spiking: simple leak + input integration, never resets.
            let cm = attr.cm.max(1e-6);
            let g_leak = attr.g_leak;
            let v = state.v_mem;
            let dv = (-g_leak * (v - attr.e_leak) + i_total) / cm;
            state.v_mem += dv * dt;
            // Clamp to a reasonable physiological range.
            if state.v_mem > 50.0 { state.v_mem = 50.0; }
            if state.v_mem < -100.0 { state.v_mem = -100.0; }

            // Graded spike emission: when v_mem exceeds a moderate threshold,
            // emit a "graded spike" so that STP updates and observer callbacks
            // fire (needed for continuous synapses driven by v_pre).
            let graded_thresh = -40.0; // mV
            let graded_refractory: u64 = 500; // 50 ms at dt=0.1 ms
            let in_graded_refractory = match state.last_spike_tick {
                u64::MAX => false,
                t => tick.saturating_sub(t) < graded_refractory,
            };
            if !in_graded_refractory && state.v_mem >= graded_thresh {
                state.last_spike_tick = tick;
                state.spike_count = state.spike_count.saturating_add(1);
                state.stdp_trace = state.stdp_trace.max(0.5);
                spiked = true;
            }
        }
        NeuronModel::HodgkinHuxley | NeuronModel::MultiCompartmentHH => {
            // Should never reach here — HH model neurons are routed to
            // update_multicomp_neuron() in phase4_update_neurons() because
            // gate variables (m_na, h_na, m_k, ...) live in CompartmentState.
            // Fallback: simple leak integration to avoid crashing.
            let cm = attr.cm.max(1e-6);
            let dv = (-attr.g_leak * (state.v_mem - attr.e_leak) + i_total) / cm;
            state.v_mem += dv * dt;
        }
    }

    // Slow housekeeping (skipped if we returned early on a spike reset).
    state.adapt_w += dt * adapt_w_rate * (state.v_mem - attr.e_leak);
    state.cai += dt * (-state.cai / cai_tau.max(1e-3));
    state.v_mem_soma = state.v_mem;
    // Clear current buffer for next tick (i_ext is NOT cleared per v2.4).
    state.i_total = 0.0;
    state.i_syn = 0.0;
    state.i_gap = 0.0;

    spiked
}

// ────────────────────────────────────────────────────────────────────────────
// Multi-compartment HH cable solver (M3.5)
// ────────────────────────────────────────────────────────────────────────────

/// Forward-Euler integration of a multi-compartment neuron.
///
/// - Tree topology read from `CompartmentAttr.parent_comp_id`
///   (with `u64::MAX` denoting the soma root).
/// - Axial coupling: `g_couple = 1 / r_axial_c`, applied symmetrically.
/// - Per-compartment leak + (optional) classical Hodgkin-Huxley Na/K
///   currents using the gate fields `m_na`, `h_na`, `m_k` of
///   [`CompartmentState`].
/// - Spike detection: rising-edge crossing of `spike_thresh` at the soma
///   (compartment 0) gated by a 2 ms refractory period. HH membranes
///   reset themselves naturally via inactivation, so no manual reset is
///   applied.
///
/// Returns `true` if the soma generated a fresh spike this tick.
#[allow(clippy::too_many_arguments)]
fn update_multicomp_neuron(
    attr: NeuronAttr,
    tp: Option<&NeuronTypeParams>,
    all_compartment_attrs: &[CompartmentAttr],
    all_comp_states: &mut [CompartmentState],
    neuron_state: &mut NeuronState,
    dt: f32,
    tick: u64,
    spike_thresh: f32,
    ion_defs: &[IonChannelDef],
    ion_channel_sets: &[IonChannelSet],
) -> bool {
    let first = attr.first_comp_id as usize;
    let n = attr.n_compartment as usize;
    let attrs = &all_compartment_attrs[first..first + n];
    let states = &mut all_comp_states[first..first + n];

    // Phase A — initialise per-compartment current accumulator.
    for s in states.iter_mut() {
        s.i_total = s.i_ext;
    }

    // Phase B — axial coupling (each parent/child edge counted once).
    // v2.4 unit fix: diameter/length in μm, r_axial in Ohm·cm.
    // g_int (S) = π * (d*1e-4)² / (4 * Ra * L*1e-4)
    //           = π * d² * 1e-8 / (4 * Ra * L * 1e-4)
    //           = π * d² * 1e-4 / (4 * Ra * L)
    // Convert to nS (×1e9): g_int_nS = π * d² * 1e5 / (4 * Ra * L)
    // Then I_axial (pA) = g_int_nS * (V_parent - V_child)  [nS * mV = pA]
    for c in 0..n {
        let parent_global = attrs[c].parent_comp_id;
        if parent_global == u64::MAX {
            continue;
        }
        let parent_local = parent_global as i64 - first as i64;
        if parent_local < 0 || parent_local as usize >= n {
            continue; // parent in another neuron — not supported
        }
        let p = parent_local as usize;
        let d_um = attrs[c].diameter;
        let l_um = attrs[c].length;
        let ra = attrs[c].r_axial.max(1e-6); // Ohm·cm
        let g_int_ns = std::f32::consts::PI * d_um * d_um * 1e5
                       / (4.0 * ra * l_um.max(1e-6));
        let v_p = states[p].v_mem;
        let v_c = states[c].v_mem;
        let i_axial = g_int_ns * (v_p - v_c); // nS * mV = pA
        states[c].i_total += i_axial;
        states[p].i_total -= i_axial;
    }

    // Phase C — leak + data-driven ion channel currents per compartment.
    for c in 0..n {
        let attr_c = attrs[c];
        let st = &mut states[c];
        let v = st.v_mem;
        let i_leak = -attr_c.g_leak * (v - attr_c.e_leak);
        st.i_total += i_leak;
        // Look up the compartment's ion-channel set (data-driven, not model-tag).
        let set_idx = attr_c.ion_channel_set as usize;
        if set_idx < ion_channel_sets.len() {
            let set = &ion_channel_sets[set_idx];
            hh_update_compartment(st, dt, ion_defs, set);
        }
    }

    // Phase D — integrate V (forward Euler) and clamp.
    for c in 0..n {
        let cm = attrs[c].cm.max(1e-6);
        let st = &mut states[c];
        st.v_mem += st.i_total / cm * dt;
        if st.v_mem > 100.0 { st.v_mem = 100.0; }
        if st.v_mem < -120.0 { st.v_mem = -120.0; }
    }

    // Phase E — somatic spike detection (compartment 0 = soma).
    let v_soma = states[0].v_mem;
    neuron_state.v_mem = v_soma;
    neuron_state.v_mem_soma = v_soma;
    // Sync the per-compartment Ca²⁺ readout up to the neuron-level field.
    neuron_state.cai = states[0].cai;

    let model = tp.map(|t| t.model).unwrap_or(NeuronModel::HodgkinHuxley);
    let is_graded = matches!(model, NeuronModel::Graded);
    let refractory_ticks: u64 = 20; // ~2 ms at dt=0.1 ms
    let in_refractory = match neuron_state.last_spike_tick {
        u64::MAX => false,
        t => tick.saturating_sub(t) < refractory_ticks,
    };

    if is_graded {
        // Graded neurons don't produce classic Na⁺ spikes, but we still
        // emit a "graded spike" event when the soma depolarises past a
        // moderate threshold (e.g. -40 mV). This triggers STP updates
        // on the pre-synaptic side and notifies observers, without a
        // voltage reset. A longer refractory (50 ms) prevents
        // continuous re-emission on sustained depolarisation.
        let graded_thresh = -40.0; // mV — well below classic spike threshold
        let graded_refractory: u64 = 500; // 50 ms at dt=0.1 ms
        let in_graded_refractory = match neuron_state.last_spike_tick {
            u64::MAX => false,
            t => tick.saturating_sub(t) < graded_refractory,
        };
        if !in_graded_refractory && v_soma >= graded_thresh {
            neuron_state.last_spike_tick = tick;
            neuron_state.spike_count = neuron_state.spike_count.saturating_add(1);
            // Graded neurons get a smaller STDP trace (0.5 vs 1.0).
            neuron_state.stdp_trace = neuron_state.stdp_trace.max(0.5);
            return true;
        }
        return false;
    }

    if !in_refractory && v_soma >= spike_thresh {
        neuron_state.last_spike_tick = tick;
        neuron_state.spike_count = neuron_state.spike_count.saturating_add(1);
        neuron_state.stdp_trace = 1.0;
        return true;
    }
    false
}

/// Data-driven Hodgkin–Huxley gate update + ionic currents.
///
/// Gate kinetics (alpha/beta rate functions) are the *mathematical form*
/// of the HH model and stay in code.  **Conductances and reversal
/// potentials** (`g_max`, `e_rev`) are read from the compartment's
/// [`IonChannelSet`] so that different cell types can share the same
/// solver with different parameters.
fn hh_update_compartment(
    st: &mut CompartmentState,
    dt: f32,
    ion_defs: &[IonChannelDef],
    channel_set: &IonChannelSet,
) {
    let v = st.v_mem;

    // Determine which gates we actually need to update from the data set.
    let has_na = channel_set.channels.iter().any(|c| {
        ion_defs.get(c.channel_id as usize)
            .map(|d| d.ion == IonType::Sodium)
            .unwrap_or(false)
    });
    let has_k = channel_set.channels.iter().any(|c| {
        ion_defs.get(c.channel_id as usize)
            .map(|d| d.ion == IonType::Potassium && !d.ca_dependent)
            .unwrap_or(false)
    });
    let has_ca = channel_set.channels.iter().any(|c| {
        ion_defs.get(c.channel_id as usize)
            .map(|d| d.ion == IonType::Calcium)
            .unwrap_or(false)
    });
    let has_kca = channel_set.channels.iter().any(|c| {
        ion_defs.get(c.channel_id as usize)
            .map(|d| d.ion == IonType::Potassium && d.ca_dependent)
            .unwrap_or(false)
    });

    if !has_na && !has_k && !has_ca && !has_kca {
        return; // Pure leak — no HH gates to advance.
    }

    // Gate rate constants (mV, ms⁻¹) — mathematical form of HH kinetics.
    let alpha_m = if (v + 40.0).abs() < 1e-6 {
        1.0
    } else {
        0.1 * (v + 40.0) / (1.0 - (-(v + 40.0) / 10.0).exp())
    };
    let beta_m = 4.0 * (-(v + 65.0) / 18.0).exp();
    let alpha_h = 0.07 * (-(v + 65.0) / 20.0).exp();
    let beta_h = 1.0 / (1.0 + (-(v + 35.0) / 10.0).exp());
    let alpha_n = if (v + 55.0).abs() < 1e-6 {
        0.1
    } else {
        0.01 * (v + 55.0) / (1.0 - (-(v + 55.0) / 10.0).exp())
    };
    let beta_n = 0.125 * (-(v + 65.0) / 80.0).exp();

    if has_na {
        st.m_na += dt * (alpha_m * (1.0 - st.m_na) - beta_m * st.m_na);
        st.h_na += dt * (alpha_h * (1.0 - st.h_na) - beta_h * st.h_na);
        st.m_na = st.m_na.clamp(0.0, 1.0);
        st.h_na = st.h_na.clamp(0.0, 1.0);
    }
    if has_k {
        st.m_k += dt * (alpha_n * (1.0 - st.m_k) - beta_n * st.m_k);
        st.m_k = st.m_k.clamp(0.0, 1.0);
    }
    // CaV gate update — Boltzmann steady-state + exponential approach.
    // Uses simplified Ca²⁺ kinetics: m_inf/h_inf from Boltzmann params
    // stored in the IonChannelDef.gate_vars.
    if has_ca {
        // Default CaV kinetics (adapted from egl19/unc2 L-type):
        // m_ca: v_half = -10 mV, slope = 7 mV
        // h_ca: v_half = -30 mV, slope = -5 mV
        let m_ca_inf = 1.0 / (1.0 + (-(v + 10.0) / 7.0).exp());
        let h_ca_inf = 1.0 / (1.0 + (-(v + 30.0) / (-5.0)).exp());
        let tau_m_ca = 0.5_f32.max(1.0 / ((v + 10.0) / 7.0).exp().max(0.01));
        let tau_h_ca = 10.0_f32.max(1.0 / ((v + 30.0) / (-5.0)).exp().max(0.01));
        st.m_ca += dt * (m_ca_inf - st.m_ca) / tau_m_ca;
        st.h_ca += dt * (h_ca_inf - st.h_ca) / tau_h_ca;
        st.m_ca = st.m_ca.clamp(0.0, 1.0);
        st.h_ca = st.h_ca.clamp(0.0, 1.0);
    }
    // Ca-dependent K (slo1/slo2 BK/SK channels): gate depends on [Ca²⁺]_i.
    // m_kca_inf = cai / (cai + K_d),  K_d ≈ 0.1 μM for SK, ≈ 10 μM for BK.
    if has_kca {
        const KC_KD: f32 = 0.1; // μM — SK channel half-activation
        const KC_TAU: f32 = 1.0; // ms
        let m_kca_inf = st.cai / (st.cai + KC_KD);
        st.m_kca += dt * (m_kca_inf - st.m_kca) / KC_TAU;
        st.m_kca = st.m_kca.clamp(0.0, 1.0);
    }

    // Data-driven current contributions — g_max / e_rev from IonChannelSet.
    // v2.4: Full ion-type coverage (Na, K, Ca, KCa, NonSpecific).
    let mut i_ca_total: f32 = 0.0; // Accumulate Ca current for Ca²⁺ dynamics.
    for ch in &channel_set.channels {
        let Some(def) = ion_defs.get(ch.channel_id as usize) else { continue; };
        let e_rev = ch.e_rev_override.unwrap_or(def.e_rev);
        let g = ch.g_max;
        match def.ion {
            IonType::Sodium => {
                let i_na = -g * st.m_na.powi(3) * st.h_na * (v - e_rev);
                st.i_total += i_na;
            }
            IonType::Potassium => {
                // Ca-dependent K channels (slo1/slo2): gate via m_kca.
                if def.ca_dependent {
                    let i_kca = -g * st.m_kca * (v - e_rev);
                    st.i_total += i_kca;
                } else {
                    let i_k = -g * st.m_k.powi(4) * (v - e_rev);
                    st.i_total += i_k;
                }
            }
            IonType::Calcium => {
                let i_ca = -g * st.m_ca * st.h_ca * (v - e_rev);
                st.i_total += i_ca;
                i_ca_total += i_ca; // Track for Ca²⁺ dynamics.
            }
            IonType::NonSpecific => {
                // Non-specific cation channels (egl36, nca): reuse m_na gate var.
                let i_ns = -g * st.m_na * (v - e_rev);
                st.i_total += i_ns;
            }
            IonType::Chloride => {
                // Chloride channels: no dedicated gate vars yet; treat as
                // ohmic leak with fixed conductance.
                let i_cl = -g * (v - e_rev);
                st.i_total += i_cl;
            }
        }
    }

    // Ca²⁺ dynamics — driven by CaV channel current (design doc §5.4).
    // cai' = -0.0001 * i_ca_total - (cai - cai_rest) / tau_ca
    // where i_ca_total is the sum of all Calcium-channel currents.
    if i_ca_total.abs() > 1e-10 {
        const CAI_REST: f32 = 0.05; // μM
        const CAI_TAU: f32 = 50.0;  // ms
        const CA_SCALE: f32 = -0.0001;
        st.cai += dt * (CA_SCALE * i_ca_total - (st.cai - CAI_REST) / CAI_TAU);
        if st.cai < 0.0 { st.cai = 0.0; }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Internal: split-borrow friendly static slice view
// ────────────────────────────────────────────────────────────────────────────

/// Lightweight view over the mmap'd static segments. Borrows only the `Mmap`
/// (and a copied `Header`) — independent from `BrainDB.neuron_states` etc.,
/// so the borrow checker allows mutating dynamic state in parallel.
#[allow(dead_code)] // some fields are reserved for the M3.5 cable solver
struct StaticView<'a> {
    pub neuron_attrs: &'a [NeuronAttr],
    pub compartment_attrs: &'a [CompartmentAttr],
    pub csr_row_ptr: &'a [u64],
    pub csr_col_idx: &'a [u64],
    pub syn_attrs: &'a [SynapseAttr],
    pub gap_junctions: &'a [GapJunction],
    pub receptors: &'a [ReceptorParams],
}

impl<'a> StaticView<'a> {
    fn new(mmap: &'a Mmap, header: &Header) -> Self {
        Self {
            neuron_attrs: cast_segment(mmap, header, off::NEURON_ATTR,
                                        header.n_neurons as usize),
            compartment_attrs: cast_segment(mmap, header, off::COMPARTMENT_ATTR,
                                             header.n_compartments as usize),
            csr_row_ptr: cast_segment(mmap, header, off::CSR_ROW_PTR,
                                       header.n_neurons as usize + 1),
            csr_col_idx: cast_segment(mmap, header, off::CSR_COL_IDX,
                                       header.n_synapses as usize),
            syn_attrs: cast_segment(mmap, header, off::SYNAPSE_ATTR,
                                     header.n_synapses as usize),
            gap_junctions: cast_segment(mmap, header, off::GAP,
                                         header.n_gap_junctions as usize),
            receptors: cast_segment(mmap, header, off::RECEPTOR,
                                     header.n_receptor_types as usize),
        }
    }

    fn out_range(&self, pre: usize) -> std::ops::Range<usize> {
        self.csr_row_ptr[pre] as usize..self.csr_row_ptr[pre + 1] as usize
    }
}

fn cast_segment<'a, T: Pod>(mmap: &'a Mmap, header: &Header, slot: usize, n: usize) -> &'a [T] {
    let off = header.offsets[slot] as usize;
    let len = n * std::mem::size_of::<T>();
    if n == 0 {
        return &[];
    }
    bytemuck::cast_slice(&mmap[off..off + len])
}

