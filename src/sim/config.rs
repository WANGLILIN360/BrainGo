//! Configuration knobs for the simulation loop.

#[derive(Clone, Debug)]
pub struct SimulationConfig {
    /// Membrane potential at which an Izhikevich / HH point neuron is
    /// considered to have spiked.
    pub spike_threshold_mv: f32,

    /// Maximum allowed synaptic weight (clamp after STDP updates).
    pub max_syn_weight: f32,

    /// Conductance magnitude below which a chemical synapse is considered
    /// inactive and removed from the active list.
    pub min_active_conductance: f32,

    /// STDP enable flag.
    pub stdp_enabled: bool,

    /// LTP magnitude `A_plus` (Song 2000).
    pub stdp_a_plus: f32,
    /// LTD magnitude `A_minus`.
    pub stdp_a_minus: f32,
    /// STDP trace decay time constant (ms).
    pub stdp_tau: f32,

    /// Apply accumulated STDP weight changes every `stdp_apply_every` ticks.
    pub stdp_apply_every: u64,

    /// Enable structural plasticity (Phase 7).
    pub structural_plasticity_enabled: bool,

    /// Enable neuromodulator diffusion (Phase 6).
    pub modulation_enabled: bool,

    /// Neuromodulator diffusion rate (1/ms) — controls how fast
    /// modulators spread between adjacent regions.
    pub modulation_diffusion_rate: f32,

    /// Neuromodulator baseline decay rate (1/ms).
    pub modulation_decay_rate: f32,

    /// Adaptation current time constant (ms) used by point neurons.
    pub adapt_w_rate: f32,

    /// Calcium decay time constant for point neurons (ms).
    pub cai_tau: f32,

    /// Enable rayon parallel neuron updates (Phase 4).
    /// Only beneficial for large networks (>10k neurons).
    pub parallel_neuron_update: bool,

    // ── Structural plasticity (Phase 7) ────────────────────────────────
    /// Activity window (ticks) for co-firing detection in structural plasticity.
    pub sp_window: u64,
    /// Initial weight for newly sprouted synapses (nS).
    pub sp_init_weight: f32,
    /// Synapses with |weight| below this threshold are pruned.
    pub sp_prune_threshold: f32,
    /// Maximum outgoing synapses per neuron (base + dynamic).
    pub sp_max_out_degree: usize,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            spike_threshold_mv: 30.0,
            max_syn_weight: 10.0,
            min_active_conductance: 1e-6,
            stdp_enabled: false,
            stdp_a_plus: 0.005,
            stdp_a_minus: 0.005,
            stdp_tau: 20.0,
            stdp_apply_every: 1000, // every 100 ms at dt=0.1 ms
            structural_plasticity_enabled: false,
            modulation_enabled: false,
            modulation_diffusion_rate: 0.001,
            modulation_decay_rate: 0.0001,
            adapt_w_rate: 0.001,
            cai_tau: 50.0,
            parallel_neuron_update: false,
            sp_window: 5000,          // 500 ms
            sp_init_weight: 0.1,      // nS
            sp_prune_threshold: 0.01, // nS
            sp_max_out_degree: 200,
        }
    }
}
