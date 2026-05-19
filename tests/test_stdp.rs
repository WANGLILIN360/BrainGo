//! Tests for STDP weight write-back (M5 partial).

use braindb::*;
use tempfile::tempdir;

fn build_pair() -> Simulation {
    let dir = tempdir().unwrap();
    let path = dir.path().join("stdp.braindb");
    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_region(BrainRegion { id: 0, neuron_count: 2, ..Default::default() }, "r");
    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "RS".into(),
        category: NeuronCategory::Interneuron,
        model: NeuronModel::Izhikevich,
        iz_params: IzhikevichParams::regular_spiking(),
        ..Default::default()
    });
    for id in 0..2u64 {
        b.add_neuron(NeuronAttr {
            id, neuron_type: nt, n_compartment: 0,
            cm: 100.0, g_leak: 10.0, e_leak: -70.0,
            ..Default::default()
        });
    }
    b.add_synapse(0, SynapseAttr {
        post_neuron: 1,
        base_weight: 1.0,
        delay_ticks: 5,
        syn_type: SYN_EXCITATORY,
        syn_mode: SYN_MODE_EVENT_DRIVEN,
        receptor_type: RECEPTOR_AMPA,
        u_se: 1.0,
        tau_rec: 100.0,
        ..Default::default()
    });
    let db = b.build(&path).unwrap();
    Simulation::new(db)
}

#[test]
fn dw_accum_applied_each_tick() {
    let mut sim = build_pair();
    sim.config.stdp_enabled = true;
    sim.config.stdp_apply_every = 1;

    let orig_w = sim.db.syn_states[0].weight;
    sim.db.syn_states[0].dw_accum = 0.3;
    sim.step();

    assert!(
        (sim.db.syn_states[0].weight - (orig_w + 0.3)).abs() < 1e-5,
        "weight should have grown by dw_accum (was {orig_w}, now {})",
        sim.db.syn_states[0].weight
    );
    assert_eq!(sim.db.syn_states[0].dw_accum, 0.0);
}

#[test]
fn stdp_clamps_to_max_and_min() {
    let mut sim = build_pair();
    sim.config.stdp_enabled = true;
    sim.config.stdp_apply_every = 1;
    sim.config.max_syn_weight = 2.0;

    // Push weight past the upper bound.
    sim.db.syn_states[0].dw_accum = 100.0;
    sim.step();
    assert_eq!(sim.db.syn_states[0].weight, 2.0);

    // Push weight below zero.
    sim.db.syn_states[0].dw_accum = -100.0;
    sim.step();
    assert_eq!(sim.db.syn_states[0].weight, 0.0);
}

#[test]
fn pre_post_spikes_change_weight() {
    let mut sim = build_pair();
    sim.config.stdp_enabled = true;
    sim.config.stdp_a_plus = 0.05;
    sim.config.stdp_a_minus = 0.05;
    sim.config.stdp_apply_every = 100; // ~10 ms

    let orig_w = sim.db.syn_states[0].weight;
    // Drive the two neurons with different currents so their firing rates
    // differ — this guarantees the STDP traces seen by pre/post differ.
    sim.present_stimulus(&[(0, 35.0), (1, 22.0)], 5_000);
    sim.run(3_000); // 300 ms

    let final_w = sim.db.syn_states[0].weight;
    assert!(
        (final_w - orig_w).abs() > 1e-4,
        "weight should drift under STDP (orig {orig_w}, final {final_w})"
    );
    assert!(final_w >= 0.0 && final_w <= sim.config.max_syn_weight);
}

#[test]
fn stdp_disabled_keeps_weight() {
    let mut sim = build_pair();
    assert!(!sim.config.stdp_enabled);
    let orig_w = sim.db.syn_states[0].weight;

    sim.present_stimulus(&[(0, 30.0), (1, 30.0)], 5_000);
    sim.run(3_000);

    assert_eq!(sim.db.syn_states[0].weight, orig_w);
}
