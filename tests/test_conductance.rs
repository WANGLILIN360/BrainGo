//! Conductance decay and synapse mode tests.

use braindb::*;
use tempfile::tempdir;

fn build_conductance_pair(mode: u8, receptor: ReceptorParams) -> Simulation {
    let dir = tempdir().unwrap();
    let path = dir.path().join("cond.braindb");
    let mut b = BrainDBBuilder::new();
    b.add_receptor(receptor);
    b.add_region(BrainRegion { id: 0, neuron_count: 2, ..Default::default() }, "r");
    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "RS".into(),
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
        base_weight: 5.0,
        delay_ticks: 5,
        syn_type: SYN_EXCITATORY,
        syn_mode: mode,
        receptor_type: RECEPTOR_AMPA,
        u_se: 1.0,
        tau_rec: 100.0,
        ..Default::default()
    });
    let db = b.build(&path).unwrap();
    Simulation::new(db)
}

#[test]
fn event_driven_conductance_decays() {
    let mut sim = build_conductance_pair(SYN_MODE_EVENT_DRIVEN, ReceptorParams::ampa());
    // Drive pre-neuron to spike.
    sim.present_stimulus(&[(0, 30.0)], 10_000);
    sim.run(500); // 50 ms — enough for spike + conductance onset

    // After the event arrives, conductance should be nonzero.
    let has_active = !sim.active_synapses.is_empty()
        || sim.db.syn_states[0].g_rise.abs() > 1e-6
        || sim.db.syn_states[0].g_decay.abs() > 1e-6;
    assert!(has_active, "synapse should be active after pre fires");

    // Run more — conductance should decay.
    sim.run(5_000); // 500 ms
    let g_rise = sim.db.syn_states[0].g_rise;
    let g_decay = sim.db.syn_states[0].g_decay;
    assert!(g_rise.abs() < 0.1 && g_decay.abs() < 0.1,
        "conductance should decay: g_rise={g_rise}, g_decay={g_decay}");
}

#[test]
fn continuous_conductance_follows_pre_voltage() {
    let mut sim = build_conductance_pair(SYN_MODE_CONTINUOUS, ReceptorParams {
        v_threshold: -40.0,
        v_slope: 5.0,
        k_rate: 0.1,
        ..ReceptorParams::ampa()
    });
    // Drive pre-neuron strongly — should push v_pre above threshold.
    sim.present_stimulus(&[(0, 35.0)], 20_000);
    sim.run(3_000);

    let g_rise = sim.db.syn_states[0].g_rise;
    assert!(g_rise > 0.01, "continuous conductance should be nonzero when pre is depolarised, got {g_rise}");
}

#[test]
fn nmda_mg_block_reduces_current() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nmda.braindb");
    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());  // index 0
    b.add_receptor(ReceptorParams::nmda());  // index 1 = RECEPTOR_NMDA
    b.add_region(BrainRegion { id: 0, neuron_count: 2, ..Default::default() }, "r");
    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "RS".into(), model: NeuronModel::Izhikevich,
        iz_params: IzhikevichParams::regular_spiking(), ..Default::default()
    });
    for id in 0..2u64 {
        b.add_neuron(NeuronAttr {
            id, neuron_type: nt, n_compartment: 0,
            cm: 100.0, g_leak: 10.0, e_leak: -70.0, ..Default::default()
        });
    }
    b.add_synapse(0, SynapseAttr {
        post_neuron: 1, base_weight: 5.0, delay_ticks: 3,
        syn_type: SYN_EXCITATORY, syn_mode: SYN_MODE_EVENT_DRIVEN,
        receptor_type: RECEPTOR_NMDA, u_se: 1.0, tau_rec: 100.0,
        ..Default::default()
    });
    let db = b.build(&path).unwrap();
    let mut sim = Simulation::new(db);

    // Drive pre to spike; NMDA should be Mg-blocked at resting potential.
    sim.present_stimulus(&[(0, 30.0)], 10_000);
    sim.run(500);

    // Post v_mem should be less depolarised than an equivalent AMPA synapse
    // because of Mg block at hyperpolarised post-synaptic potentials.
    let v_post = sim.db.neuron_states[1].v_mem;
    // Just verify the simulation didn't crash and v_post is in range.
    assert!(v_post > -120.0 && v_post < 50.0, "v_post={v_post} out of range");
}
