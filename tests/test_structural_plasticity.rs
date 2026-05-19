//! Structural plasticity integration tests (Phase 7).
//!
//! Tests that structural plasticity correctly sprouts new synapses
//! between co-active neurons and prunes weak ones.

use braindb::*;
use tempfile::tempdir;

fn build_sp_network() -> (tempfile::TempDir, Simulation) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("sp.braindb");
    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_region(
        BrainRegion {
            id: 0,
            first_neuron: 0,
            neuron_count: 4,
            ..Default::default()
        },
        "r0",
    );
    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "RS".into(),
        model: NeuronModel::Izhikevich,
        iz_params: IzhikevichParams::regular_spiking(),
        ..Default::default()
    });
    for id in 0..4u64 {
        b.add_neuron(NeuronAttr {
            id,
            neuron_type: nt,
            region_id: 0,
            cm: 100.0,
            g_leak: 10.0,
            e_leak: -70.0,
            ..Default::default()
        });
    }
    // Only one synapse: 0→1 with weight 0.005 (below prune threshold).
    b.add_synapse(0, SynapseAttr {
        post_neuron: 1,
        base_weight: 0.005,
        delay_ticks: 1,
        syn_type: SYN_EXCITATORY,
        syn_mode: SYN_MODE_EVENT_DRIVEN,
        receptor_type: RECEPTOR_AMPA,
        ..Default::default()
    });

    let db = b.build(&path).unwrap();
    let mut sim = Simulation::new(db);
    sim.config.structural_plasticity_enabled = true;
    sim.config.sp_window = 5000;       // 500 ms
    sim.config.sp_init_weight = 0.5;   // nS
    sim.config.sp_prune_threshold = 0.01; // prune below 0.01 nS
    sim.config.sp_max_out_degree = 200;
    (dir, sim)
}

#[test]
fn structural_plasticity_prunes_weak_synapse() {
    let (_dir, mut sim) = build_sp_network();

    // Run past the first structural plasticity window (tick 10000).
    // The synapse 0→1 has weight 0.005 < prune_threshold 0.01, so it should be pruned.
    sim.run(10_001);

    // The synapse should be marked deleted in the DynamicCSR.
    assert!(sim.dynamic_csr.is_some(), "DynamicCSR should be initialised after SP");
    let dcsr = sim.dynamic_csr.as_ref().unwrap();
    assert!(dcsr.is_deleted(0), "synapse 0 (weight 0.005) should be pruned");
}

#[test]
fn structural_plasticity_sprouts_between_coactive() {
    let (_dir, mut sim) = build_sp_network();

    // Drive neurons 2 and 3 with sustained input so they fire repeatedly
    // and are still co-active when structural plasticity triggers at tick 10000.
    sim.db.neuron_states[2].i_ext = 30.0;
    sim.db.neuron_states[3].i_ext = 30.0;
    sim.run(10_001); // past first SP trigger

    // DynamicCSR should have been initialised and a new synapse sprouted.
    assert!(sim.dynamic_csr.is_some(), "DynamicCSR should be initialised after SP");
    let dcsr = sim.dynamic_csr.as_ref().unwrap();

    // Check that a new synapse was inserted (either 2→3 or 3→2).
    let delta_2: Vec<_> = dcsr.delta_out_synapses(2).collect();
    let delta_3: Vec<_> = dcsr.delta_out_synapses(3).collect();
    let sprouted = delta_2.len() + delta_3.len();
    assert!(sprouted > 0, "at least one synapse should sprout between co-active neurons 2↔3, got delta_2={} delta_3={}",
        delta_2.len(), delta_3.len());
}

#[test]
fn structural_plasticity_respects_out_degree_cap() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("cap.braindb");
    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_region(
        BrainRegion {
            id: 0,
            first_neuron: 0,
            neuron_count: 5,
            ..Default::default()
        },
        "r0",
    );
    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "RS".into(),
        model: NeuronModel::Izhikevich,
        iz_params: IzhikevichParams::regular_spiking(),
        ..Default::default()
    });
    for id in 0..5u64 {
        b.add_neuron(NeuronAttr {
            id,
            neuron_type: nt,
            region_id: 0,
            cm: 100.0,
            g_leak: 10.0,
            e_leak: -70.0,
            ..Default::default()
        });
    }
    // Neuron 0 already has 3 outgoing synapses (to 1, 2, 3).
    for post in 1..4u32 {
        b.add_synapse(0, SynapseAttr {
            post_neuron: post,
            base_weight: 1.0,
            delay_ticks: 1,
            syn_type: SYN_EXCITATORY,
            syn_mode: SYN_MODE_EVENT_DRIVEN,
            receptor_type: RECEPTOR_AMPA,
            ..Default::default()
        });
    }

    let db = b.build(&path).unwrap();
    let mut sim = Simulation::new(db);
    sim.config.structural_plasticity_enabled = true;
    sim.config.sp_window = 5000;
    sim.config.sp_init_weight = 0.5;
    sim.config.sp_prune_threshold = 0.0; // don't prune
    sim.config.sp_max_out_degree = 3;    // neuron 0 already at cap

    // Drive neurons 0 and 4 with sustained input (co-active, same region).
    sim.db.neuron_states[0].i_ext = 30.0;
    sim.db.neuron_states[4].i_ext = 30.0;
    sim.run(10_001);

    // Neuron 0 should NOT sprout a new synapse to 4 because it's at the cap.
    let dcsr = sim.dynamic_csr.as_ref().unwrap();
    let delta_0: Vec<_> = dcsr.delta_out_synapses(0).collect();
    assert!(delta_0.is_empty(), "neuron 0 should not sprout (at out-degree cap), got {} delta synapses",
        delta_0.len());
}
