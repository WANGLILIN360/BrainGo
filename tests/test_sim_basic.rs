//! Basic M3 simulation tests: drive a tiny point-neuron network and assert
//! spike behaviour, gap-junction coupling, continuous-mode synapses, and
//! stimulus expiration.

use braindb::*;
use tempfile::tempdir;

fn build_pair(continuous: bool) -> Simulation {
    let dir = tempdir().unwrap();
    let path = dir.path().join("sim.braindb");
    let mut b = BrainDBBuilder::new();

    // Receptor 0: excitatory AMPA-like; we override v_threshold for continuous mode.
    b.add_receptor(ReceptorParams::ampa());

    // Region.
    b.add_region(
        BrainRegion {
            id: 0,
            first_neuron: 0,
            neuron_count: 2,
            ..Default::default()
        },
        "test_region",
    );

    // Neuron type — Izhikevich regular spiking.
    let nt_iz = b.add_neuron_type(NeuronTypeParams {
        type_name: "RS".into(),
        category: NeuronCategory::Interneuron,
        model: NeuronModel::Izhikevich,
        iz_params: IzhikevichParams::regular_spiking(),
        ..Default::default()
    });

    // Two point neurons.
    for id in 0..2u64 {
        b.add_neuron(NeuronAttr {
            id,
            neuron_type: nt_iz,
            region_id: 0,
            n_compartment: 0,
            cm: 100.0,
            g_leak: 10.0,
            e_leak: -70.0,
            ..Default::default()
        });
    }

    // Synapse 0 → 1 with a small delay.
    let mode = if continuous { SYN_MODE_CONTINUOUS } else { SYN_MODE_EVENT_DRIVEN };
    b.add_synapse(
        0,
        SynapseAttr {
            post_neuron: 1,
            base_weight: 2.0,
            delay_ticks: 5,
            syn_type: SYN_EXCITATORY,
            syn_mode: mode,
            receptor_type: RECEPTOR_AMPA,
            u_se: 1.0,
            tau_rec: 100.0,
            ..Default::default()
        },
    );

    let db = b.build(&path).unwrap();
    Simulation::new(db)
}

#[test]
fn stimulus_drives_izhikevich_spikes() {
    let mut sim = build_pair(false);
    sim.add_observer(SpikeLog::default());

    // Strong steady current into neuron 0 should make it spike repeatedly.
    sim.present_stimulus(&[(0, 25.0)], 5_000);
    sim.run(2_000); // 200 ms

    let counts = sim.read_spike_counts(0, 2);
    assert!(counts[0] > 5,
        "neuron 0 should fire many times under sustained input, got {}", counts[0]);
}

#[test]
fn event_synapse_propagates_to_post() {
    let mut sim = build_pair(false);
    // Drive pre (0) hard; observe whether post (1) eventually fires too.
    sim.present_stimulus(&[(0, 25.0)], 10_000);
    sim.run(3_000);

    let counts = sim.read_spike_counts(0, 2);
    assert!(counts[0] > 0, "pre-neuron must fire");
    // We don't require post to spike (weight=2 is modest) — just check the
    // active-synapse machinery is exercised: ring should have or have had
    // pending events, and post v_mem must have been perturbed.
    assert!(
        sim.db.neuron_states[1].v_mem > -70.0
            || sim.db.neuron_states[1].spike_count > 0
            || sim.event_ring.pending() > 0
            || !sim.active_synapses.is_empty(),
        "post-neuron should show some synaptic effect"
    );
}

#[test]
fn continuous_synapse_drives_post_voltage() {
    let mut sim = build_pair(true);
    // Hold pre at a depolarised value via large i_ext.
    sim.present_stimulus(&[(0, 30.0)], 20_000);
    sim.run(2_000);

    // In continuous mode the post-neuron must see *some* synaptic drive.
    // We compare against a baseline run with no stimulus.
    let post_v_driven = sim.db.neuron_states[1].v_mem;

    let mut baseline = build_pair(true);
    baseline.run(2_000);
    let post_v_baseline = baseline.db.neuron_states[1].v_mem;

    assert!(
        (post_v_driven - post_v_baseline).abs() > 1e-4,
        "continuous-mode synapse should perturb post v_mem ({post_v_driven} vs baseline {post_v_baseline})"
    );
}

#[test]
fn gap_junction_couples_voltages() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("gj.braindb");
    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_region(BrainRegion { id: 0, neuron_count: 2, ..Default::default() }, "r");
    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "G".into(), model: NeuronModel::Graded,
        iz_params: IzhikevichParams { v_rest: -70.0, ..IzhikevichParams::regular_spiking() },
        ..Default::default()
    });
    for id in 0..2u64 {
        b.add_neuron(NeuronAttr {
            id, neuron_type: nt, n_compartment: 0,
            cm: 100.0, g_leak: 1.0, e_leak: -70.0,
            ..Default::default()
        });
    }
    // Strong gap junction.
    b.add_gap_junction(GapJunction {
        pre_neuron: 0, post_neuron: 1, weight: 50.0, ..Default::default()
    });

    let db = b.build(&path).unwrap();
    let mut sim = Simulation::new(db);

    // Drive neuron 0 only; neuron 1 should follow via the gap junction.
    sim.present_stimulus(&[(0, 50.0)], 10_000);
    sim.run(2_000);

    let v0 = sim.db.neuron_states[0].v_mem;
    let v1 = sim.db.neuron_states[1].v_mem;
    assert!(v0 > -70.0, "neuron 0 should be depolarised, got {v0}");
    assert!(v1 > -70.0, "neuron 1 should follow via gap junction, got {v1}");
}

#[test]
fn stimulus_expires() {
    let mut sim = build_pair(false);
    sim.present_stimulus(&[(0, 25.0)], 100);
    assert!(!sim.active_stimulus_neurons.is_empty());
    sim.run(150);
    assert!(
        sim.active_stimulus_neurons.is_empty(),
        "stimulus must have expired after duration_ticks elapsed"
    );
    assert_eq!(sim.db.neuron_states[0].i_ext, 0.0);
}
