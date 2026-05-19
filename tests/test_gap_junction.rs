//! Gap junction tests: coupling, kill_neuron severing, bidirectional current.

use braindb::*;
use tempfile::tempdir;

fn build_gap_pair(weight: f32) -> Simulation {
    let dir = tempdir().unwrap();
    let path = dir.path().join("gap.braindb");
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
    b.add_gap_junction(GapJunction {
        pre_neuron: 0, post_neuron: 1, weight, ..Default::default()
    });
    let db = b.build(&path).unwrap();
    Simulation::new(db)
}

#[test]
fn gap_junction_equalises_voltages() {
    let mut sim = build_gap_pair(100.0);
    // Drive neuron 0 only.
    sim.present_stimulus(&[(0, 50.0)], 20_000);
    sim.run(5_000);

    let v0 = sim.db.neuron_states[0].v_mem;
    let v1 = sim.db.neuron_states[1].v_mem;
    // With strong coupling, voltages should converge.
    assert!((v0 - v1).abs() < 5.0, "voltages should converge: v0={v0}, v1={v1}");
}

#[test]
fn kill_neuron_severs_gap_junction() {
    let mut sim = build_gap_pair(100.0);
    sim.kill_neuron(0);
    sim.present_stimulus(&[(1, 50.0)], 20_000);
    sim.run(5_000);

    // Neuron 0 is dead — its gap junction weight should be zeroed.
    assert_eq!(sim.gap_junction_weights[0], 0.0, "gap junction weight should be zeroed after kill");
    // Dead neuron's v_mem should not change.
    let v0 = sim.db.neuron_states[0].v_mem;
    assert!((v0 - (-70.0)).abs() < 5.0, "dead neuron v_mem should stay near rest, got {v0}");
}

#[test]
fn weak_gap_junction_minimal_coupling() {
    let mut sim = build_gap_pair(0.01);
    sim.present_stimulus(&[(0, 50.0)], 20_000);
    sim.run(5_000);

    let v0 = sim.db.neuron_states[0].v_mem;
    let v1 = sim.db.neuron_states[1].v_mem;
    // With very weak coupling, v1 should be much less depolarised than v0.
    assert!((v0 - v1).abs() > 5.0, "weak gap should not equalise: v0={v0}, v1={v1}");
}

#[test]
fn gap_junction_bidirectional_current() {
    let mut sim = build_gap_pair(50.0);
    // Drive neuron 1 (post side) — current should flow back to neuron 0.
    sim.present_stimulus(&[(1, 50.0)], 20_000);
    sim.run(5_000);

    let v0 = sim.db.neuron_states[0].v_mem;
    assert!(v0 > -70.0, "neuron 0 should be depolarised via reverse gap current, got {v0}");
}
