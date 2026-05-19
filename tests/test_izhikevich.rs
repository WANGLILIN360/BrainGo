//! Izhikevich point-neuron model tests: RS, FS, IB parameter sets.

use braindb::*;
use tempfile::tempdir;

fn build_izh_neuron(iz: IzhikevichParams) -> Simulation {
    let dir = tempdir().unwrap();
    let path = dir.path().join("iz.braindb");
    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_region(BrainRegion { id: 0, neuron_count: 1, ..Default::default() }, "r");
    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "IZ".into(),
        category: NeuronCategory::Interneuron,
        model: NeuronModel::Izhikevich,
        iz_params: iz,
        ..Default::default()
    });
    b.add_neuron(NeuronAttr {
        id: 0, neuron_type: nt, n_compartment: 0,
        cm: 100.0, g_leak: 10.0, e_leak: -70.0,
        ..Default::default()
    });
    let db = b.build(&path).unwrap();
    Simulation::new(db)
}

#[test]
fn regular_spiking_fires_under_sustained_input() {
    let mut sim = build_izh_neuron(IzhikevichParams::regular_spiking());
    sim.present_stimulus(&[(0, 20.0)], 10_000);
    sim.run(2_000);
    let count = sim.db.neuron_states[0].spike_count;
    assert!(count > 3, "RS neuron should spike repeatedly, got {count}");
}

#[test]
fn fast_spiking_has_higher_rate() {
    let mut sim_rs = build_izh_neuron(IzhikevichParams::regular_spiking());
    let mut sim_fs = build_izh_neuron(IzhikevichParams::fast_spiking());
    sim_rs.present_stimulus(&[(0, 20.0)], 10_000);
    sim_fs.present_stimulus(&[(0, 20.0)], 10_000);
    sim_rs.run(3_000);
    sim_fs.run(3_000);
    let rs_count = sim_rs.db.neuron_states[0].spike_count;
    let fs_count = sim_fs.db.neuron_states[0].spike_count;
    assert!(fs_count > rs_count,
        "FS ({fs_count}) should fire more than RS ({rs_count}) at same input");
}

#[test]
fn intrinsically_bursting_bursts_without_input() {
    let mut sim = build_izh_neuron(IzhikevichParams::intrinsically_bursting());
    // IB neurons should burst even without external input (or with very small input).
    sim.db.neuron_states[0].i_ext = 5.0;
    sim.run(5_000);
    let count = sim.db.neuron_states[0].spike_count;
    assert!(count > 0, "IB neuron should burst spontaneously, got {count}");
}

#[test]
fn izhikevich_v_mem_resets_on_spike() {
    let mut sim = build_izh_neuron(IzhikevichParams::regular_spiking());
    sim.present_stimulus(&[(0, 30.0)], 10_000);
    sim.run(100);
    // After spiking, v_mem should be near c = -65 mV.
    let v = sim.db.neuron_states[0].v_mem;
    assert!(v < -50.0, "v_mem should be near reset after spike, got {v}");
}
