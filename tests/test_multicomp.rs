//! Multi-compartment HH cable solver (M3.5) integration tests.

use braindb::*;
use tempfile::tempdir;

/// Build a neuron with `n_comp` compartments arranged as a chain
/// (0 = soma, 1 → 0, 2 → 1, …). Returns the loaded simulation.
fn build_chain(n_comp: u32, model: NeuronModel) -> Simulation {
    let dir = tempdir().unwrap();
    let path = dir.path().join("multicomp.braindb");
    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_region(BrainRegion { id: 0, neuron_count: 1, ..Default::default() }, "r");

    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "HH".into(),
        category: NeuronCategory::Interneuron,
        model,
        iz_params: IzhikevichParams { v_rest: -65.0, ..IzhikevichParams::regular_spiking() },
        ..Default::default()
    });

    // Single neuron with `n_comp` compartments. The compartment indices are
    // 0..n_comp (compartments are added in order, so the global ID equals
    // the order of insertion when there are no other neurons).
    b.add_neuron(NeuronAttr {
        id: 0,
        neuron_type: nt,
        region_id: 0,
        first_comp_id: 0,
        n_compartment: n_comp,
        cm: 1.0,
        g_leak: 0.3,
        e_leak: -65.0,
        ..Default::default()
    });

    for c in 0..n_comp {
        b.add_compartment(CompartmentAttr {
            id: c as u64,
            neuron_id: 0,
            parent_comp_id: if c == 0 { u64::MAX } else { (c - 1) as u64 },
            comp_type: if c == 0 {
                CompType::Soma as u8
            } else {
                CompType::BasalDend as u8
            },
            ion_channel_set: u32::MAX,
            length: 10.0,
            diameter: 1.0,
            cm: 10.0,           // 10 pF — large enough for stability
            r_axial: 150.0,     // 150 Ohm·cm — typical axoplasmic resistivity
            x: c as f32 * 10.0,
            y: 0.0,
            z: 0.0,
            g_leak: 0.3,
            e_leak: -65.0,
            ..Default::default()
        });
    }

    let db = b.build(&path).unwrap();
    Simulation::new(db)
}

#[test]
fn multicomp_passive_chain_propagates_voltage() {
    // Graded model — no HH, just leak + axial. Inject current at distal
    // compartment and watch the soma voltage rise.
    let mut sim = build_chain(3, NeuronModel::Graded);

    // Set i_ext on the distal compartment (index 2).
    // Use 5.0 pA — enough to depolarise but stays below graded spike threshold.
    sim.db.comp_states[2].i_ext = 5.0;

    sim.run(2_000); // 200 ms

    let v_distal = sim.db.comp_states[2].v_mem;
    let v_mid    = sim.db.comp_states[1].v_mem;
    let v_soma   = sim.db.comp_states[0].v_mem;

    // Voltage should fall off from injection site to soma but the soma must
    // be depolarised compared to its resting potential of -65 mV.
    assert!(v_distal > v_mid, "distal {v_distal} should exceed mid {v_mid}");
    assert!(v_mid > v_soma,   "mid {v_mid} should exceed soma {v_soma}");
    assert!(v_soma > -65.0,   "soma {v_soma} should be depolarised");

    // The neuron-level v_mem should mirror the soma compartment.
    assert!((sim.db.neuron_states[0].v_mem - v_soma).abs() < 1e-3);
    assert!((sim.db.neuron_states[0].v_mem_soma - v_soma).abs() < 1e-3);
}

#[test]
fn multicomp_skipped_when_malformed() {
    // Build a neuron declaring 4 compartments but only register 2; the
    // engine must skip it without panicking.
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad.braindb");
    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_region(BrainRegion { id: 0, neuron_count: 1, ..Default::default() }, "r");
    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "X".into(), model: NeuronModel::Graded, ..Default::default()
    });
    b.add_neuron(NeuronAttr {
        id: 0, neuron_type: nt, first_comp_id: 0, n_compartment: 4,
        cm: 1.0, g_leak: 0.3, e_leak: -65.0,
        ..Default::default()
    });
    // Only add 2 compartments, leaving a 2-compartment shortfall.
    for c in 0..2u64 {
        b.add_compartment(CompartmentAttr {
            id: c, neuron_id: 0,
            parent_comp_id: if c == 0 { u64::MAX } else { c - 1 },
            ..Default::default()
        });
    }
    let db = b.build(&path).unwrap();
    let mut sim = Simulation::new(db);

    sim.run(10);
    assert!(sim.skipped_multicomp >= 10);
}
