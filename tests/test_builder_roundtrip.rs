//! Build a tiny BrainDB, round-trip it through the file, verify CSR + snapshot.

use braindb::core::naming::name_hash;
use braindb::*;
use std::path::PathBuf;
use tempfile::tempdir;

/// A minimal 4-neuron / 5-synapse / 1-gap test fixture.
fn make_tiny(builder: &mut BrainDBBuilder) {
    // 1 receptor type, 1 region.
    let _r_ampa = builder.add_receptor(ReceptorParams::ampa());
    let region = BrainRegion {
        id: 0,
        name_hash: name_hash("test_region"),
        first_neuron: 0,
        neuron_count: 4,
        ..Default::default()
    };
    builder.add_region(region, "test_region");

    // Neuron type.
    let nt = NeuronTypeParams {
        type_name: "RS".to_string(),
        category: NeuronCategory::Interneuron,
        model: NeuronModel::Izhikevich,
        iz_params: IzhikevichParams::regular_spiking(),
        ..Default::default()
    };
    let nt_id = builder.add_neuron_type(nt);

    // 4 point neurons.
    for id in 0..4u64 {
        let mut a = NeuronAttr {
            id,
            neuron_type: nt_id,
            region_id: 0,
            n_compartment: 0,
            ..Default::default()
        };
        a.x = id as f32;
        builder.add_neuron(a);
    }

    // Synapses, deliberately added out of order to exercise CSR sorting:
    //   2 → 3 (w=0.7)
    //   0 → 1 (w=0.5)
    //   1 → 2 (w=0.6)
    //   0 → 2 (w=0.3)
    //   3 → 0 (w=0.9)
    let mk = |post, w| SynapseAttr { post_neuron: post, base_weight: w, ..Default::default() };
    builder.add_synapse(2, mk(3, 0.7));
    builder.add_synapse(0, mk(1, 0.5));
    builder.add_synapse(1, mk(2, 0.6));
    builder.add_synapse(0, mk(2, 0.3));
    builder.add_synapse(3, mk(0, 0.9));

    builder.add_gap_junction(GapJunction {
        pre_neuron: 0,
        post_neuron: 1,
        weight: 0.05,
        ..Default::default()
    });
}

#[test]
fn build_and_open_roundtrip() {
    let dir = tempdir().unwrap();
    let path: PathBuf = dir.path().join("tiny.braindb");

    let mut b = BrainDBBuilder::new();
    make_tiny(&mut b);
    let db = b.build(&path).expect("build .braindb");

    // Header / counts.
    assert_eq!(db.header.n_neurons, 4);
    assert_eq!(db.header.n_synapses, 5);
    assert_eq!(db.header.n_gap_junctions, 1);
    assert_eq!(db.header.n_regions, 1);
    assert_eq!(db.header.n_receptor_types, 1);
    assert_eq!(db.header.n_neuron_types, 1);
    assert_eq!(db.header.dt, 0.1);

    // Neuron attrs round-tripped.
    assert_eq!(db.neuron_attrs().len(), 4);
    for (i, a) in db.neuron_attrs().iter().enumerate() {
        assert_eq!(a.id, i as u64);
        assert_eq!(a.x, i as f32);
    }

    // CSR row_ptr length n_neurons+1, monotonically non-decreasing,
    // last = n_synapses.
    let rp = db.csr_row_ptr();
    assert_eq!(rp.len(), 5);
    assert_eq!(rp[0], 0);
    assert_eq!(*rp.last().unwrap(), 5);
    for w in rp.windows(2) {
        assert!(w[0] <= w[1]);
    }

    // Outgoing degrees: neuron 0 has 2, neuron 1 has 1, neuron 2 has 1, neuron 3 has 1.
    assert_eq!(rp[1] - rp[0], 2);
    assert_eq!(rp[2] - rp[1], 1);
    assert_eq!(rp[3] - rp[2], 1);
    assert_eq!(rp[4] - rp[3], 1);

    // Synapse attrs and col_idx aligned with CSR.
    let sa = db.syn_attrs();
    let ci = db.csr_col_idx();
    assert_eq!(sa.len(), 5);
    assert_eq!(ci.len(), 5);
    for (s, c) in sa.iter().zip(ci.iter()) {
        assert_eq!(s.post_neuron as u64, *c);
    }

    // Out-range API.
    let r0 = db.out_range(0);
    assert_eq!(r0.len(), 2);
    let posts: Vec<u32> = sa[r0].iter().map(|s| s.post_neuron).collect();
    assert!(posts.contains(&1) && posts.contains(&2));

    // Receptor table mmap'd correctly.
    assert_eq!(db.receptors().len(), 1);
    assert!((db.receptors()[0].e_rev - 0.0).abs() < 1e-6);

    // Gap junctions.
    assert_eq!(db.gap_junctions().len(), 1);
    assert_eq!(db.gap_junctions()[0].pre_neuron, 0);

    // Dynamic state initialised: synapse `weight` should equal `base_weight`.
    for (st, at) in db.syn_states.iter().zip(db.syn_attrs().iter()) {
        assert_eq!(st.weight, at.base_weight);
    }
    // Neuron v_mem initialised to Izhikevich `c`.
    let expected_v = IzhikevichParams::regular_spiking().c;
    for s in &db.neuron_states {
        assert!((s.v_mem - expected_v).abs() < 1e-5);
    }

    // Meta section round-tripped.
    assert_eq!(db.meta.neuron_types.len(), 1);
    assert_eq!(db.meta.neuron_types[0].type_name, "RS");
}

#[test]
fn snapshot_save_load_roundtrip() {
    let dir = tempdir().unwrap();
    let dbpath = dir.path().join("snap.braindb");
    let snappath = dir.path().join("snap.braindb.snapshot");

    let mut b = BrainDBBuilder::new();
    make_tiny(&mut b);
    let mut db = b.build(&dbpath).unwrap();

    // Mutate dynamic state.
    db.current_tick = 12345;
    db.neuron_states[1].v_mem = -42.5;
    db.neuron_states[1].spike_count = 7;
    db.syn_states[2].weight = 0.111;
    db.syn_states[2].g_rise = 0.222;
    db.syn_states[2].is_active = 1;
    db.save_snapshot(&snappath).unwrap();

    // Re-open the DB (state reset to defaults) then reload the snapshot.
    let mut db2 = BrainDB::open(&dbpath).unwrap();
    assert_eq!(db2.current_tick, 0);
    assert_ne!(db2.neuron_states[1].v_mem, -42.5);

    db2.load_snapshot(&snappath).unwrap();
    assert_eq!(db2.current_tick, 12345);
    assert_eq!(db2.neuron_states[1].v_mem, -42.5);
    assert_eq!(db2.neuron_states[1].spike_count, 7);
    assert_eq!(db2.syn_states[2].weight, 0.111);
    assert_eq!(db2.syn_states[2].g_rise, 0.222);
    assert_eq!(db2.syn_states[2].is_active, 1);
}

#[test]
fn rejects_invalid_synapse_reference() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad.braindb");

    let mut b = BrainDBBuilder::new();
    b.add_neuron(NeuronAttr { id: 0, ..Default::default() });
    b.add_synapse(0, SynapseAttr { post_neuron: 99, ..Default::default() });

    let err = b.build(&path).err().expect("should reject bad synapse");
    let msg = format!("{err}");
    assert!(msg.contains("post_neuron"), "got: {msg}");
}
