//! Mmap round-trip and snapshot integrity tests.

use braindb::*;
use std::path::PathBuf;
use tempfile::tempdir;

fn build_tiny_db() -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mmap.braindb");
    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_region(BrainRegion { id: 0, neuron_count: 3, ..Default::default() }, "r");
    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "RS".into(), model: NeuronModel::Izhikevich,
        iz_params: IzhikevichParams::regular_spiking(), ..Default::default()
    });
    for id in 0..3u64 {
        b.add_neuron(NeuronAttr { id, neuron_type: nt, ..Default::default() });
    }
    b.add_synapse(0, SynapseAttr { post_neuron: 1, base_weight: 0.5, ..Default::default() });
    b.add_synapse(1, SynapseAttr { post_neuron: 2, base_weight: 0.3, ..Default::default() });
    b.add_gap_junction(GapJunction { pre_neuron: 0, post_neuron: 2, weight: 0.1, ..Default::default() });
    b.build(&path).unwrap();
    (dir, path)
}

#[test]
fn mmap_open_reads_correct_counts() {
    let (_dir, path) = build_tiny_db();
    let db = BrainDB::open(&path).unwrap();
    assert_eq!(db.header.n_neurons, 3);
    assert_eq!(db.header.n_synapses, 2);
    assert_eq!(db.header.n_gap_junctions, 1);
    assert_eq!(db.header.n_regions, 1);
}

#[test]
fn mmap_neuron_attrs_accessible() {
    let (_dir, path) = build_tiny_db();
    let db = BrainDB::open(&path).unwrap();
    let attrs = db.neuron_attrs();
    assert_eq!(attrs.len(), 3);
    assert_eq!(attrs[0].id, 0);
    assert_eq!(attrs[2].id, 2);
}

#[test]
fn mmap_csr_structure_correct() {
    let (_dir, path) = build_tiny_db();
    let db = BrainDB::open(&path).unwrap();
    let rp = db.csr_row_ptr();
    assert_eq!(rp.len(), 4); // n_neurons + 1
    assert_eq!(rp[0], 0);
    assert_eq!(*rp.last().unwrap(), 2); // total synapses
    // Neuron 0 has 1 outgoing, neuron 1 has 1, neuron 2 has 0.
    assert_eq!(rp[1] - rp[0], 1);
    assert_eq!(rp[2] - rp[1], 1);
    assert_eq!(rp[3] - rp[2], 0);
}

#[test]
fn snapshot_preserves_mutated_state() {
    let (dir, dbpath) = build_tiny_db();
    let snappath = dir.path().join("mmap.braindb.snapshot");

    let mut db = BrainDB::open(&dbpath).unwrap();
    db.neuron_states[0].v_mem = -42.0;
    db.neuron_states[0].spike_count = 99;
    db.syn_states[0].weight = 0.777;
    db.current_tick = 9999;
    db.save_snapshot(&snappath).unwrap();

    let mut db2 = BrainDB::open(&dbpath).unwrap();
    db2.load_snapshot(&snappath).unwrap();
    assert_eq!(db2.neuron_states[0].v_mem, -42.0);
    assert_eq!(db2.neuron_states[0].spike_count, 99);
    assert!((db2.syn_states[0].weight - 0.777).abs() < 1e-5);
    assert_eq!(db2.current_tick, 9999);
}

#[test]
fn snapshot_rejects_size_mismatch() {
    let (dir, dbpath) = build_tiny_db();
    let snappath = dir.path().join("other.braindb.snapshot");

    // Build a different-sized DB and save a snapshot.
    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_region(BrainRegion { id: 0, neuron_count: 5, ..Default::default() }, "r");
    let nt = b.add_neuron_type(NeuronTypeParams::default());
    for id in 0..5u64 {
        b.add_neuron(NeuronAttr { id, neuron_type: nt, ..Default::default() });
    }
    let other_path = dir.path().join("other.braindb");
    let other_db = b.build(&other_path).unwrap();
    other_db.save_snapshot(&snappath).unwrap();

    // Try to load it into the 3-neuron DB — should fail.
    let mut db = BrainDB::open(&dbpath).unwrap();
    let result = db.load_snapshot(&snappath);
    assert!(result.is_err(), "should reject snapshot with different neuron count");
}

#[test]
fn mmap_gap_junctions_readable() {
    let (_dir, path) = build_tiny_db();
    let db = BrainDB::open(&path).unwrap();
    let gjs = db.gap_junctions();
    assert_eq!(gjs.len(), 1);
    assert_eq!(gjs[0].pre_neuron, 0);
    assert_eq!(gjs[0].post_neuron, 2);
    assert!((gjs[0].weight - 0.1).abs() < 1e-5);
}
