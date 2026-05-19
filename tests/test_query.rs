//! Tests for `query::connectivity` and `query::oscillation`.

use braindb::*;
use tempfile::tempdir;

/// 5-neuron line graph: 0 → 1 → 2 → 3 → 4
fn build_line() -> BrainDB {
    let dir = tempdir().unwrap();
    let path = dir.path().join("line.braindb");
    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_region(BrainRegion { id: 0, neuron_count: 5, ..Default::default() }, "r");
    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "X".into(), model: NeuronModel::LIF, ..Default::default()
    });
    for id in 0..5u64 {
        b.add_neuron(NeuronAttr { id, neuron_type: nt, region_id: 0, ..Default::default() });
    }
    for i in 0..4u32 {
        b.add_synapse(i, SynapseAttr {
            post_neuron: i + 1,
            base_weight: 0.5,
            delay_ticks: 1,
            ..Default::default()
        });
    }
    b.build(&path).unwrap()
}

#[test]
fn bfs_downstream_line() {
    let db = build_line();
    let hits = bfs_downstream(&db, 0, 10);
    let ids: Vec<u64> = hits.iter().map(|(n, _)| *n).collect();
    assert_eq!(ids, vec![1, 2, 3, 4]);
    let depths: Vec<u32> = hits.iter().map(|(_, d)| *d).collect();
    assert_eq!(depths, vec![1, 2, 3, 4]);
}

#[test]
fn bfs_downstream_respects_max_hops() {
    let db = build_line();
    let hits = bfs_downstream(&db, 0, 2);
    let ids: Vec<u64> = hits.iter().map(|(n, _)| *n).collect();
    assert_eq!(ids, vec![1, 2]);
}

#[test]
fn bfs_upstream_line() {
    let db = build_line();
    let hits = bfs_upstream(&db, 4, 10);
    let mut ids: Vec<u64> = hits.iter().map(|(n, _)| *n).collect();
    ids.sort();
    assert_eq!(ids, vec![0, 1, 2, 3]);
}

#[test]
fn strongest_path_line() {
    let db = build_line();
    let (path, score) = strongest_path(&db, 0, 4).expect("path exists");
    assert_eq!(path, vec![0, 1, 2, 3, 4]);
    // 4 edges with weight 0.5 → log(0.5) * 4
    let expected = (0.5_f32.ln()) * 4.0;
    assert!((score - expected).abs() < 1e-3, "score {score} vs {expected}");
}

#[test]
fn strongest_path_picks_higher_product() {
    // Build a diamond: 0 → 1 → 3 (w=0.9 each) and 0 → 2 → 3 (w=0.4 each).
    let dir = tempdir().unwrap();
    let path = dir.path().join("diamond.braindb");
    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_region(BrainRegion { id: 0, neuron_count: 4, ..Default::default() }, "r");
    let nt = b.add_neuron_type(NeuronTypeParams { type_name: "X".into(), ..Default::default() });
    for id in 0..4u64 {
        b.add_neuron(NeuronAttr { id, neuron_type: nt, ..Default::default() });
    }
    let mk = |post, w| SynapseAttr { post_neuron: post, base_weight: w, delay_ticks: 1, ..Default::default() };
    b.add_synapse(0, mk(1, 0.9));
    b.add_synapse(1, mk(3, 0.9));
    b.add_synapse(0, mk(2, 0.4));
    b.add_synapse(2, mk(3, 0.4));
    let db = b.build(&path).unwrap();

    let (route, _) = strongest_path(&db, 0, 3).unwrap();
    assert_eq!(route, vec![0, 1, 3]);
}

#[test]
fn strongest_path_unreachable() {
    let db = build_line();
    // Reverse direction: target 0 from source 4 — no edge.
    assert!(strongest_path(&db, 4, 0).is_none());
}

#[test]
fn region_mean_lfp_basic() {
    let db = build_line();
    // All neurons start at e_leak = -70 mV.
    let mean = region_mean_lfp(&db, 0, &db.neuron_states);
    assert!((mean - (-70.0)).abs() < 1.0, "mean v_mem expected ~-70, got {mean}");
}
