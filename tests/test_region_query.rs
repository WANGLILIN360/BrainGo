//! Region-level query integration tests.
//!
//! Tests for region_pathway_info, outgoing_pathways, incoming_pathways,
//! and region_connectivity_matrix.

use braindb::*;
use braindb::query::region_query;
use tempfile::tempdir;

fn build_two_region_network() -> (tempfile::TempDir, BrainDB) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("regions.braindb");
    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_receptor(ReceptorParams::gaba_a());

    // Region 0: neurons 0..3 (sensory)
    b.add_region(
        BrainRegion {
            id: 0,
            first_neuron: 0,
            neuron_count: 3,
            ..Default::default()
        },
        "sensory",
    );
    // Region 1: neurons 3..6 (interneuron)
    b.add_region(
        BrainRegion {
            id: 1,
            first_neuron: 3,
            neuron_count: 3,
            ..Default::default()
        },
        "inter",
    );

    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "LIF".into(),
        model: NeuronModel::LIF,
        ..Default::default()
    });
    for id in 0..6u64 {
        let region_id = if id < 3 { 0u32 } else { 1u32 };
        b.add_neuron(NeuronAttr {
            id,
            neuron_type: nt,
            region_id,
            ..Default::default()
        });
    }

    // Cross-region synapses: 0→3 (excitatory), 1→4 (excitatory), 2→5 (excitatory)
    for (pre, post) in [(0u32, 3u32), (1u32, 4u32), (2u32, 5u32)] {
        b.add_synapse(pre, SynapseAttr {
            post_neuron: post,
            base_weight: 2.0,
            delay_ticks: 1,
            syn_type: SYN_EXCITATORY,
            syn_mode: SYN_MODE_EVENT_DRIVEN,
            receptor_type: RECEPTOR_AMPA,
            ..Default::default()
        });
    }
    // Intra-region synapse: 3→4 (inhibitory)
    b.add_synapse(3, SynapseAttr {
        post_neuron: 4,
        base_weight: -1.0,
        delay_ticks: 1,
        syn_type: SYN_INHIBITORY,
        syn_mode: SYN_MODE_EVENT_DRIVEN,
        receptor_type: RECEPTOR_GABAA,
        ..Default::default()
    });

    // Add a long-range pathway between regions.
    b.add_pathway(LongRangePathway {
        source_region: 0,
        target_region: 1,
        pathway_type: 0,
        fiber_count: 3,
        conduction_speed: 0.5,
        ..Default::default()
    });

    let db = b.build(&path).unwrap();
    (dir, db)
}

#[test]
fn region_pathway_info_cross_region() {
    let (_dir, db) = build_two_region_network();
    let info = region_query::region_pathway_info(&db, 0, 1).unwrap();

    assert_eq!(info.source_region, 0);
    assert_eq!(info.target_region, 1);
    assert_eq!(info.synapse_count, 3, "3 cross-region synapses 0→3, 1→4, 2→5");
    assert!((info.total_weight - 6.0).abs() < 0.01, "total_weight should be 6.0, got {}", info.total_weight);
    assert!((info.mean_weight - 2.0).abs() < 0.01, "mean_weight should be 2.0, got {}", info.mean_weight);
    assert_eq!(info.pre_neuron_count, 3, "3 unique pre-neurons");
    assert_eq!(info.post_neuron_count, 3, "3 unique post-neurons");
}

#[test]
fn region_pathway_info_no_connection() {
    let (_dir, db) = build_two_region_network();
    // Region 1 → Region 0 has no synapses.
    let info = region_query::region_pathway_info(&db, 1, 0).unwrap();
    assert_eq!(info.synapse_count, 0);
    assert_eq!(info.total_weight, 0.0);
}

#[test]
fn region_pathway_info_same_region() {
    let (_dir, db) = build_two_region_network();
    // Region 1 → Region 1 has one intra-region synapse (3→4).
    let info = region_query::region_pathway_info(&db, 1, 1).unwrap();
    assert_eq!(info.synapse_count, 1);
    assert!((info.total_weight - (-1.0)).abs() < 0.01, "intra-region inhibitory weight");
}

#[test]
fn outgoing_pathways_from_sensory() {
    let (_dir, db) = build_two_region_network();
    let out = region_query::outgoing_pathways(&db, 0);
    assert_eq!(out.len(), 1, "region 0 should project to 1 other region");
    assert_eq!(out[0].target_region, 1);
    assert_eq!(out[0].synapse_count, 3);
}

#[test]
fn incoming_pathways_to_interneuron() {
    let (_dir, db) = build_two_region_network();
    let inc = region_query::incoming_pathways(&db, 1);
    assert_eq!(inc.len(), 1, "region 1 should receive from 1 other region");
    assert_eq!(inc[0].source_region, 0);
    assert_eq!(inc[0].synapse_count, 3);
}

#[test]
fn pathways_between_regions() {
    let (_dir, db) = build_two_region_network();
    let pws = region_query::pathways_between(&db, 0, 1);
    assert_eq!(pws.len(), 1, "one long-range pathway between 0 and 1");
    assert_eq!(pws[0].source_region, 0);
    assert_eq!(pws[0].target_region, 1);
    assert_eq!(pws[0].fiber_count, 3);
}

#[test]
fn region_connectivity_matrix_shape() {
    let (_dir, db) = build_two_region_network();
    let mat = region_query::region_connectivity_matrix(&db);
    assert_eq!(mat.len(), 2, "2 regions → 2×2 matrix");
    assert_eq!(mat[0].len(), 2);

    // (0,1) = total weight of synapses from region 0 → 1 = 6.0
    assert!((mat[0][1] - 6.0).abs() < 0.01, "mat[0][1] should be 6.0, got {}", mat[0][1]);
    // (1,0) = 0 (no synapses from region 1 → 0)
    assert!((mat[1][0]).abs() < 0.01, "mat[1][0] should be 0, got {}", mat[1][0]);
    // (1,1) = -1.0 (intra-region inhibitory synapse 3→4)
    assert!((mat[1][1] - (-1.0)).abs() < 0.01, "mat[1][1] should be -1.0, got {}", mat[1][1]);
}
