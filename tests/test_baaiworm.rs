//! BAAIWorm C. elegans full-connectome integration test.
//!
//! Loads the entire 302-neuron C. elegans connectome from the BAAIWorm
//! project directory and verifies that the resulting BrainDB is structurally
//! correct.

use braindb::storage::loader::BAAIWormLoader;
use braindb::storage::mmap_db::BrainDB;
use std::path::PathBuf;
use tempfile::tempdir;

/// Path to the C. elegans connectome data directory.
/// Defaults to the bundled data/celegans/ inside this crate.
/// Override with BAAIWORM_DIR env var for external data sources.
fn eworm_dir() -> PathBuf {
    let base = std::env::var("BAAIWORM_DIR").unwrap_or_else(|_| {
        format!("{}/data/celegans",
            std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()))
    });
    PathBuf::from(base)
}

#[test]
fn baaiworm_loads_302_neurons() {
    let dir = eworm_dir();
    if !dir.exists() {
        eprintln!("SKIP: BAAIWorm directory not found at {:?}", dir);
        eprintln!("      Set BAAIWORM_DIR env var to run this test.");
        return;
    }

    let out_dir = tempdir().unwrap();
    let db_path = out_dir.path().join("celegans.braindb");

    let loader = BAAIWormLoader::load_from_dir(&dir)
        .expect("BAAIWormLoader::load_from_dir failed");
    let db = loader.into_braindb(&db_path)
        .expect("into_braindb failed");

    // C. elegans has 302 neurons.
    assert!(db.header.n_neurons >= 200, "expected ≥200 neurons, got {}", db.header.n_neurons);

    // Each neuron has 2 compartments (soma + neurite).
    assert_eq!(db.header.n_compartments, db.header.n_neurons * 2,
        "each neuron should have 2 compartments");

    // Should have chemical synapses (C. elegans has ~6000+).
    assert!(db.header.n_synapses > 0, "expected chemical synapses, got 0");

    // Should have gap junctions (C. elegans has ~500+).
    assert!(db.header.n_gap_junctions > 0, "expected gap junctions, got 0");

    // Reopen from disk to verify mmap round-trip.
    let db2 = BrainDB::open(&db_path).expect("reopen failed");
    assert_eq!(db2.header.n_neurons, db.header.n_neurons);
    assert_eq!(db2.header.n_synapses, db.header.n_synapses);
    assert_eq!(db2.header.n_gap_junctions, db.header.n_gap_junctions);
}

#[test]
fn baaiworm_simulation_runs() {
    let dir = eworm_dir();
    if !dir.exists() {
        eprintln!("SKIP: BAAIWorm directory not found at {:?}", dir);
        return;
    }

    let out_dir = tempdir().unwrap();
    let db_path = out_dir.path().join("celegans-sim.braindb");

    let loader = BAAIWormLoader::load_from_dir(&dir).unwrap();
    let db = loader.into_braindb(&db_path).unwrap();
    let mut sim = braindb::sim::engine::Simulation::new(db);

    // Run 1000 ticks (100 ms) without crashing.
    sim.run(1_000);

    // All neurons should still be alive (v_mem in physiological range).
    let n = sim.db.neuron_states.len();
    let mut n_valid = 0;
    for st in &sim.db.neuron_states {
        if st.v_mem > -120.0 && st.v_mem < 100.0 {
            n_valid += 1;
        }
    }
    assert!(n_valid > n / 2, "only {n_valid}/{n} neurons in valid range after 100ms");
}
