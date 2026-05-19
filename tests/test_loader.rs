//! Loader integration tests — verify ConnectomeLoader and NeuroMLLoader.

use braindb::*;
use braindb::storage::loader::connectome::ConnectomeLoader;
use braindb::storage::loader::neuroml::NeuroMLLoader;
use tempfile::tempdir;

#[test]
fn connectome_loader_csv_basic() {
    let dir = tempdir().unwrap();
    let csv_path = dir.path().join("connectome.csv");
    let db_path = dir.path().join("loaded.braindb");

    // Write a minimal CSV with the expected column names.
    std::fs::write(&csv_path, "pre_id,post_id,weight,delay_ms\n0,1,0.5,0.5\n1,2,0.3,0.3\n2,0,0.8,0.2\n").unwrap();

    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_region(BrainRegion { id: 0, neuron_count: 3, ..Default::default() }, "r");
    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "LIF".into(), model: NeuronModel::LIF, ..Default::default()
    });
    for id in 0..3u64 {
        b.add_neuron(NeuronAttr { id, neuron_type: nt, ..Default::default() });
    }
    ConnectomeLoader::load_csv(&csv_path, &mut b).unwrap();
    let db = b.build(&db_path).unwrap();

    assert_eq!(db.header.n_synapses, 3);
    assert_eq!(db.header.n_neurons, 3);
}

#[test]
fn connectome_loader_json_basic() {
    let dir = tempdir().unwrap();
    let json_path = dir.path().join("connectome.json");
    let db_path = dir.path().join("loaded.braindb");

    let json = r#"[
        {"pre_id": 0, "post_id": 1, "weight": 1.0, "delay_ms": 0.2, "syn_type": 0, "receptor_type": 0},
        {"pre_id": 1, "post_id": 0, "weight": 0.5, "delay_ms": 0.3, "syn_type": 0, "receptor_type": 0}
    ]"#;
    std::fs::write(&json_path, json).unwrap();

    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_region(BrainRegion { id: 0, neuron_count: 2, ..Default::default() }, "r");
    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "LIF".into(), model: NeuronModel::LIF, ..Default::default()
    });
    for id in 0..2u64 {
        b.add_neuron(NeuronAttr { id, neuron_type: nt, ..Default::default() });
    }
    ConnectomeLoader::load_json(&json_path, &mut b).unwrap();
    let db = b.build(&db_path).unwrap();

    assert_eq!(db.header.n_synapses, 2);
}

#[test]
fn neuroml_swc_loader() {
    let dir = tempdir().unwrap();
    let swc_path = dir.path().join("morpho.swc");
    let db_path = dir.path().join("swc.braindb");

    // Minimal SWC: 3 compartments, chain topology.
    let swc = "1 1 0.0 0.0 0.0 0.5 -1\n2 3 10.0 0.0 0.0 0.3 1\n3 3 20.0 0.0 0.0 0.2 2\n";
    std::fs::write(&swc_path, swc).unwrap();

    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_region(BrainRegion { id: 0, neuron_count: 1, ..Default::default() }, "r");
    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "HH".into(), model: NeuronModel::MultiCompartmentHH,
        iz_params: IzhikevichParams { v_rest: -65.0, ..IzhikevichParams::regular_spiking() },
        ..Default::default()
    });
    b.add_neuron(NeuronAttr {
        id: 0, neuron_type: nt, region_id: 0,
        first_comp_id: 0, n_compartment: 3,
        cm: 1.0, g_leak: 0.3, e_leak: -65.0,
        ..Default::default()
    });

    NeuroMLLoader::load_swc(&swc_path, &mut b, 0, 150.0, 1.0).unwrap();

    let db = b.build(&db_path).unwrap();
    assert_eq!(db.header.n_compartments, 3);
}
