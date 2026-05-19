//! BrainDB CLI — command-line interface for building, loading, running,
//! and querying brain network simulations.

#[cfg(feature = "cli")]
use clap::{Parser, Subcommand};

#[cfg(feature = "cli")]
use braindb::{
    BrainDBBuilder, BrainDB, BrainRegion, NeuronAttr, NeuronTypeParams, NeuronModel,
    IzhikevichParams, SynapseAttr, ReceptorParams,
    SYN_EXCITATORY, SYN_MODE_EVENT_DRIVEN, RECEPTOR_AMPA,
    bfs_downstream, region_pathway_info,
};
#[cfg(feature = "cli")]
use braindb::sim::engine::Simulation;
#[cfg(feature = "cli")]
use braindb::storage::loader::BAAIWormLoader;
#[cfg(feature = "cli")]
use rand::SeedableRng;

#[cfg(feature = "cli")]
#[derive(Parser)]
#[command(name = "braindb", version, about = "BrainDB — brain network simulation database")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[cfg(feature = "cli")]
#[derive(Subcommand)]
enum Commands {
    /// Build a new .braindb file from a network description.
    Build {
        /// Output .braindb path.
        #[arg(short, long)]
        output: String,
        /// Number of neurons.
        #[arg(short, long, default_value = "10")]
        neurons: u32,
        /// Neuron model: izhikevich, lif, graded.
        #[arg(short, long, default_value = "izhikevich")]
        model: String,
        /// Connectivity probability (0.0–1.0).
        #[arg(short, long, default_value = "0.3")]
        p_connect: f64,
    },

    /// Load a connectome directory into a .braindb file.
    /// Use `data/celegans/` for the bundled C. elegans data.
    LoadWorm {
        /// Path to connectome data directory.
        dir: String,
        /// Output .braindb path.
        #[arg(short, long, default_value = "celegans.braindb")]
        output: String,
    },

    /// Open an existing .braindb and print summary info.
    Info {
        /// Path to .braindb file.
        path: String,
    },

    /// Run a simulation on a .braindb file.
    Run {
        /// Path to .braindb file.
        path: String,
        /// Duration in milliseconds.
        #[arg(short, long, default_value = "100")]
        duration_ms: u64,
        /// Inject current into neuron(s), e.g. "0:30,1:20".
        #[arg(short, long)]
        stimulus: Option<String>,
        /// Enable STDP.
        #[arg(long)]
        stdp: bool,
        /// Enable structural plasticity.
        #[arg(long)]
        structural_plasticity: bool,
        /// Save snapshot after run.
        #[arg(long)]
        snapshot: Option<String>,
    },

    /// Query connectivity in a .braindb file.
    Query {
        /// Path to .braindb file.
        path: String,
        /// Query type: downstream, upstream, pathway, matrix.
        #[command(subcommand)]
        query: QueryCommands,
    },

    /// Save a snapshot of the current simulation state.
    Snapshot {
        /// Path to .braindb file.
        path: String,
        /// Snapshot output path.
        #[arg(short, long)]
        output: String,
    },

    /// Load a snapshot into an existing .braindb.
    LoadSnapshot {
        /// Path to .braindb file.
        path: String,
        /// Snapshot path to load.
        snapshot: String,
    },
}

#[cfg(feature = "cli")]
#[derive(Subcommand)]
enum QueryCommands {
    /// BFS downstream from a neuron.
    Downstream {
        /// Source neuron ID.
        neuron_id: u32,
        /// Max hops.
        #[arg(short, long, default_value = "3")]
        max_hops: usize,
    },
    /// Query inter-region pathway info.
    Pathway {
        /// Source region ID.
        source: u32,
        /// Target region ID.
        target: u32,
    },
    /// Print region connectivity matrix.
    Matrix,
}

#[cfg(feature = "cli")]
fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build { output, neurons, model, p_connect } => cmd_build(&output, neurons, &model, p_connect),
        Commands::LoadWorm { dir, output } => cmd_load_worm(&dir, &output),

        Commands::Info { path } => cmd_info(&path),
        Commands::Run { path, duration_ms, stimulus, stdp, structural_plasticity, snapshot } => {
            cmd_run(&path, duration_ms, stimulus.as_deref(), stdp, structural_plasticity, snapshot.as_deref())
        }
        Commands::Query { path, query } => cmd_query(&path, query),
        Commands::Snapshot { path, output } => cmd_snapshot(&path, &output),
        Commands::LoadSnapshot { path, snapshot } => cmd_load_snapshot(&path, &snapshot),
    }
}

#[cfg(feature = "cli")]
fn cmd_build(output: &str, n_neurons: u32, model_str: &str, p_connect: f64) {
    let model = match model_str {
        "lif" => NeuronModel::LIF,
        "graded" => NeuronModel::Graded,
        _ => NeuronModel::Izhikevich,
    };
    let mut b = BrainDBBuilder::new();
    b.add_receptor(ReceptorParams::ampa());
    b.add_region(BrainRegion { id: 0, first_neuron: 0, neuron_count: n_neurons, ..Default::default() }, "default");
    let nt = b.add_neuron_type(NeuronTypeParams {
        type_name: "main".into(),
        model,
        iz_params: IzhikevichParams::regular_spiking(),
        ..Default::default()
    });
    for id in 0..n_neurons {
        b.add_neuron(NeuronAttr { id: id as u64, neuron_type: nt, region_id: 0, ..Default::default() });
    }
    // Random connections.
    let mut rng = rand::rngs::StdRng::from_seed([42u8; 32]);
    for pre in 0..n_neurons {
        for post in 0..n_neurons {
            if pre == post { continue; }
            if rand::Rng::gen_ratio(&mut rng, (p_connect * 1000.0) as u32, 1000) {
                b.add_synapse(pre, SynapseAttr {
                    post_neuron: post,
                    base_weight: 1.0,
                    delay_ticks: 1,
                    syn_type: SYN_EXCITATORY,
                    syn_mode: SYN_MODE_EVENT_DRIVEN,
                    receptor_type: RECEPTOR_AMPA,
                    ..Default::default()
                });
            }
        }
    }
    let db = b.build(std::path::Path::new(output)).expect("build failed");
    println!("Built {} → {} neurons, {} synapses, {} gap junctions",
        output, db.header.n_neurons, db.header.n_synapses, db.header.n_gap_junctions);
}

#[cfg(feature = "cli")]
fn cmd_load_worm(dir: &str, output: &str) {
    let loader = BAAIWormLoader::load_from_dir(std::path::Path::new(dir))
        .expect("load_from_dir failed");
    let db = loader.into_braindb(std::path::Path::new(output))
        .expect("into_braindb failed");
    println!("Loaded from {} → {} neurons, {} synapses, {} gap junctions",
        dir, db.header.n_neurons, db.header.n_synapses, db.header.n_gap_junctions);
}

#[cfg(feature = "cli")]
fn cmd_info(path: &str) {
    let db = BrainDB::open(std::path::Path::new(path)).expect("open failed");
    let h = &db.header;
    println!("BrainDB: {}", path);
    println!("  Version:        {}", h.version);
    println!("  Neurons:        {}", h.n_neurons);
    println!("  Synapses:       {}", h.n_synapses);
    println!("  Gap junctions:  {}", h.n_gap_junctions);
    println!("  Compartments:   {}", h.n_compartments);
    println!("  Regions:        {}", db.regions().len());
    println!("  Pathways:       {}", db.pathways().len());
    println!("  Neuron types:   {}", db.meta.neuron_types.len());
    println!("  Ion channels:   {}", db.meta.ion_channels.len());
    println!("  Ion channel sets: {}", db.meta.ion_channel_sets.len());
}

#[cfg(feature = "cli")]
fn cmd_run(path: &str, duration_ms: u64, stimulus: Option<&str>, stdp: bool, sp: bool, snapshot: Option<&str>) {
    let db = BrainDB::open(std::path::Path::new(path)).expect("open failed");
    let mut sim = Simulation::new(db);
    sim.config.stdp_enabled = stdp;
    sim.config.structural_plasticity_enabled = sp;

    // Parse stimulus string: "0:30,1:20" → inject into neurons 0 and 1.
    if let Some(s) = stimulus {
        for pair in s.split(',') {
            let parts: Vec<&str> = pair.split(':').collect();
            if parts.len() == 2 {
                if let (Ok(nid), Ok(cur)) = (parts[0].parse::<usize>(), parts[1].parse::<f32>()) {
                    if nid < sim.db.neuron_states.len() {
                        sim.db.neuron_states[nid].i_ext = cur;
                    }
                }
            }
        }
    }

    let ticks = duration_ms * 10; // dt = 0.1 ms
    println!("Running {} ms ({} ticks)...", duration_ms, ticks);
    sim.run(ticks);

    // Print summary.
    let n_spikes: u64 = sim.db.neuron_states.iter().map(|s| s.spike_count as u64).sum();
    let v_min = sim.db.neuron_states.iter().map(|s| s.v_mem).fold(f32::INFINITY, f32::min);
    let v_max = sim.db.neuron_states.iter().map(|s| s.v_mem).fold(f32::NEG_INFINITY, f32::max);
    println!("Done. Total spikes: {}, V range: [{:.1}, {:.1}] mV", n_spikes, v_min, v_max);

    if let Some(snap_path) = snapshot {
        sim.db.save_snapshot(std::path::Path::new(snap_path)).expect("save_snapshot failed");
        println!("Snapshot saved to {}", snap_path);
    }
}

#[cfg(feature = "cli")]
fn cmd_query(path: &str, query: QueryCommands) {
    let db = BrainDB::open(std::path::Path::new(path)).expect("open failed");
    match query {
        QueryCommands::Downstream { neuron_id, max_hops } => {
            let hits = bfs_downstream(&db, neuron_id as u64, max_hops as u32);
            println!("Downstream from neuron {} (max {} hops):", neuron_id, max_hops);
            for (nid, hops) in &hits {
                println!("  neuron {} (hops: {})", nid, hops);
            }
            println!("Total: {} neurons", hits.len());
        }
        QueryCommands::Pathway { source, target } => {
            match region_pathway_info(&db, source, target) {
                Some(info) => {
                    println!("Pathway region {} → {}:", source, target);
                    println!("  Synapses:       {}", info.synapse_count);
                    println!("  Total weight:   {:.3}", info.total_weight);
                    println!("  Mean weight:    {:.3}", info.mean_weight);
                    println!("  Pre neurons:    {}", info.pre_neuron_count);
                    println!("  Post neurons:   {}", info.post_neuron_count);
                }
                None => println!("No pathway found between regions {} → {}", source, target),
            }
        }
        QueryCommands::Matrix => {
            let mat = braindb::query::region_query::region_connectivity_matrix(&db);
            let n = mat.len();
            print!("     ");
            for j in 0..n { print!("{:>8}", j); }
            println!();
            for i in 0..n {
                print!("{:>4} ", i);
                for j in 0..n {
                    print!("{:>8.2}", mat[i][j]);
                }
                println!();
            }
        }
    }
}

#[cfg(feature = "cli")]
fn cmd_snapshot(path: &str, output: &str) {
    let db = BrainDB::open(std::path::Path::new(path)).expect("open failed");
    db.save_snapshot(std::path::Path::new(output)).expect("save_snapshot failed");
    println!("Snapshot saved: {} → {}", path, output);
}

#[cfg(feature = "cli")]
fn cmd_load_snapshot(path: &str, snapshot: &str) {
    let mut db = BrainDB::open(std::path::Path::new(path)).expect("open failed");
    db.load_snapshot(std::path::Path::new(snapshot)).expect("load_snapshot failed");
    println!("Snapshot loaded: {} → {}", snapshot, path);
}

// ── No-feature fallback ──────────────────────────────────────────────────
#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!("braindb-cli requires the 'cli' feature. Build with:");
    eprintln!("  cargo build --features cli");
}
