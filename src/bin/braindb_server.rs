//! BrainDB Server — REST API + Web management interface.
//!
//! Provides an HTTP server for interacting with BrainDB simulations
//! via a REST API and a built-in web dashboard.

#[cfg(feature = "server")]
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
#[cfg(feature = "server")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "server")]
use std::collections::HashMap;
#[cfg(feature = "server")]
use std::sync::Arc;
#[cfg(feature = "server")]
use tokio::sync::Mutex;
#[cfg(feature = "server")]
use tower_http::cors::CorsLayer;
#[cfg(feature = "server")]
use tracing::info;

// ── Shared state ────────────────────────────────────────────────────────

#[cfg(feature = "server")]
struct AppState {
    /// Open databases: name → BrainDB.
    databases: Mutex<HashMap<String, BrainDB>>,
    /// Running simulations: name → Simulation.
    simulations: Mutex<HashMap<String, Simulation>>,
}

#[cfg(feature = "server")]
use braindb::sim::engine::Simulation;
#[cfg(feature = "server")]
use braindb::storage::mmap_db::BrainDB;

// ── API types ───────────────────────────────────────────────────────────

#[cfg(feature = "server")]
#[derive(Serialize)]
struct DbInfo {
    path: String,
    n_neurons: u64,
    n_synapses: u64,
    n_gap_junctions: u64,
    n_compartments: u64,
    n_regions: usize,
}

#[cfg(feature = "server")]
#[derive(Deserialize)]
struct RunRequest {
    /// Duration in milliseconds.
    duration_ms: u64,
    /// Stimulus: neuron_id → current (pA).
    #[serde(default)]
    stimulus: HashMap<u32, f32>,
    /// Enable STDP.
    #[serde(default)]
    stdp: bool,
    /// Enable structural plasticity.
    #[serde(default)]
    structural_plasticity: bool,
}

#[cfg(feature = "server")]
#[derive(Serialize)]
struct RunResult {
    ticks: u64,
    total_spikes: u64,
    v_min: f32,
    v_max: f32,
}

#[cfg(feature = "server")]
#[derive(Serialize)]
struct NeuronStateResponse {
    neuron_id: u32,
    v_mem: f32,
    spike_count: u64,
    i_total: f32,
}

#[cfg(feature = "server")]
#[derive(Deserialize)]
struct StimulusQuery {
    neuron_id: u32,
    current: f32,
}

#[cfg(feature = "server")]
#[derive(Serialize)]
struct GraphNode {
    id: u32,
    label: String,
    region: u32,
    v_mem: f32,
    spike_count: u64,
    group: u32,
    x: f32,
    y: f32,
    z: f32,
}

#[cfg(feature = "server")]
#[derive(Serialize)]
struct GraphEdge {
    source: u32,
    target: u32,
    weight: f32,
    #[serde(rename = "type")]
    edge_type: String, // "chemical" | "gap"
}

#[cfg(feature = "server")]
#[derive(Serialize)]
struct GraphData {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    regions: Vec<RegionInfo>,
}

#[cfg(feature = "server")]
#[derive(Serialize)]
struct RegionInfo {
    id: u32,
    name: String,
    neuron_count: u32,
}

#[cfg(feature = "server")]
#[derive(Serialize)]
struct CompartmentInfo {
    id: u64,
    comp_type: u8,       // 0=Soma, 1=ApicalDend, 2=BasalDend, 3=Axon
    parent_comp_id: u64, // u64::MAX = root
    x: f32,
    y: f32,
    z: f32,
    diameter: f32,
    length: f32,
    v_mem: f32,
}

#[cfg(feature = "server")]
#[derive(Serialize)]
struct NeuronMorphology {
    neuron_id: u32,
    label: String,
    region: u32,
    v_mem: f32,
    spike_count: u64,
    compartments: Vec<CompartmentInfo>,
}

#[cfg(feature = "server")]
#[derive(Serialize)]
struct MorphologyData {
    neurons: Vec<NeuronMorphology>,
    regions: Vec<RegionInfo>,
}

// ── Route handlers ──────────────────────────────────────────────────────

#[cfg(feature = "server")]
#[derive(Serialize)]
struct NeighborHit {
    neuron_id: u64,
    hops: u32,
}

#[cfg(feature = "server")]
#[derive(Serialize)]
struct StrongestPathResult {
    path: Vec<u64>,
    total_log_weight: f32,
}

#[cfg(feature = "server")]
#[derive(Serialize)]
struct RegionPathwayResult {
    source_region: u32,
    target_region: u32,
    synapse_count: usize,
    total_weight: f32,
}

#[cfg(feature = "server")]
#[derive(Serialize)]
struct NeuronDetail {
    id: u32,
    label: String,
    region: u32,
    region_name: String,
    v_mem: f32,
    spike_count: u64,
    i_total: f32,
    i_ext: f32,
    i_syn: f32,
    i_gap: f32,
    stdp_trace: f32,
    cai: f32,
    is_alive: bool,
    x: f32,
    y: f32,
    z: f32,
    n_compartment: u32,
    out_synapses: u32,
    in_synapses: u32,
}

#[cfg(feature = "server")]
#[derive(Deserialize)]
struct NeighborsQuery {
    id: u64,
    #[serde(default = "default_depth")]
    depth: u32,
}

#[cfg(feature = "server")]
fn default_depth() -> u32 { 1 }

#[cfg(feature = "server")]
#[derive(Deserialize)]
struct StrongestPathQuery {
    from: u64,
    to: u64,
}

#[cfg(feature = "server")]
#[derive(Deserialize)]
struct SimConfigUpdate {
    #[serde(default)]
    stdp_enabled: Option<bool>,
    #[serde(default)]
    structural_plasticity_enabled: Option<bool>,
    #[serde(default)]
    modulation_enabled: Option<bool>,
    #[serde(default)]
    spike_threshold_mv: Option<f32>,
    #[serde(default)]
    max_syn_weight: Option<f32>,
}

#[cfg(feature = "server")]
async fn api_list_dbs(State(state): State<Arc<AppState>>) -> Json<Vec<String>> {
    let dbs = state.databases.lock().await;
    Json(state.simulations.lock().await.keys().cloned().chain(dbs.keys().cloned()).collect())
}

#[cfg(feature = "server")]
#[derive(Deserialize)]
struct OpenDbRequest {
    path: String,
}

#[cfg(feature = "server")]
async fn api_open_db(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OpenDbRequest>,
) -> Result<Json<DbInfo>, StatusCode> {
    let p = std::path::Path::new(&req.path);
    info!("Opening database: {:?} (exists: {})", p, p.exists());
    let db = BrainDB::open(p).map_err(|e| {
        info!("Open error: {:?}", e);
        StatusCode::NOT_FOUND
    })?;
    let name = std::path::Path::new(&req.path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("db")
        .to_string();
    let info = DbInfo {
        path: req.path.clone(),
        n_neurons: db.header.n_neurons,
        n_synapses: db.header.n_synapses,
        n_gap_junctions: db.header.n_gap_junctions,
        n_compartments: db.header.n_compartments,
        n_regions: db.regions().len(),
    };
    state.databases.lock().await.insert(name.clone(), db);
    info!("Opened database: {}", req.path);
    Ok(Json(info))
}

#[cfg(feature = "server")]
async fn api_db_info(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<DbInfo>, StatusCode> {
    let dbs = state.databases.lock().await;
    let db = dbs.get(&name).ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(DbInfo {
        path: name.clone(),
        n_neurons: db.header.n_neurons,
        n_synapses: db.header.n_synapses,
        n_gap_junctions: db.header.n_gap_junctions,
        n_compartments: db.header.n_compartments,
        n_regions: db.regions().len(),
    }))
}

#[cfg(feature = "server")]
async fn api_init_simulation(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<StatusCode, StatusCode> {
    let db = {
        let mut dbs = state.databases.lock().await;
        dbs.remove(&name).ok_or(StatusCode::NOT_FOUND)?
    };
    let sim = Simulation::new(db);
    state.simulations.lock().await.insert(name.clone(), sim);
    info!("Initialised simulation: {}", name);
    Ok(StatusCode::OK)
}

#[cfg(feature = "server")]
async fn api_run_simulation(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<RunRequest>,
) -> Result<Json<RunResult>, StatusCode> {
    let mut sims = state.simulations.lock().await;
    let sim = sims.get_mut(&name).ok_or(StatusCode::NOT_FOUND)?;
    sim.config.stdp_enabled = req.stdp;
    sim.config.structural_plasticity_enabled = req.structural_plasticity;
    for (&nid, &cur) in &req.stimulus {
        if (nid as usize) < sim.db.neuron_states.len() {
            sim.db.neuron_states[nid as usize].i_ext = cur;
        }
    }
    let ticks = req.duration_ms * 10;
    sim.run(ticks);
    let total_spikes: u64 = sim.db.neuron_states.iter().map(|s| s.spike_count as u64).sum();
    let v_min = sim.db.neuron_states.iter().map(|s| s.v_mem).fold(f32::INFINITY, f32::min);
    let v_max = sim.db.neuron_states.iter().map(|s| s.v_mem).fold(f32::NEG_INFINITY, f32::max);
    Ok(Json(RunResult { ticks, total_spikes, v_min, v_max }))
}

#[cfg(feature = "server")]
async fn api_neuron_states(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<Vec<NeuronStateResponse>>, StatusCode> {
    let sims = state.simulations.lock().await;
    let sim = sims.get(&name).ok_or(StatusCode::NOT_FOUND)?;
    let states: Vec<_> = sim.db.neuron_states.iter().enumerate().map(|(i, s)| {
        NeuronStateResponse {
            neuron_id: i as u32,
            v_mem: s.v_mem,
            spike_count: s.spike_count as u64,
            i_total: s.i_total,
        }
    }).collect();
    Ok(Json(states))
}

#[cfg(feature = "server")]
async fn api_set_stimulus(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(q): Query<StimulusQuery>,
) -> StatusCode {
    let mut sims = state.simulations.lock().await;
    if let Some(sim) = sims.get_mut(&name) {
        if (q.neuron_id as usize) < sim.db.neuron_states.len() {
            sim.db.neuron_states[q.neuron_id as usize].i_ext = q.current;
            return StatusCode::OK;
        }
    }
    StatusCode::NOT_FOUND
}

#[cfg(feature = "server")]
async fn api_snapshot(
    State(state): State<Arc<AppState>>,
    Path((name, snap_path)): Path<(String, String)>,
) -> StatusCode {
    let sims = state.simulations.lock().await;
    if let Some(sim) = sims.get(&name) {
        if sim.db.save_snapshot(std::path::Path::new(&snap_path)).is_ok() {
            return StatusCode::OK;
        }
    }
    StatusCode::INTERNAL_SERVER_ERROR
}

// ── Query endpoints ────────────────────────────────────────────────────

#[cfg(feature = "server")]
async fn api_query_neighbors(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(q): Query<NeighborsQuery>,
) -> Result<Json<Vec<NeighborHit>>, StatusCode> {
    let hits = {
        let sims = state.simulations.lock().await;
        if let Some(sim) = sims.get(&name) {
            braindb::query::connectivity::bfs_downstream(&sim.db, q.id, q.depth)
        } else {
            drop(sims);
            let dbs = state.databases.lock().await;
            let db = dbs.get(&name).ok_or(StatusCode::NOT_FOUND)?;
            braindb::query::connectivity::bfs_downstream(db, q.id, q.depth)
        }
    };
    Ok(Json(hits.into_iter().map(|(neuron_id, hops)| NeighborHit { neuron_id, hops }).collect()))
}

#[cfg(feature = "server")]
async fn api_query_upstream(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(q): Query<NeighborsQuery>,
) -> Result<Json<Vec<NeighborHit>>, StatusCode> {
    let hits = {
        let sims = state.simulations.lock().await;
        if let Some(sim) = sims.get(&name) {
            braindb::query::connectivity::bfs_upstream(&sim.db, q.id, q.depth)
        } else {
            drop(sims);
            let dbs = state.databases.lock().await;
            let db = dbs.get(&name).ok_or(StatusCode::NOT_FOUND)?;
            braindb::query::connectivity::bfs_upstream(db, q.id, q.depth)
        }
    };
    Ok(Json(hits.into_iter().map(|(neuron_id, hops)| NeighborHit { neuron_id, hops }).collect()))
}

#[cfg(feature = "server")]
async fn api_query_strongest_path(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(q): Query<StrongestPathQuery>,
) -> Result<Json<StrongestPathResult>, StatusCode> {
    let result = {
        let sims = state.simulations.lock().await;
        if let Some(sim) = sims.get(&name) {
            braindb::query::connectivity::strongest_path(&sim.db, q.from, q.to)
        } else {
            drop(sims);
            let dbs = state.databases.lock().await;
            let db = dbs.get(&name).ok_or(StatusCode::NOT_FOUND)?;
            braindb::query::connectivity::strongest_path(db, q.from, q.to)
        }
    };
    match result {
        Some((path, weight)) => Ok(Json(StrongestPathResult { path, total_log_weight: weight })),
        None => Ok(Json(StrongestPathResult { path: Vec::new(), total_log_weight: 0.0 })),
    }
}

#[cfg(feature = "server")]
async fn api_query_region_pathway(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(q): Query<RegionPathwayQuery>,
) -> Result<Json<RegionPathwayResult>, StatusCode> {
    let info = {
        let sims = state.simulations.lock().await;
        if let Some(sim) = sims.get(&name) {
            braindb::query::region_query::region_pathway_info(&sim.db, q.source, q.target)
        } else {
            drop(sims);
            let dbs = state.databases.lock().await;
            let db = dbs.get(&name).ok_or(StatusCode::NOT_FOUND)?;
            braindb::query::region_query::region_pathway_info(db, q.source, q.target)
        }
    };
    match info {
        Some(i) => Ok(Json(RegionPathwayResult {
            source_region: i.source_region,
            target_region: i.target_region,
            synapse_count: i.synapse_count,
            total_weight: i.total_weight,
        })),
        None => Ok(Json(RegionPathwayResult {
            source_region: q.source,
            target_region: q.target,
            synapse_count: 0,
            total_weight: 0.0,
        })),
    }
}

#[cfg(feature = "server")]
#[derive(Deserialize)]
struct RegionPathwayQuery {
    source: u32,
    target: u32,
}

#[cfg(feature = "server")]
async fn api_query_region_lfp(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(q): Query<RegionIdQuery>,
) -> Result<Json<f32>, StatusCode> {
    let lfp = {
        let sims = state.simulations.lock().await;
        if let Some(sim) = sims.get(&name) {
            braindb::query::oscillation::region_mean_lfp(&sim.db, q.region_id, &sim.db.neuron_states)
        } else {
            drop(sims);
            let dbs = state.databases.lock().await;
            let db = dbs.get(&name).ok_or(StatusCode::NOT_FOUND)?;
            braindb::query::oscillation::region_mean_lfp(db, q.region_id, &db.neuron_states)
        }
    };
    Ok(Json(lfp))
}

#[cfg(feature = "server")]
#[derive(Deserialize)]
struct RegionIdQuery {
    region_id: u32,
}

// ── Neuron detail endpoint ─────────────────────────────────────────────

#[cfg(feature = "server")]
async fn api_neuron_detail(
    State(state): State<Arc<AppState>>,
    Path((name, nid)): Path<(String, u32)>,
) -> Result<Json<NeuronDetail>, StatusCode> {
    let (attr, st, region_name, out_count, in_count, type_name) = {
        let sims = state.simulations.lock().await;
        if let Some(sim) = sims.get(&name) {
            let i = nid as usize;
            let attrs = sim.db.neuron_attrs();
            if i >= attrs.len() { return Err(StatusCode::NOT_FOUND); }
            let attr = attrs[i];
            let st = sim.db.neuron_states.get(i).cloned().unwrap_or_default();
            let region_name = sim.db.meta.region_names.iter()
                .find(|(id, _)| *id == attr.region_id)
                .map(|(_, name)| name.clone())
                .unwrap_or_else(|| format!("region_{}", attr.region_id));
            let out_count = sim.db.out_range(i).len() as u32;
            // Count incoming synapses from reverse CSR.
            let in_count = if nid as usize + 1 < sim.rev_csr_row_ptr.len() {
                (sim.rev_csr_row_ptr[nid as usize + 1] - sim.rev_csr_row_ptr[nid as usize]) as u32
            } else { 0 };
            let type_name = sim.db.meta.neuron_types.get(attr.neuron_type as usize)
                .map(|t| t.type_name.clone())
                .unwrap_or_else(|| format!("N{}", nid));
            (attr, st, region_name, out_count, in_count, type_name)
        } else {
            drop(sims);
            let dbs = state.databases.lock().await;
            let db = dbs.get(&name).ok_or(StatusCode::NOT_FOUND)?;
            let i = nid as usize;
            let attrs = db.neuron_attrs();
            if i >= attrs.len() { return Err(StatusCode::NOT_FOUND); }
            let attr = attrs[i];
            let st = db.neuron_states.get(i).cloned().unwrap_or_default();
            let region_name = db.meta.region_names.iter()
                .find(|(id, _)| *id == attr.region_id)
                .map(|(_, name)| name.clone())
                .unwrap_or_else(|| format!("region_{}", attr.region_id));
            let out_count = db.out_range(i).len() as u32;
            let in_count = 0u32; // no rev_csr in raw BrainDB
            let type_name = db.meta.neuron_types.get(attr.neuron_type as usize)
                .map(|t| t.type_name.clone())
                .unwrap_or_else(|| format!("N{}", nid));
            (attr, st, region_name, out_count, in_count, type_name)
        }
    };
    Ok(Json(NeuronDetail {
        id: nid,
        label: type_name,
        region: attr.region_id,
        region_name,
        v_mem: st.v_mem,
        spike_count: st.spike_count as u64,
        i_total: st.i_total,
        i_ext: st.i_ext,
        i_syn: st.i_syn,
        i_gap: st.i_gap,
        stdp_trace: st.stdp_trace,
        cai: st.cai,
        is_alive: true, // simplified — no flags in raw db
        x: attr.x,
        y: attr.y,
        z: attr.z,
        n_compartment: attr.n_compartment,
        out_synapses: out_count,
        in_synapses: in_count,
    }))
}

// ── Neuron control endpoints ───────────────────────────────────────────

#[cfg(feature = "server")]
async fn api_kill_neuron(
    State(state): State<Arc<AppState>>,
    Path((name, nid)): Path<(String, u32)>,
) -> StatusCode {
    let mut sims = state.simulations.lock().await;
    if let Some(sim) = sims.get_mut(&name) {
        sim.kill_neuron(nid);
        return StatusCode::OK;
    }
    StatusCode::NOT_FOUND
}

#[cfg(feature = "server")]
async fn api_activate_neuron(
    State(state): State<Arc<AppState>>,
    Path((name, nid)): Path<(String, u32)>,
) -> StatusCode {
    let mut sims = state.simulations.lock().await;
    if let Some(sim) = sims.get_mut(&name) {
        sim.activate_neuron(nid);
        return StatusCode::OK;
    }
    StatusCode::NOT_FOUND
}

#[cfg(feature = "server")]
async fn api_sim_config(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<SimConfigUpdate>,
) -> StatusCode {
    let mut sims = state.simulations.lock().await;
    if let Some(sim) = sims.get_mut(&name) {
        if let Some(v) = req.stdp_enabled { sim.config.stdp_enabled = v; }
        if let Some(v) = req.structural_plasticity_enabled { sim.config.structural_plasticity_enabled = v; }
        if let Some(v) = req.modulation_enabled { sim.config.modulation_enabled = v; }
        if let Some(v) = req.spike_threshold_mv { sim.config.spike_threshold_mv = v; }
        if let Some(v) = req.max_syn_weight { sim.config.max_syn_weight = v; }
        return StatusCode::OK;
    }
    StatusCode::NOT_FOUND
}

#[cfg(feature = "server")]
async fn api_sim_step(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<RunResult>, StatusCode> {
    let mut sims = state.simulations.lock().await;
    let sim = sims.get_mut(&name).ok_or(StatusCode::NOT_FOUND)?;
    sim.step();
    let total_spikes: u64 = sim.db.neuron_states.iter().map(|s| s.spike_count as u64).sum();
    let v_min = sim.db.neuron_states.iter().map(|s| s.v_mem).fold(f32::INFINITY, f32::min);
    let v_max = sim.db.neuron_states.iter().map(|s| s.v_mem).fold(f32::NEG_INFINITY, f32::max);
    Ok(Json(RunResult { ticks: 1, total_spikes, v_min, v_max }))
}

#[cfg(feature = "server")]
async fn api_recently_fired(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<Vec<u32>>, StatusCode> {
    let sims = state.simulations.lock().await;
    let sim = sims.get(&name).ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(sim.recently_fired.clone()))
}

#[cfg(feature = "server")]
async fn api_graph(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<GraphData>, StatusCode> {
    // Try simulation first (has live state), fall back to raw database.
    let (neuron_attrs, neuron_states, regions, row_ptr, col_idx, syn_attrs, gap_junctions, meta) = {
        let sims = state.simulations.lock().await;
        if let Some(sim) = sims.get(&name) {
            let db = &sim.db;
            (
                db.neuron_attrs().to_vec(),
                db.neuron_states.clone(),
                db.regions().to_vec(),
                db.csr_row_ptr().to_vec(),
                db.csr_col_idx().to_vec(),
                db.syn_attrs().to_vec(),
                db.gap_junctions().to_vec(),
                db.meta.clone(),
            )
        } else {
            drop(sims);
            let dbs = state.databases.lock().await;
            let db = dbs.get(&name).ok_or(StatusCode::NOT_FOUND)?;
            (
                db.neuron_attrs().to_vec(),
                db.neuron_states.clone(),
                db.regions().to_vec(),
                db.csr_row_ptr().to_vec(),
                db.csr_col_idx().to_vec(),
                db.syn_attrs().to_vec(),
                db.gap_junctions().to_vec(),
                db.meta.clone(),
            )
        }
    };

    // Build region name lookup.
    let region_names: HashMap<u32, String> = meta.region_names.iter()
        .map(|(id, name)| (*id, name.clone()))
        .collect();

    // Build neuron type name lookup.
    let type_names: Vec<String> = meta.neuron_types.iter()
        .map(|t| t.type_name.clone())
        .collect();

    // Nodes.
    let nodes: Vec<GraphNode> = (0..neuron_attrs.len()).map(|i| {
        let attr = &neuron_attrs[i];
        let st = neuron_states.get(i);
        let label = if (attr.id as usize) < type_names.len() {
            type_names[attr.id as usize].clone()
        } else {
            format!("N{}", attr.id)
        };
        GraphNode {
            id: attr.id as u32,
            label,
            region: attr.region_id,
            v_mem: st.map(|s| s.v_mem).unwrap_or(-65.0),
            spike_count: st.map(|s| s.spike_count as u64).unwrap_or(0),
            group: attr.region_id,
            x: attr.x,
            y: attr.y,
            z: attr.z,
        }
    }).collect();

    // Chemical synapse edges (from CSR).
    let mut edges: Vec<GraphEdge> = Vec::new();
    for pre in 0..(row_ptr.len().saturating_sub(1)) as u32 {
        let s = row_ptr[pre as usize] as usize;
        let e = row_ptr[pre as usize + 1] as usize;
        for si in s..e {
            let post = col_idx[si] as u32;
            let w = syn_attrs.get(si).map(|a| a.base_weight).unwrap_or(0.0);
            edges.push(GraphEdge {
                source: pre,
                target: post,
                weight: w,
                edge_type: "chemical".into(),
            });
        }
    }

    // Gap junction edges.
    for gj in &gap_junctions {
        edges.push(GraphEdge {
            source: gj.pre_neuron,
            target: gj.post_neuron,
            weight: gj.weight,
            edge_type: "gap".into(),
        });
    }

    // Region info.
    let region_infos: Vec<RegionInfo> = regions.iter().map(|r| {
        let name = region_names.get(&r.id).cloned().unwrap_or_else(|| format!("region_{}", r.id));
        RegionInfo { id: r.id, name, neuron_count: r.neuron_count }
    }).collect();

    Ok(Json(GraphData { nodes, edges, regions: region_infos }))
}

#[cfg(feature = "server")]
async fn api_morphology(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<MorphologyData>, StatusCode> {
    let (neuron_attrs, comp_attrs, comp_states, neuron_states, regions, meta) = {
        let sims = state.simulations.lock().await;
        if let Some(sim) = sims.get(&name) {
            let db = &sim.db;
            (
                db.neuron_attrs().to_vec(),
                db.compartment_attrs().to_vec(),
                db.comp_states.clone(),
                db.neuron_states.clone(),
                db.regions().to_vec(),
                db.meta.clone(),
            )
        } else {
            drop(sims);
            let dbs = state.databases.lock().await;
            let db = dbs.get(&name).ok_or(StatusCode::NOT_FOUND)?;
            (
                db.neuron_attrs().to_vec(),
                db.compartment_attrs().to_vec(),
                db.comp_states.clone(),
                db.neuron_states.clone(),
                db.regions().to_vec(),
                db.meta.clone(),
            )
        }
    };

    let type_names: Vec<String> = meta.neuron_types.iter()
        .map(|t| t.type_name.clone())
        .collect();

    let region_names: HashMap<u32, String> = meta.region_names.iter()
        .map(|(id, name)| (*id, name.clone()))
        .collect();

    // Build neurons with their compartment trees.
    let mut neurons = Vec::new();
    for (i, attr) in neuron_attrs.iter().enumerate() {
        let nid = attr.id as u32;
        let first = attr.first_comp_id as usize;
        let n_comp = attr.n_compartment as usize;

        let label = if (attr.neuron_type as usize) < type_names.len() {
            type_names[attr.neuron_type as usize].clone()
        } else {
            format!("N{}", attr.id)
        };

        let st = neuron_states.get(i);
        let v_mem = st.map(|s| s.v_mem).unwrap_or(-65.0);
        let spike_count = st.map(|s| s.spike_count as u64).unwrap_or(0);

        let mut compartments = Vec::new();
        for ci in first..(first + n_comp) {
            if let Some(ca) = comp_attrs.get(ci) {
                let v = comp_states.get(ci).map(|s| s.v_mem).unwrap_or(-65.0);
                compartments.push(CompartmentInfo {
                    id: ca.id,
                    comp_type: ca.comp_type,
                    parent_comp_id: ca.parent_comp_id,
                    x: ca.x,
                    y: ca.y,
                    z: ca.z,
                    diameter: ca.diameter,
                    length: ca.length,
                    v_mem: v,
                });
            }
        }

        neurons.push(NeuronMorphology {
            neuron_id: nid,
            label,
            region: attr.region_id,
            v_mem,
            spike_count,
            compartments,
        });
    }

    let region_infos: Vec<RegionInfo> = regions.iter().map(|r| {
        let name = region_names.get(&r.id).cloned().unwrap_or_else(|| format!("region_{}", r.id));
        RegionInfo { id: r.id, name, neuron_count: r.neuron_count }
    }).collect();

    Ok(Json(MorphologyData { neurons, regions: region_infos }))
}

// ── Dashboard HTML (embedded) ───────────────────────────────────────────

#[cfg(feature = "server")]
async fn dashboard() -> axum::response::Html<String> {
    axum::response::Html(DASHBOARD_HTML.to_string())
}

#[cfg(feature = "server")]
const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BrainDB — 3D Neural Morphology</title>
<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}
</script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0e1a;color:#e2e8f0;overflow:hidden}
.header{background:#131a2e;padding:10px 20px;border-bottom:1px solid #1e2d4a;display:flex;align-items:center;gap:12px;z-index:10;position:relative}
.header h1{font-size:17px;color:#7dd3fc}
.header .badge{background:#0284c7;color:#0a0e1a;padding:2px 8px;border-radius:9999px;font-size:10px;font-weight:700}
.sidebar{position:fixed;right:0;top:42px;width:340px;height:calc(100vh - 42px);background:#131a2e;border-left:1px solid #1e2d4a;padding:14px;overflow-y:auto;z-index:10}
.sidebar h2{font-size:11px;color:#64748b;margin:10px 0 6px;text-transform:uppercase;letter-spacing:.8px}
.sidebar h2:first-child{margin-top:0}
input,select,button{background:#0a0e1a;border:1px solid #1e2d4a;color:#e2e8f0;padding:5px 9px;border-radius:5px;font-size:12px}
button{cursor:pointer;background:#0284c7;border:none;font-weight:600;transition:background .2s}
button:hover{background:#38bdf8}
button.sm{padding:3px 7px;font-size:10px}
.row{display:flex;gap:5px;align-items:center;margin-bottom:5px}
label{font-size:11px;color:#64748b;min-width:65px}
.stat{display:inline-block;background:#0a0e1a;padding:2px 8px;border-radius:3px;margin:1px;font-size:11px}
.stat b{color:#7dd3fc}
#three-canvas{position:fixed;top:42px;left:0;width:calc(100vw - 340px);height:calc(100vh - 42px);background:#0a0e1a}
#tooltip{position:absolute;background:#131a2e;border:1px solid #1e2d4a;border-radius:6px;padding:8px;font-size:11px;pointer-events:none;z-index:20;display:none;max-width:240px}
#tooltip .tt-title{color:#7dd3fc;font-weight:600;margin-bottom:3px}
#tooltip .tt-row{color:#64748b}
#tooltip .tt-row span{color:#e2e8f0}
#log{padding:6px;background:#0a0e1a;border-radius:5px;font-family:monospace;font-size:10px;max-height:100px;overflow-y:auto;white-space:pre-wrap;color:#64748b}
.filter-row{display:flex;gap:3px;flex-wrap:wrap;margin-bottom:5px}
.filter-btn{padding:2px 7px;border-radius:3px;font-size:10px;cursor:pointer;border:1px solid #1e2d4a;background:#0a0e1a;color:#e2e8f0}
.filter-btn.active{background:#0284c7;color:#0a0e1a;border-color:#0284c7}
.morpho-legend{display:flex;gap:8px;margin-top:4px}
.morpho-legend span{font-size:10px;display:flex;align-items:center;gap:3px}
.morpho-legend .dot{width:10px;height:4px;border-radius:2px;display:inline-block}
#detail-canvas{width:100%;height:260px;background:#0a0e1a;border:1px solid #1e2d4a;border-radius:6px;margin-top:6px}
</style>
</head>
<body>
<div class="header">
  <h1>🧠 BrainDB</h1>
  <span class="badge">3D Morphology</span>
  <div style="flex:1"></div>
  <input id="db-path" value="C:/Users/Administrator/celegans.braindb" style="width:240px">
  <button id="btn-open">Open & Visualize</button>
</div>
<canvas id="three-canvas"></canvas>
<div class="sidebar">
  <h2>Network Stats</h2>
  <div id="db-stats">No database loaded.</div>
  <h2>Morphology Legend</h2>
  <div class="morpho-legend">
    <span><span class="dot" style="background:#7dd3fc"></span> Soma</span>
    <span><span class="dot" style="background:#c084fc"></span> Dendrite</span>
    <span><span class="dot" style="background:#f97316"></span> Axon</span>
    <span><span class="dot" style="background:#22c55e"></span> Gap Jct</span>
  </div>
  <h2>Region Filter</h2>
  <div id="region-filters" class="filter-row"></div>
  <h2>Edge Filter</h2>
  <div class="filter-row">
    <div class="filter-btn active" data-type="chemical" id="filter-chemical">Chemical</div>
    <div class="filter-btn active" data-type="gap" id="filter-gap">Gap Junction</div>
  </div>
  <h2>View</h2>
  <div class="row"><button class="sm" id="btn-reset">⌂ Reset View</button><button class="sm" id="btn-focus">⊕ Focus</button></div>
  <h2>Simulation</h2>
  <div class="row"><label>Duration:</label><input id="run-dur" value="100" style="width:60px"> ms</div>
  <div class="row"><label>Stimulus:</label><input id="stim" placeholder="0:30,1:20" style="flex:1"></div>
  <div class="row"><label><input type="checkbox" id="chk-stdp"> STDP</label></div>
  <div class="row"><button id="btn-run">▶ Run</button><button class="sm" id="btn-refresh">↻ Refresh</button></div>
  <h2>Selected Neuron</h2>
  <div id="neuron-detail" style="font-size:11px;color:#64748b">Click a neuron to inspect.</div>
  <canvas id="detail-canvas"></canvas>
  <h2>Log</h2>
  <div id="log">Ready.</div>
</div>
<div id="tooltip"><div class="tt-title"></div><div class="tt-row"></div></div>

<script type="module">
import*as THREE from'three';
import{OrbitControls}from'three/addons/controls/OrbitControls.js';

// ── Globals ────────────────────────────────────────────────────────────
let currentDb=null, graphData=null, morphData=null;
let edgeFilters=new Set(['chemical','gap']), activeRegions=new Set();
let scene,camera,renderer,controls,raycaster,mouse;
let selectedNeuron=null;
let detailScene,detailCamera,detailRenderer;
// Instanced rendering objects
let somaInstMesh=null, spikeInstMesh=null, glowInstMesh=null;
let branchGroup=null, edgeGroup=null;
let neuronInfo=[]; // [{id,x,y,z,region,vMem,spikes,label}, ...]
let nodeMap=new Map();

const RC=['#0ea5e9','#f97316','#a855f7','#22c55e','#ef4444','#eab308','#ec4899','#14b8a6','#6366f1','#f43f5e'];
const COMP_COLORS={0:0x7dd3fc,1:0xc084fc,2:0xa78bfa,3:0xf97316,255:0x64748b};
const COMP_NAMES={0:'Soma',1:'Apical Dendrite',2:'Basal Dendrite',3:'Axon',255:'Other'};
const MAX_N=2048;
const tmpObj=new THREE.Object3D();
const tmpColor=new THREE.Color();

// ── Helpers ────────────────────────────────────────────────────────────
window.log=function(m){const l=document.getElementById('log');l.textContent+=m+'\n';l.scrollTop=l.scrollHeight};
async function api(method,path,body){
  const opts={method,headers:{'Content-Type':'application/json'}};
  if(body)opts.body=JSON.stringify(body);
  try{const r=await fetch('/api'+path,opts);
    if(r.status!==200){log('API '+method+' '+path+' → '+r.status);return null}
    const t=await r.text();if(!t)return null;try{return JSON.parse(t)}catch(e){log('Parse error');return null}
  }catch(e){log('Error: '+e.message);return null}
}

// ── Init Three.js ──────────────────────────────────────────────────────
function initThree(){
  const canvas=document.getElementById('three-canvas');
  const W=canvas.clientWidth, H=canvas.clientHeight;
  scene=new THREE.Scene();
  scene.background=new THREE.Color(0x0a0e1a);
  scene.fog=new THREE.FogExp2(0x0a0e1a,0.001);
  camera=new THREE.PerspectiveCamera(50,W/H,0.1,50000);
  camera.position.set(0,0,500);
  renderer=new THREE.WebGLRenderer({canvas,antialias:true});
  renderer.setSize(W,H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  controls=new OrbitControls(camera,canvas);
  controls.enableDamping=true;controls.dampingFactor=0.08;
  controls.rotateSpeed=0.8;controls.zoomSpeed=1.2;
  scene.add(new THREE.AmbientLight(0x4466aa,0.6));
  const dir=new THREE.DirectionalLight(0xffffff,0.8);
  dir.position.set(200,300,400);scene.add(dir);
  const dir2=new THREE.DirectionalLight(0x4488ff,0.3);
  dir2.position.set(-200,-100,-200);scene.add(dir2);
  raycaster=new THREE.Raycaster();
  raycaster.params.Points={threshold:3};
  mouse=new THREE.Vector2();
  canvas.addEventListener('click',onCanvasClick);
  canvas.addEventListener('mousemove',onCanvasMove);
  // Detail view
  const dc=document.getElementById('detail-canvas');
  detailScene=new THREE.Scene();detailScene.background=new THREE.Color(0x0a0e1a);
  detailCamera=new THREE.PerspectiveCamera(50,dc.clientWidth/dc.clientHeight,0.01,5000);
  detailCamera.position.set(0,0,100);
  detailRenderer=new THREE.WebGLRenderer({canvas:dc,antialias:true});
  detailRenderer.setSize(dc.clientWidth,dc.clientHeight);
  detailScene.add(new THREE.AmbientLight(0x4466aa,0.6));
  const dd=new THREE.DirectionalLight(0xffffff,0.8);dd.position.set(50,80,100);detailScene.add(dd);
  window.addEventListener('resize',()=>{
    const w=canvas.clientWidth,h=canvas.clientHeight;
    camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);
  });
  animate();
}

function animate(){
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene,camera);
  if(detailScene.children.length>2)detailRenderer.render(detailScene,detailCamera);
}

// ── Build scene with InstancedMesh (high performance) ─────────────────
function buildScene(){
  if(!graphData)return;
  // Clear old instanced objects
  if(somaInstMesh)scene.remove(somaInstMesh);
  if(spikeInstMesh)scene.remove(spikeInstMesh);
  if(glowInstMesh)scene.remove(glowInstMesh);
  if(branchGroup)scene.remove(branchGroup);
  if(edgeGroup)scene.remove(edgeGroup);

  const visNodes=graphData.nodes.filter(n=>activeRegions.has(n.region));
  const nidSet=new Set(visNodes.map(n=>n.id));
  const visEdges=graphData.edges.filter(e=>edgeFilters.has(e.type)&&nidSet.has(e.source)&&nidSet.has(e.target));

  // Morphology lookup
  const morphMap=new Map();
  if(morphData)morphData.neurons.forEach(n=>morphMap.set(n.neuron_id,n));

  // Center
  let cx=0,cy=0,cz=0;
  visNodes.forEach(n=>{cx+=n.x;cy+=n.y;cz+=n.z});
  if(visNodes.length>0){cx/=visNodes.length;cy/=visNodes.length;cz/=visNodes.length}

  const nCount=visNodes.length;
  const morphScale=nCount<=30?1.5:nCount<=100?1:nCount<=300?0.6:0.3;
  const s=morphScale;
  const somaR=2.5*s;

  // Build neuronInfo array
  neuronInfo=[];
  nodeMap=new Map();
  visNodes.forEach((n,i)=>{
    const info={id:n.id,x:n.x-cx,y:n.y-cy,z:n.z-cz,region:n.region,vMem:n.v_mem,spikes:n.spike_count,label:n.label};
    neuronInfo.push(info);
    nodeMap.set(n.id,{...info,idx:i});
  });

  // ── Soma InstancedMesh (1 draw call for all neurons) ──
  const somaGeo=new THREE.SphereGeometry(somaR,8,6);
  const somaMat=new THREE.MeshPhongMaterial({vertexColors:true,transparent:true,opacity:0.7,shininess:60});
  somaInstMesh=new THREE.InstancedMesh(somaGeo,somaMat,Math.max(1,nCount));
  somaInstMesh.count=nCount;
  // Set positions and colors
  neuronInfo.forEach((n,i)=>{
    tmpObj.position.set(n.x,n.y,n.z);
    tmpObj.updateMatrix();
    somaInstMesh.setMatrixAt(i,tmpObj.matrix);
    const regionColor=parseInt(RC[n.region%RC.length].replace('#','0x'));
    tmpColor.setHex(regionColor);
    somaInstMesh.setColorAt(i,tmpColor);
  });
  somaInstMesh.instanceMatrix.needsUpdate=true;
  somaInstMesh.instanceColor.needsUpdate=true;
  somaInstMesh.userData={type:'soma'};
  scene.add(somaInstMesh);

  // ── Spike glow InstancedMesh (yellow, only for spiking neurons) ──
  const spikeGeo=new THREE.SphereGeometry(somaR*0.5,6,4);
  const spikeMat=new THREE.MeshBasicMaterial({vertexColors:true,transparent:true,opacity:0.8});
  spikeInstMesh=new THREE.InstancedMesh(spikeGeo,spikeMat,Math.max(1,nCount));
  spikeInstMesh.count=0; // start with 0, updateSpikeVisuals will set
  spikeInstMesh.userData={type:'spike'};
  scene.add(spikeInstMesh);

  // ── Active glow InstancedMesh (region-colored halo) ──
  const glowGeo=new THREE.SphereGeometry(somaR*1.8,6,4);
  const glowMat=new THREE.MeshBasicMaterial({vertexColors:true,transparent:true,opacity:0.12,side:THREE.BackSide});
  glowInstMesh=new THREE.InstancedMesh(glowGeo,glowMat,Math.max(1,nCount));
  glowInstMesh.count=0;
  glowInstMesh.userData={type:'glow'};
  scene.add(glowInstMesh);

  // ── Neurite branches (merged geometry per neuron) ──
  branchGroup=new THREE.Group();
  const branchGeo=new THREE.CylinderGeometry(1,1,1,4,1); // unit cylinder, scaled per instance
  visNodes.forEach(n=>{
    const mn=morphMap.get(n.id);
    if(!mn||!mn.compartments||mn.compartments.length<=1)return;
    const pos=nodeMap.get(n.id);
    const comps=mn.compartments;
    const compMap=new Map();comps.forEach(c=>compMap.set(c.id,c));
    const somas=comps.filter(c=>c.comp_type===0);
    let ox=0,oy=0,oz=0;
    if(somas.length>0){somas.forEach(c=>{ox+=c.x;oy+=c.y;oz+=c.z});ox/=somas.length;oy/=somas.length;oz/=somas.length}
    comps.forEach(c=>{
      if(c.comp_type===0)return;
      const parent=compMap.get(c.parent_comp_id);
      if(!parent)return;
      const x1=(parent.x-ox)*s,y1=(parent.y-oy)*s,z1=(parent.z-oz)*s;
      const x2=(c.x-ox)*s,y2=(c.y-oy)*s,z2=(c.z-oz)*s;
      const dx=x2-x1,dy=y2-y1,dz=z2-z1;
      const len=Math.sqrt(dx*dx+dy*dy+dz*dz);
      if(len<0.01)return;
      const r1=Math.max(0.15,parent.diameter*s*0.5);
      const r2=Math.max(0.1,c.diameter*s*0.5);
      const color=COMP_COLORS[c.comp_type]||0x64748b;
      const mat=new THREE.MeshPhongMaterial({color,transparent:true,opacity:0.5,shininess:20});
      const mesh=new THREE.Mesh(branchGeo,mat);
      mesh.scale.set(r2*2,len,r1*2); // scale unit cylinder
      mesh.position.set((x1+x2)/2+pos.x,(y1+y2)/2+pos.y,(z1+z2)/2+pos.z);
      const dir=new THREE.Vector3(dx,dy,dz).normalize();
      const quat=new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,1,0),dir);
      mesh.quaternion.copy(quat);
      branchGroup.add(mesh);
    });
  });
  scene.add(branchGroup);

  // ── Edges (2 LineSegments = 2 draw calls) ──
  edgeGroup=new THREE.Group();
  const gapVerts=[],chemVerts=[];
  visEdges.forEach(e=>{
    const src=nodeMap.get(e.source),tgt=nodeMap.get(e.target);
    if(!src||!tgt)return;
    const arr=e.type==='gap'?gapVerts:chemVerts;
    arr.push(src.x,src.y,src.z,tgt.x,tgt.y,tgt.z);
  });
  if(chemVerts.length>0){
    const geo=new THREE.BufferGeometry();
    geo.setAttribute('position',new THREE.Float32BufferAttribute(chemVerts,3));
    edgeGroup.add(new THREE.LineSegments(geo,new THREE.LineBasicMaterial({color:0x334155,transparent:true,opacity:0.15})));
  }
  if(gapVerts.length>0){
    const geo=new THREE.BufferGeometry();
    geo.setAttribute('position',new THREE.Float32BufferAttribute(gapVerts,3));
    edgeGroup.add(new THREE.LineSegments(geo,new THREE.LineBasicMaterial({color:0x22c55e,transparent:true,opacity:0.2})));
  }
  scene.add(edgeGroup);

  // Auto-fit camera
  let maxDist=0;
  visNodes.forEach(n=>{
    const d=Math.sqrt((n.x-cx)**2+(n.y-cy)**2+(n.z-cz)**2);
    if(d>maxDist)maxDist=d;
  });
  camera.position.set(0,0,maxDist*1.8+50);
  controls.target.set(0,0,0);
  controls.update();

  // Initial visual update
  updateNeuronVisuals();
  log('3D rendered: '+visNodes.length+' neurons, '+visEdges.length+' edges (InstancedMesh)');
}

// ── Update neuron visuals (firing state) without rebuilding scene ──
function updateNeuronVisuals(){
  if(!somaInstMesh||neuronInfo.length===0)return;
  let spikeCount=0, glowCount=0;
  neuronInfo.forEach((n,i)=>{
    const isActive=n.vMem>-55;
    const hasSpiked=n.spikes>0;
    const regionColor=parseInt(RC[n.region%RC.length].replace('#','0x'));
    // Update soma color: bright if active, dim if resting
    if(hasSpiked){
      tmpColor.setHex(0xfef08a); // yellow for spiking
    }else if(isActive){
      tmpColor.setHex(regionColor);
    }else{
      tmpColor.setHex(regionColor); tmpColor.multiplyScalar(0.5);
    }
    somaInstMesh.setColorAt(i,tmpColor);
    // Spike glow
    if(hasSpiked){
      tmpObj.position.set(n.x,n.y,n.z);
      tmpObj.scale.set(1,1,1);
      tmpObj.updateMatrix();
      spikeInstMesh.setMatrixAt(spikeCount,tmpObj.matrix);
      tmpColor.setHex(0xfef08a);
      spikeInstMesh.setColorAt(spikeCount,tmpColor);
      spikeCount++;
    }
    // Active glow
    if(isActive){
      tmpObj.position.set(n.x,n.y,n.z);
      tmpObj.scale.set(1,1,1);
      tmpObj.updateMatrix();
      glowInstMesh.setMatrixAt(glowCount,tmpObj.matrix);
      tmpColor.setHex(regionColor);
      glowInstMesh.setColorAt(glowCount,tmpColor);
      glowCount++;
    }
  });
  somaInstMesh.instanceColor.needsUpdate=true;
  spikeInstMesh.count=spikeCount;
  if(spikeCount>0){spikeInstMesh.instanceMatrix.needsUpdate=true;spikeInstMesh.instanceColor.needsUpdate=true}
  glowInstMesh.count=glowCount;
  if(glowCount>0){glowInstMesh.instanceMatrix.needsUpdate=true;glowInstMesh.instanceColor.needsUpdate=true}
}

// ── Fast state update (only update vMem/spikes, no scene rebuild) ──
function updateFromGraph(){
  if(!graphData||!somaInstMesh)return;
  const visIds=new Set(neuronInfo.map(n=>n.id));
  graphData.nodes.forEach(n=>{
    if(!visIds.has(n.id))return;
    const info=nodeMap.get(n.id);
    if(!info)return;
    info.vMem=n.v_mem;
    info.spikes=n.spike_count;
    const ni=neuronInfo[info.idx];
    if(ni){ni.vMem=n.v_mem;ni.spikes=n.spike_count}
  });
  updateNeuronVisuals();
}

// ── Picking ───────────────────────────────────────────────────────────
function onCanvasClick(e){
  const canvas=document.getElementById('three-canvas');
  const rect=canvas.getBoundingClientRect();
  mouse.x=((e.clientX-rect.left)/rect.width)*2-1;
  mouse.y=-((e.clientY-rect.top)/rect.height)*2+1;
  raycaster.setFromCamera(mouse,camera);
  if(!somaInstMesh)return;
  const hits=raycaster.intersectObject(somaInstMesh);
  if(hits.length>0){
    const idx=hits[0].instanceId;
    const n=neuronInfo[idx];
    if(n){
      const node=graphData.nodes.find(nd=>nd.id===n.id);
      if(node)selectNeuron(node);
    }
  }
}

function onCanvasMove(e){
  const canvas=document.getElementById('three-canvas');
  const rect=canvas.getBoundingClientRect();
  mouse.x=((e.clientX-rect.left)/rect.width)*2-1;
  mouse.y=-((e.clientY-rect.top)/rect.height)*2+1;
  raycaster.setFromCamera(mouse,camera);
  const tt=document.getElementById('tooltip');
  if(!somaInstMesh){tt.style.display='none';return}
  const hits=raycaster.intersectObject(somaInstMesh);
  if(hits.length>0){
    const idx=hits[0].instanceId;
    const n=neuronInfo[idx];
    if(n){
      const regionName=graphData.regions.find(r=>r.id===n.region)?.name||'region_'+n.region;
      tt.style.display='block';
      tt.style.left=(e.clientX+10)+'px';tt.style.top=(e.clientY-8)+'px';
      tt.querySelector('.tt-title').textContent=n.label+' (#'+n.id+')';
      tt.querySelector('.tt-row').innerHTML=
        'V<sub>mem</sub>: <span>'+n.vMem.toFixed(1)+' mV</span><br>'+
        'Spikes: <span>'+n.spikes+'</span><br>'+
        'Region: <span>'+regionName+'</span>';
    }
  }else{tt.style.display='none'}
}

// ── Select neuron & detail view ────────────────────────────────────────
function selectNeuron(d){
  selectedNeuron=d;
  const mn=morphData?morphData.neurons.find(n=>n.neuron_id===d.id):null;
  const regionName=graphData.regions.find(r=>r.id===d.region)?.name||d.region;
  let html='<b style="color:#7dd3fc">'+d.label+'</b> (#'+d.id+')<br>'+
    'V<sub>mem</sub>: '+d.v_mem.toFixed(2)+' mV<br>'+
    'Spikes: '+d.spike_count+'<br>'+
    'Region: '+regionName;
  if(mn){
    html+='<br>Compartments: '+mn.compartments.length;
    const sc=mn.compartments.filter(c=>c.comp_type===0).length;
    const dc=mn.compartments.filter(c=>c.comp_type===1||c.comp_type===2).length;
    const ac=mn.compartments.filter(c=>c.comp_type===3).length;
    html+='<br><span style="color:#7dd3fc">Soma:'+sc+'</span> <span style="color:#c084fc">Dend:'+dc+'</span> <span style="color:#f97316">Axon:'+ac+'</span>';
  }
  document.getElementById('neuron-detail').innerHTML=html;
  renderDetailMorphology(d.id);
}

function renderDetailMorphology(neuronId){
  while(detailScene.children.length>2)detailScene.remove(detailScene.lastChild);
  const mn=morphData?morphData.neurons.find(n=>n.neuron_id===neuronId):null;
  if(!mn||!mn.compartments||mn.compartments.length<=1)return;
  const comps=mn.compartments;
  const compMap=new Map();comps.forEach(c=>compMap.set(c.id,c));
  let cx=0,cy=0,cz=0;
  comps.forEach(c=>{cx+=c.x;cy+=c.y;cz+=c.z});
  cx/=comps.length;cy/=comps.length;cz/=comps.length;
  const group=new THREE.Group();
  // Soma
  const somas=comps.filter(c=>c.comp_type===0);
  let somaR=5;
  if(somas.length>0)somaR=Math.max(3,somas.reduce((a,c)=>a+c.diameter,0)/somas.length*0.5);
  const somaGeo=new THREE.SphereGeometry(somaR,12,10);
  const somaMat=new THREE.MeshPhongMaterial({color:0x7dd3fc,emissive:0x112233,transparent:true,opacity:0.8,shininess:60});
  group.add(new THREE.Mesh(somaGeo,somaMat));
  // Branches
  const unitCyl=new THREE.CylinderGeometry(1,1,1,6,1);
  comps.forEach(c=>{
    if(c.comp_type===0)return;
    const parent=compMap.get(c.parent_comp_id);
    if(!parent)return;
    const x1=parent.x-cx,y1=parent.y-cy,z1=parent.z-cz;
    const x2=c.x-cx,y2=c.y-cy,z2=c.z-cz;
    const dx=x2-x1,dy=y2-y1,dz=z2-z1;
    const len=Math.sqrt(dx*dx+dy*dy+dz*dz);
    if(len<0.01)return;
    const r1=Math.max(0.3,parent.diameter*0.5);
    const r2=Math.max(0.2,c.diameter*0.5);
    const color=COMP_COLORS[c.comp_type]||0x64748b;
    const mat=new THREE.MeshPhongMaterial({color,transparent:true,opacity:0.7,shininess:30});
    const mesh=new THREE.Mesh(unitCyl,mat);
    mesh.scale.set(r2*2,len,r1*2);
    mesh.position.set((x1+x2)/2,(y1+y2)/2,(z1+z2)/2);
    const dir=new THREE.Vector3(dx,dy,dz).normalize();
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir);
    group.add(mesh);
  });
  detailScene.add(group);
  // Fit camera
  let maxD=0;
  comps.forEach(c=>{const d=Math.sqrt((c.x-cx)**2+(c.y-cy)**2+(c.z-cz)**2);if(d>maxD)maxD=d});
  detailCamera.position.set(0,0,maxD*1.5+20);
  detailCamera.lookAt(0,0,0);
}

// ── Dashboard actions ─────────────────────────────────────────────────
window.openAndLoad=async function(){
  const path=document.getElementById('db-path').value;
  if(!path){log('Enter a database path');return}
  const info=await api('POST','/db/open',{path});
  if(!info)return;
  currentDb=path.split(/[\\/]/).pop().replace('.braindb','');
  document.getElementById('db-stats').innerHTML=
    '<span class="stat">Neurons: <b>'+info.n_neurons+'</b></span>'+
    '<span class="stat">Synapses: <b>'+info.n_synapses+'</b></span>'+
    '<span class="stat">Gap Jct: <b>'+info.n_gap_junctions+'</b></span>'+
    '<span class="stat">Compartments: <b>'+info.n_compartments+'</b></span>'+
    '<span class="stat">Regions: <b>'+info.n_regions+'</b></span>';
  await api('POST','/sim/'+encodeURIComponent(currentDb)+'/init');
  loadData();
};
async function loadData(){
  if(!currentDb)return;
  log('Loading 3D data...');
  const[gd,md]=await Promise.all([
    api('GET','/graph/'+encodeURIComponent(currentDb)),
    api('GET','/morphology/'+encodeURIComponent(currentDb))
  ]);
  if(!gd){log('ERROR: Failed to load graph');return}
  graphData=gd;morphData=md;
  log('Graph: '+graphData.nodes.length+' nodes, '+graphData.edges.length+' edges');
  if(morphData)log('Morphology: '+morphData.neurons.length+' neurons');
  if(graphData.regions.length===0){
    const rmap=new Map();
    graphData.nodes.forEach(n=>{if(!rmap.has(n.region))rmap.set(n.region,0);rmap.set(n.region,rmap.get(n.region)+1)});
    graphData.regions=Array.from(rmap.entries()).map(([id,count])=>({id,name:'Region '+id,neuron_count:count}));
  }
  activeRegions=new Set(graphData.regions.map(r=>r.id));
  const rf=document.getElementById('region-filters');
  rf.innerHTML='';
  graphData.regions.forEach(r=>{
    const b=document.createElement('div');
    b.className='filter-btn active';
    b.dataset.region=r.id;
    b.textContent=r.name+' ('+r.neuron_count+')';
    b.onclick=function(){
      if(activeRegions.has(r.id)){activeRegions.delete(r.id);b.classList.remove('active')}
      else{activeRegions.add(r.id);b.classList.add('active')}
      buildScene();
    };
    rf.appendChild(b);
  });
  buildScene();
}
window.toggleEdgeFilter=function(el){
  const t=el.dataset.type;
  if(edgeFilters.has(t)){edgeFilters.delete(t);el.classList.remove('active')}
  else{edgeFilters.add(t);el.classList.add('active')}
  buildScene();
};

// ── Camera helpers ─────────────────────────────────────────────────────
window.resetCamera=function(){
  if(!graphData)return;
  let cx=0,cy=0,cz=0,maxDist=0;
  const vis=graphData.nodes.filter(n=>activeRegions.has(n.region));
  vis.forEach(n=>{cx+=n.x;cy+=n.y;cz+=n.z});
  if(vis.length>0){cx/=vis.length;cy/=vis.length;cz/=vis.length}
  vis.forEach(n=>{const d=Math.sqrt((n.x-cx)**2+(n.y-cy)**2+(n.z-cz)**2);if(d>maxDist)maxDist=d});
  camera.position.set(0,0,maxDist*1.8+50);
  controls.target.set(0,0,0);
  controls.update();
};

window.focusSelected=function(){
  if(!selectedNeuron)return;
  const info=nodeMap.get(selectedNeuron.id);
  if(!info)return;
  controls.target.set(info.x,info.y,info.z);
  camera.position.set(info.x+50,info.y+50,info.z+150);
  controls.update();
};

// ── Simulation with real-time firing animation ────────────────────────
window.runSim=async function(){
  if(!currentDb){log('Open a database first!');return}
  const dur=parseInt(document.getElementById('run-dur').value)||100;
  const stimStr=document.getElementById('stim').value;
  const stimulus={};
  if(stimStr)stimStr.split(',').forEach(p=>{const[n,c]=p.split(':');stimulus[parseInt(n)]=parseFloat(c)});
  log('Running '+dur+' ms...');
  const result=await api('POST','/sim/'+encodeURIComponent(currentDb)+'/run',{
    duration_ms:dur,stimulus,stdp:document.getElementById('chk-stdp').checked,structural_plasticity:false
  });
  if(result)log('Done: '+result.total_spikes+' spikes, V=['+result.v_min.toFixed(1)+', '+result.v_max.toFixed(1)+'] mV');
  // Fast update: only refresh neuron states, no scene rebuild
  const gd=await api('GET','/graph/'+encodeURIComponent(currentDb));
  if(gd){graphData=gd;updateFromGraph()}
};

window.refreshScene=async function(){
  if(!currentDb)return;
  const[gd,md]=await Promise.all([
    api('GET','/graph/'+encodeURIComponent(currentDb)),
    api('GET','/morphology/'+encodeURIComponent(currentDb))
  ]);
  graphData=gd;morphData=md;
  if(graphData)buildScene();
};

// ── Event bindings (ES module scope requires addEventListener) ────────
document.getElementById('btn-open').addEventListener('click',()=>window.openAndLoad());
document.getElementById('btn-run').addEventListener('click',()=>window.runSim());
document.getElementById('btn-refresh').addEventListener('click',()=>window.refreshScene());
document.getElementById('btn-reset').addEventListener('click',()=>window.resetCamera());
document.getElementById('btn-focus').addEventListener('click',()=>window.focusSelected());
document.getElementById('filter-chemical').addEventListener('click',function(){window.toggleEdgeFilter(this)});
document.getElementById('filter-gap').addEventListener('click',function(){window.toggleEdgeFilter(this)});

// ── Boot ───────────────────────────────────────────────────────────────
initThree();
</script>
</body>
</html>"#;

// ── Main ────────────────────────────────────────────────────────────────

#[cfg(feature = "server")]
#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let state = Arc::new(AppState {
        databases: Mutex::new(HashMap::new()),
        simulations: Mutex::new(HashMap::new()),
    });

    let api_routes = Router::new()
        .route("/dbs", get(api_list_dbs))
        .route("/db/open", post(api_open_db))
        .route("/db/:name/info", get(api_db_info))
        .route("/sim/:name/init", post(api_init_simulation))
        .route("/sim/:name/run", post(api_run_simulation))
        .route("/sim/:name/step", post(api_sim_step))
        .route("/sim/:name/neurons", get(api_neuron_states))
        .route("/sim/:name/stimulus", post(api_set_stimulus))
        .route("/sim/:name/snapshot/:snap_path", post(api_snapshot))
        .route("/sim/:name/config", post(api_sim_config))
        .route("/sim/:name/recently_fired", get(api_recently_fired))
        .route("/sim/:name/kill/:nid", post(api_kill_neuron))
        .route("/sim/:name/activate/:nid", post(api_activate_neuron))
        .route("/graph/:name", get(api_graph))
        .route("/morphology/:name", get(api_morphology))
        .route("/neuron/:name/:nid", get(api_neuron_detail))
        .route("/query/:name/neighbors", get(api_query_neighbors))
        .route("/query/:name/upstream", get(api_query_upstream))
        .route("/query/:name/strongest_path", get(api_query_strongest_path))
        .route("/query/:name/region_pathway", get(api_query_region_pathway))
        .route("/query/:name/region_lfp", get(api_query_region_lfp))
        .with_state(state);

    let app = Router::new()
        .route("/", get(dashboard))
        .nest("/api", api_routes)
        .layer(CorsLayer::permissive());

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], 3000));
    info!("🧠 BrainDB Server listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[cfg(not(feature = "server"))]
fn main() {
    eprintln!("braindb-server requires the 'server' feature. Build with:");
    eprintln!("  cargo build --features server");
}
