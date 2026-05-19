//! BAAIWorm connectome loader (M4).
//!
//! Reads the BAAIWorm project directory structure:
//!
//! - `components/param/cell/*.json` — per-neuron biophysical parameters (17 conductances)
//! - `components/param/connection/SI5-302.xlsx` — chemical & gap-junction adjacency matrices
//! - `network/config.json` — cell name ↔ numeric ID mapping
//!
//! And converts them into a [`BrainDB`] via [`BrainDBBuilder`].
//!
//! Key design decisions (v2.4 §6.1, §15.3–15.4):
//! - Each of the 302 neurons gets its own `NeuronTypeParams` (unique conductances).
//! - Most *C. elegans* neurons use `NeuronModel::Graded`; a few use `HodgkinHuxley`.
//! - All chemical synapses use `SYN_MODE_CONTINUOUS` (Sigmoid + single-exponential).
//! - Each neuron has 2 compartments: soma + neurite.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use rand::SeedableRng;

use calamine::DataType as CalData;

use crate::core::compartment::{CompartmentAttr, CompType};
use crate::core::gap_junction::GapJunction;
use crate::core::ion_channel::{ChannelConductance, IonChannelDef, IonChannelSet, IonType};
use crate::core::neuron::{NeuronAttr, NEURON_ALIVE};
use crate::core::neuron_type::{NeuronModel, NeuronTypeParams};
use crate::core::receptor::ReceptorParams;
use crate::core::synapse::{
    SynapseAttr, SYN_EXCITATORY, SYN_INHIBITORY, SYN_MODE_CONTINUOUS,
};
use crate::error::{BrainDBError, Result};
use crate::storage::builder::BrainDBBuilder;
use crate::storage::mmap_db::BrainDB;

// ── JSON deserialization structs ──────────────────────────────────────────

/// Per-neuron JSON file format (e.g. `ADAL.json`).
#[derive(Clone, Debug, Deserialize)]
struct CellParamJson {
    soma: SomaParams,
    #[serde(default)]
    neurite: NeuriteParams,
}

#[derive(Clone, Debug, Deserialize)]
struct SomaParams {
    #[serde(rename = "Ra")]
    ra: f64,
    #[serde(rename = "cm")]
    cm: f64,
    #[serde(rename = "gpas")]
    gpas: f64,
    #[serde(rename = "epas")]
    epas: f64,
    #[serde(rename = "gbshl1")]
    gbshl1: f64,
    #[serde(rename = "gbshk1")]
    gbshk1: f64,
    #[serde(rename = "gbkvs1")]
    gbkvs1: f64,
    #[serde(rename = "gbegl2")]
    gbegl2: f64,
    #[serde(rename = "gbegl36")]
    gbegl36: f64,
    #[serde(rename = "gbkqt3")]
    gbkqt3: f64,
    #[serde(rename = "gbegl19")]
    gbegl19: f64,
    #[serde(rename = "gbunc2")]
    gbunc2: f64,
    #[serde(rename = "gbcca1")]
    gbcca1: f64,
    #[serde(rename = "gbslo1_egl19")]
    gbslo1_egl19: f64,
    #[serde(rename = "gbslo1_unc2")]
    gbslo1_unc2: f64,
    #[serde(rename = "gbslo2_egl19")]
    gbslo2_egl19: f64,
    #[serde(rename = "gbslo2_unc2")]
    gbslo2_unc2: f64,
    #[serde(rename = "gbkcnl")]
    gbkcnl: f64,
    #[serde(rename = "gbnca")]
    gbnca: f64,
    #[serde(rename = "gbirk")]
    gbirk: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct NeuriteParams {
    #[serde(rename = "Ra", default)]
    ra: f64,
    #[serde(rename = "cm", default)]
    cm: f64,
    #[serde(rename = "gpas", default)]
    gpas: f64,
    #[serde(rename = "epas", default)]
    epas: f64,
}

/// Top-level config.json structure (partial — only what we need).
#[derive(Clone, Debug, Deserialize)]
struct ConfigJson {
    cell_info: CellInfo,
    dir_info: DirInfo,
    #[serde(default)]
    cnt_info: CntInfo,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct CntInfo {
    #[serde(default)]
    weight_range: WeightRange,
    #[serde(default)]
    inh_prob: f64,
    /// Synapse count → weight: weight = syn_a * count^(-syn_b), clipped to range.
    #[serde(default = "default_syn_a")]
    syn_a: f64,
    #[serde(default = "default_syn_b")]
    syn_b: f64,
    /// Gap junction count → weight: weight = gj_a * count^(-gj_b), clipped to range.
    #[serde(default = "default_gj_a")]
    gj_a: f64,
    #[serde(default = "default_gj_b")]
    gj_b: f64,
    /// Synaptic weight distribution parameters (log-normal shape).
    #[serde(default = "default_syn_mu")]
    syn_mu: f64,
    #[serde(default = "default_syn_scale")]
    syn_scale: f64,
    #[serde(default = "default_gj_mu")]
    gj_mu: f64,
    #[serde(default = "default_gj_scale")]
    gj_scale: f64,
    /// Scale factor applied to synapse count before weight computation.
    #[serde(default = "default_syn_cnt_scale")]
    syn_cnt_scale: f64,
    #[serde(default = "default_gj_cnt_scale")]
    gj_cnt_scale: f64,
}

fn default_syn_a() -> f64 { 23.91 }
fn default_syn_b() -> f64 { 0.02285 }
fn default_gj_a() -> f64 { 20.49 }
fn default_gj_b() -> f64 { 0.02184 }
fn default_syn_mu() -> f64 { 0.44 }
fn default_syn_scale() -> f64 { 0.63 }
fn default_gj_mu() -> f64 { 0.7 }
fn default_gj_scale() -> f64 { 0.4 }
fn default_syn_cnt_scale() -> f64 { 0.5 }
fn default_gj_cnt_scale() -> f64 { 0.7 }

#[derive(Clone, Debug, Default, Deserialize)]
struct WeightRange {
    #[serde(default = "default_syn_range")]
    syn: (f64, f64),
    #[serde(default = "default_gj_range")]
    gj: (f64, f64),
}

fn default_syn_range() -> (f64, f64) { (0.1, 1.0) }
fn default_gj_range() -> (f64, f64) { (1e-5, 1e-4) }

#[derive(Clone, Debug, Deserialize)]
struct CellInfo {
    #[serde(rename = "cells_name_dic")]
    cells_name_dic: HashMap<String, String>,
}

#[derive(Clone, Debug, Deserialize)]
struct DirInfo {
    #[serde(rename = "cell_param_dir")]
    cell_param_dir: String,
    #[serde(rename = "adj_matrix_dir")]
    adj_matrix_dir: String,
    #[serde(rename = "synapse_sheet")]
    synapse_sheet: String,
    #[serde(rename = "gap_junction_sheet")]
    gap_junction_sheet: String,
}

// ── BAAIWormCellParam — in-memory representation ────────────────────────

/// Parsed per-neuron biophysical parameters.
#[derive(Clone, Debug)]
pub struct BAAIWormCellParam {
    pub name: String,
    pub id: u64,
    pub soma: SomaParamValues,
    pub neurite: NeuriteParamValues,
}

#[derive(Clone, Debug)]
pub struct SomaParamValues {
    pub ra: f32,
    pub cm: f32,
    pub gpas: f32,
    pub epas: f32,
    // 16 channel conductances (nS/μm²); cainternm is handled as cai update
    pub gbshl1: f32,
    pub gbshk1: f32,
    pub gbkvs1: f32,
    pub gbegl2: f32,
    pub gbegl36: f32,
    pub gbkqt3: f32,
    pub gbegl19: f32,
    pub gbunc2: f32,
    pub gbcca1: f32,
    pub gbslo1_egl19: f32,
    pub gbslo1_unc2: f32,
    pub gbslo2_egl19: f32,
    pub gbslo2_unc2: f32,
    pub gbkcnl: f32,
    pub gbnca: f32,
    pub gbirk: f32,
}

#[derive(Clone, Debug)]
pub struct NeuriteParamValues {
    pub ra: f32,
    pub cm: f32,
    pub gpas: f32,
    pub epas: f32,
}

impl From<&SomaParams> for SomaParamValues {
    fn from(s: &SomaParams) -> Self {
        Self {
            ra: s.ra as f32,
            cm: s.cm as f32,
            gpas: s.gpas as f32,
            epas: s.epas as f32,
            gbshl1: s.gbshl1 as f32,
            gbshk1: s.gbshk1 as f32,
            gbkvs1: s.gbkvs1 as f32,
            gbegl2: s.gbegl2 as f32,
            gbegl36: s.gbegl36 as f32,
            gbkqt3: s.gbkqt3 as f32,
            gbegl19: s.gbegl19 as f32,
            gbunc2: s.gbunc2 as f32,
            gbcca1: s.gbcca1 as f32,
            gbslo1_egl19: s.gbslo1_egl19 as f32,
            gbslo1_unc2: s.gbslo1_unc2 as f32,
            gbslo2_egl19: s.gbslo2_egl19 as f32,
            gbslo2_unc2: s.gbslo2_unc2 as f32,
            gbkcnl: s.gbkcnl as f32,
            gbnca: s.gbnca as f32,
            gbirk: s.gbirk as f32,
        }
    }
}

impl From<&NeuriteParams> for NeuriteParamValues {
    fn from(n: &NeuriteParams) -> Self {
        Self {
            ra: n.ra as f32,
            cm: n.cm as f32,
            gpas: n.gpas as f32,
            epas: n.epas as f32,
        }
    }
}

// ── BAAIWormLoader ──────────────────────────────────────────────────────

/// Loader for the BAAIWorm *C. elegans* connectome.
///
/// Reads the BAAIWorm project directory and converts it into a [`BrainDB`].
pub struct BAAIWormLoader {
    /// Per-neuron biophysical parameters, keyed by cell name.
    cell_params: HashMap<String, BAAIWormCellParam>,
    /// Chemical synapse count matrix: `syn_matrix[pre][post]`.
    syn_matrix: Vec<Vec<u32>>,
    /// Gap junction count matrix: `gap_matrix[pre][post]`.
    gap_matrix: Vec<Vec<u32>>,
    /// Cell name → numeric ID (0..301).
    name_to_id: HashMap<String, u64>,
    /// Numeric ID → cell name.
    id_to_name: HashMap<u64, String>,
    /// Number of neurons (typically 302).
    n_neurons: usize,
    /// Weight ranges from config (syn_min, syn_max, gj_min, gj_max).
    syn_weight_range: (f32, f32),
    gj_weight_range: (f32, f32),
    /// Probability that a non-GABAergic synapse is inhibitory.
    inh_prob: f32,
    /// Synapse weight params: w = syn_a * (count * syn_cnt_scale)^(-syn_b) * syn_scale, clipped.
    syn_a: f32,
    syn_b: f32,
    syn_cnt_scale: f32,
    #[allow(dead_code)] // reserved for future log-normal weight sampling
    syn_mu: f32,
    syn_scale: f32,
    /// Gap junction weight params: w = gj_a * (count * gj_cnt_scale)^(-gj_b) * gj_scale, clipped.
    gj_a: f32,
    gj_b: f32,
    gj_cnt_scale: f32,
    #[allow(dead_code)] // reserved for future log-normal weight sampling
    gj_mu: f32,
    gj_scale: f32,
}

impl BAAIWormLoader {
    /// Load all BAAIWorm data from the project directory.
    ///
    /// Expected directory layout (relative to `dir`):
    /// ```text
    /// eworm/
    ///   network/config.json
    ///   components/param/cell/<NAME>.json
    ///   components/param/connection/SI5-302.xlsx
    /// ```
    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        // 1. Read config.json for name↔ID mapping and directory paths.
        let config_path = dir.join("network").join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: ConfigJson = serde_json::from_str(&config_str)?;

        // Build name↔ID mapping (only neurons with index < 302).
        let mut name_to_id = HashMap::new();
        let mut id_to_name = HashMap::new();
        for (id_str, name) in &config.cell_info.cells_name_dic {
            let id: u64 = id_str.parse().unwrap_or(u64::MAX);
            if id < 302 {
                name_to_id.insert(name.clone(), id);
                id_to_name.insert(id, name.clone());
            }
        }
        let n_neurons = name_to_id.len().max(302);

        // 2. Read per-neuron JSON files.
        let cell_param_dir = dir.join(&config.dir_info.cell_param_dir);
        let mut cell_params = HashMap::new();
        for (name, &id) in &name_to_id {
            let json_path = cell_param_dir.join(format!("{}.json", name));
            if !json_path.exists() {
                // Skip cells without parameter files (e.g. muscle cells).
                continue;
            }
            let json_str = std::fs::read_to_string(&json_path)?;
            let cp: CellParamJson = serde_json::from_str(&json_str)?;
            cell_params.insert(name.clone(), BAAIWormCellParam {
                name: name.clone(),
                id,
                soma: SomaParamValues::from(&cp.soma),
                neurite: NeuriteParamValues::from(&cp.neurite),
            });
        }

        // 3. Read Excel adjacency matrices.
        let xlsx_path = dir.join(&config.dir_info.adj_matrix_dir);
        let syn_sheet = &config.dir_info.synapse_sheet;
        let gap_sheet = &config.dir_info.gap_junction_sheet;

        let syn_matrix = read_adjacency_xlsx(&xlsx_path, syn_sheet, n_neurons)?;
        let gap_matrix = read_adjacency_xlsx(&xlsx_path, gap_sheet, n_neurons)?;

        let syn_weight_range = (
            config.cnt_info.weight_range.syn.0 as f32,
            config.cnt_info.weight_range.syn.1 as f32,
        );
        let gj_weight_range = (
            config.cnt_info.weight_range.gj.0 as f32,
            config.cnt_info.weight_range.gj.1 as f32,
        );

        Ok(Self {
            cell_params,
            syn_matrix,
            gap_matrix,
            name_to_id,
            id_to_name,
            n_neurons,
            syn_weight_range,
            gj_weight_range,
            inh_prob: config.cnt_info.inh_prob as f32,
            syn_a: config.cnt_info.syn_a as f32,
            syn_b: config.cnt_info.syn_b as f32,
            syn_cnt_scale: config.cnt_info.syn_cnt_scale as f32,
            syn_mu: config.cnt_info.syn_mu as f32,
            syn_scale: config.cnt_info.syn_scale as f32,
            gj_a: config.cnt_info.gj_a as f32,
            gj_b: config.cnt_info.gj_b as f32,
            gj_cnt_scale: config.cnt_info.gj_cnt_scale as f32,
            gj_mu: config.cnt_info.gj_mu as f32,
            gj_scale: config.cnt_info.gj_scale as f32,
        })
    }

    /// Convert loaded data into a [`BrainDB`] file at `output_path`.
    ///
    /// This creates a `BrainDBBuilder`, registers all entities, and calls
    /// `builder.build()`.
    pub fn into_braindb(self, output_path: &Path) -> Result<BrainDB> {
        let mut builder = BrainDBBuilder::new();

        // ── Register receptor types (continuous Sigmoid mode) ──────────
        let receptor_exc = builder.add_receptor(ReceptorParams {
            // Continuous Sigmoid parameters for excitatory synapses
            v_threshold: -20.0,
            v_slope: 5.0,
            k_rate: 0.1,
            // Standard AMPA-like kinetics (unused in continuous mode but
            // filled for completeness).
            ..ReceptorParams::ampa()
        });
        let receptor_inh = builder.add_receptor(ReceptorParams {
            v_threshold: -20.0,
            v_slope: 5.0,
            k_rate: 0.1,
            ..ReceptorParams::gaba_a()
        });

        // ── Register worm-specific ion channels (§15.4) ───────────────
        let ch_shl1 = register_channel(&mut builder, "shl1", IonType::Sodium, 50.0);
        let ch_egl19 = register_channel(&mut builder, "egl19", IonType::Calcium, 120.0);
        let ch_unc2 = register_channel(&mut builder, "unc2", IonType::Calcium, 120.0);
        let ch_cca1 = register_channel(&mut builder, "cca1", IonType::Calcium, 120.0);
        let ch_shk1 = register_channel(&mut builder, "shk1", IonType::Potassium, -77.0);
        let ch_kvs1 = register_channel(&mut builder, "kvs1", IonType::Potassium, -77.0);
        let ch_kqt3 = register_channel(&mut builder, "kqt3", IonType::Potassium, -77.0);
        let ch_kcnl = register_channel(&mut builder, "kcnl", IonType::Potassium, -77.0);
        let ch_irk = register_channel(&mut builder, "irk", IonType::Potassium, -77.0);
        let ch_egl2 = register_channel(&mut builder, "egl2", IonType::Potassium, -77.0);
        let ch_egl36 = register_channel(&mut builder, "egl36", IonType::NonSpecific, 0.0);
        let ch_nca = register_channel(&mut builder, "nca", IonType::NonSpecific, 0.0);
        // Ca-dependent K channels (slo1/slo2) — ca_source_channel set below.
        let ch_slo1_egl19 = builder.add_ion_channel(IonChannelDef {
            name: "slo1_egl19".into(),
            ion: IonType::Potassium,
            e_rev: -77.0,
            gate_vars: vec![],
            ca_dependent: true,
            ca_source_channel: Some(ch_egl19),
        });
        let ch_slo1_unc2 = builder.add_ion_channel(IonChannelDef {
            name: "slo1_unc2".into(),
            ion: IonType::Potassium,
            e_rev: -77.0,
            gate_vars: vec![],
            ca_dependent: true,
            ca_source_channel: Some(ch_unc2),
        });
        let ch_slo2_egl19 = builder.add_ion_channel(IonChannelDef {
            name: "slo2_egl19".into(),
            ion: IonType::Potassium,
            e_rev: -77.0,
            gate_vars: vec![],
            ca_dependent: true,
            ca_source_channel: Some(ch_egl19),
        });
        let ch_slo2_unc2 = builder.add_ion_channel(IonChannelDef {
            name: "slo2_unc2".into(),
            ion: IonType::Potassium,
            e_rev: -77.0,
            gate_vars: vec![],
            ca_dependent: true,
            ca_source_channel: Some(ch_unc2),
        });

        // cainternm is handled as CompartmentState.cai update, not a channel.

        // ── Create neurons (each with unique type + 2 compartments) ───
        // Sort by ID for contiguous neuron_attr layout.
        let mut sorted_names: Vec<(u64, &str)> = self.name_to_id.iter()
            .map(|(n, &id)| (id, n.as_str()))
            .collect();
        sorted_names.sort_by_key(|(id, _)| *id);

        // ── Register brain regions (must be done before neurons) ────────
        let n_regions = REGION_RULES.len();
        // Count neurons per region.
        let mut region_counts = vec![0u32; n_regions];
        for (_neuron_id, name) in &sorted_names {
            let rid = infer_region(name) as usize;
            if rid < n_regions {
                region_counts[rid] += 1;
            }
        }
        // Recompute first_neuron by scanning sorted order.
        let mut rid_offset = vec![0u64; n_regions];
        let mut counter = 0u64;
        let mut first_seen = vec![true; n_regions];
        for (_, name) in &sorted_names {
            let rid = infer_region(name) as usize;
            if rid < n_regions && first_seen[rid] {
                rid_offset[rid] = counter;
                first_seen[rid] = false;
            }
            counter += 1;
        }
        for rid in 0..n_regions {
            if region_counts[rid] > 0 {
                builder.add_region(
                    crate::core::region::BrainRegion {
                        id: rid as u32,
                        _pad_id: 0,
                        name_hash: 0,
                        first_neuron: rid_offset[rid],
                        neuron_count: region_counts[rid],
                        _pad0: 0,
                        cx: 0.0, cy: 0.0, cz: 0.0,
                        _pad_xyz: 0,
                        modulation: Default::default(),
                        glia: Default::default(),
                        sensory_start: 0,
                        sensory_end: 0,
                        motor_start: 0,
                        motor_end: 0,
                        _pad_tail: [0; 2],
                    },
                    REGION_RULES[rid].0,
                );
            }
        }

        let mut region_counts_so_far = vec![0u32; n_regions];

        for (neuron_id, name) in &sorted_names {
            let neuron_id = *neuron_id;
            let param = match self.cell_params.get(*name) {
                Some(p) => p,
                None => continue, // skip cells without JSON params
            };

            // Determine model: most worm neurons are graded.
            let model = if is_graded_neuron(name) {
                NeuronModel::Graded
            } else {
                NeuronModel::HodgkinHuxley
            };

            // Build soma channel set from non-zero conductances.
            let mut soma_channels = Vec::new();
            // 16 active channel conductances (cainternm is handled as cai update, not a channel)
            let conductances: [(f32, u32); 16] = [
                (param.soma.gbshl1, ch_shl1),
                (param.soma.gbshk1, ch_shk1),
                (param.soma.gbkvs1, ch_kvs1),
                (param.soma.gbegl2, ch_egl2),
                (param.soma.gbegl36, ch_egl36),
                (param.soma.gbkqt3, ch_kqt3),
                (param.soma.gbegl19, ch_egl19),
                (param.soma.gbunc2, ch_unc2),
                (param.soma.gbcca1, ch_cca1),
                (param.soma.gbslo1_egl19, ch_slo1_egl19),
                (param.soma.gbslo1_unc2, ch_slo1_unc2),
                (param.soma.gbslo2_egl19, ch_slo2_egl19),
                (param.soma.gbslo2_unc2, ch_slo2_unc2),
                (param.soma.gbkcnl, ch_kcnl),
                (param.soma.gbnca, ch_nca),
                (param.soma.gbirk, ch_irk),
            ];
            for &(g_max, ch_id) in &conductances {
                if g_max > 0.0 {
                    soma_channels.push(ChannelConductance {
                        channel_id: ch_id,
                        g_max,
                        e_rev_override: None,
                    });
                }
            }

            let soma_ics_id = builder.add_ion_channel_set(IonChannelSet {
                name: format!("{}_soma", name),
                channels: soma_channels,
            });

            // Neurite: passive only (no active channels in BAAIWorm).
            let neurite_ics_id = builder.add_ion_channel_set(IonChannelSet {
                name: format!("{}_neurite", name),
                channels: vec![],
            });

            // Per-neuron type (§15.3: each neuron gets its own type).
            let type_id = builder.add_neuron_type(NeuronTypeParams {
                type_name: name.to_string(),
                model,
                ..Default::default()
            });

            // Region inference from neuron name prefix.
            let region_id = infer_region(name);

            // Initial position: seed by region along body axis (force-directed will refine).
            const REGION_X: [f32; 6] = [
                0.0,    // Ventral Cord — midbody
                -400.0, // Anterior Sensory — head tip
                -250.0, // Anterior Interneuron — head
                150.0,  // Motor Neurons — body wall
                -100.0, // Lateral/Head Motor — anterior midbody
                300.0,  // Other/Tail — tail
            ];
            let rid = region_id as usize;
            let base_x = if rid < REGION_X.len() { REGION_X[rid] } else { 0.0 };
            let _idx_in_region = region_counts_so_far[rid];
            region_counts_so_far[rid] += 1;
            // Small deterministic jitter so initial positions aren't degenerate.
            let jitter = ((neuron_id as f32 * 1.618).fract() - 0.5) * 40.0;
            let x = base_x + jitter;
            let y = ((neuron_id as f32 * 2.345).fract() - 0.5) * 40.0;
            let z = ((neuron_id as f32 * 3.789).fract() - 0.5) * 40.0;

            // Neuron attribute.
            builder.add_neuron(NeuronAttr {
                id: neuron_id,
                first_comp_id: neuron_id * 2, // soma = id*2, neurite = id*2+1
                neuron_type: type_id,
                region_id,
                n_compartment: 2,
                cm: param.soma.cm,
                g_leak: param.soma.gpas,
                e_leak: param.soma.epas,
                x, y, z,
                flags: NEURON_ALIVE,
                _pad: [0; 3],
                _reserved: 0,
            });

            // Soma compartment.
            builder.add_compartment(CompartmentAttr {
                id: neuron_id * 2,
                neuron_id: neuron_id,
                parent_comp_id: u64::MAX, // soma = root
                comp_type: CompType::Soma as u8,
                _pad0: [0; 3],
                ion_channel_set: soma_ics_id,
                length: 10.0,
                diameter: 5.0, // C. elegans soma ~5 μm
                cm: param.soma.cm,
                r_axial: param.soma.ra,
                x: 0.0, y: 0.0, z: 0.0,
                g_leak: param.soma.gpas,
                e_leak: param.soma.epas,
                _pad1: 0,
                _reserved: [0u64; 7],
            });

            // Neurite compartment (parent = soma).
            builder.add_compartment(CompartmentAttr {
                id: neuron_id * 2 + 1,
                neuron_id: neuron_id,
                parent_comp_id: neuron_id * 2, // parent is soma
                comp_type: CompType::Axon as u8,
                _pad0: [0; 3],
                ion_channel_set: neurite_ics_id,
                length: 100.0,
                diameter: 0.5, // C. elegans neurite ~0.5 μm
                cm: param.neurite.cm,
                r_axial: param.neurite.ra,
                x: 0.0, y: 0.0, z: 0.0,
                g_leak: param.neurite.gpas,
                e_leak: param.neurite.epas,
                _pad1: 0,
                _reserved: [0u64; 7],
            });
        }

        // ── Create chemical synapses (all ContinuousConductance) ──────
        let n = self.n_neurons.min(self.syn_matrix.len());
        // Deterministic RNG for reproducible weight sampling.
        let mut rng = rand::rngs::StdRng::from_seed([42u8; 32]);
        for pre_id in 0..n {
            let row_len = self.syn_matrix[pre_id].len().min(n);
            for post_id in 0..row_len {
                let n_syn = self.syn_matrix[pre_id][post_id];
                if n_syn == 0 {
                    continue;
                }
                // Determine excitatory/inhibitory:
                // GABAergic neurons are always inhibitory;
                // non-GABAergic neurons have inh_prob chance of being inhibitory.
                let is_gabaergic = !is_excitatory_neuron(
                    self.id_to_name.get(&(pre_id as u64)).map(|s| s.as_str()).unwrap_or("")
                );
                let is_inhibitory = is_gabaergic || rand::Rng::gen_ratio(&mut rng, (self.inh_prob * 1000.0) as u32, 1000);
                let (syn_type, receptor_type) = if is_inhibitory {
                    (SYN_INHIBITORY, receptor_inh)
                } else {
                    (SYN_EXCITATORY, receptor_exc)
                };
                // Weight from count-based distribution:
                // w = syn_a * (count * syn_cnt_scale)^(-syn_b) * syn_scale, clipped to range.
                let scaled_count = (n_syn as f32) * self.syn_cnt_scale;
                let w = if scaled_count > 0.0 {
                    self.syn_a * scaled_count.powf(-self.syn_b) * self.syn_scale
                } else {
                    self.syn_weight_range.0
                };
                let w_clipped = w.clamp(self.syn_weight_range.0, self.syn_weight_range.1);
                let base_weight = if syn_type == SYN_INHIBITORY { -w_clipped } else { w_clipped };
                for _ in 0..n_syn {
                    builder.add_synapse(pre_id as u32, SynapseAttr {
                        post_neuron: post_id as u32,
                        post_comp: 0,  // target soma
                        pre_comp: 1,   // source neurite
                        base_weight,
                        delay_ticks: 3,
                        syn_type,
                        syn_mode: SYN_MODE_CONTINUOUS,
                        receptor_type,
                        _pad0: [0; 3],
                        u_se: 0.5,
                        u_fac: 0.0,
                        tau_rec: 100.0,
                    });
                }
            }
        }

        // ── Create gap junctions ──────────────────────────────────────
        let n = self.n_neurons.min(self.gap_matrix.len());
        for pre_id in 0..n {
            let row_len = self.gap_matrix[pre_id].len().min(n);
            for post_id in 0..row_len {
                let n_gap = self.gap_matrix[pre_id][post_id];
                if n_gap == 0 {
                    continue;
                }
                // Weight from count-based distribution:
                // w = gj_a * (count * gj_cnt_scale)^(-gj_b) * gj_scale, clipped to range.
                let scaled_count = (n_gap as f32) * self.gj_cnt_scale;
                let w = if scaled_count > 0.0 {
                    self.gj_a * scaled_count.powf(-self.gj_b) * self.gj_scale
                } else {
                    self.gj_weight_range.0
                };
                let w_clipped = w.clamp(self.gj_weight_range.0, self.gj_weight_range.1);
                for _ in 0..n_gap {
                    builder.add_gap_junction(GapJunction {
                        pre_neuron: pre_id as u32,
                        post_neuron: post_id as u32,
                        pre_comp: 0,
                        post_comp: 0,
                        weight: w_clipped,
                        _pad: 0,
                        _reserved: 0,
                    });
                }
            }
        }

        // ── Force-directed layout: refine positions using connectivity ─────
        // Synapses & gap junctions act as springs pulling connected neurons closer.
        // All-pairs repulsion prevents overlap. Region anchors preserve body-axis structure.
        {
            let n = self.n_neurons;
            let mut pos: Vec<[f32; 3]> = (0..n).map(|i| {
                let attr = builder.neuron_attr(i as u64);
                [attr.x, attr.y, attr.z]
            }).collect();

            // Collect edges with strength (connection count as spring constant).
            let mut edges: Vec<(usize, usize, f32)> = Vec::new();
            let sn = n.min(self.syn_matrix.len());
            for pre in 0..sn {
                let rl = self.syn_matrix[pre].len().min(n);
                for post in 0..rl {
                    let cnt = self.syn_matrix[pre][post];
                    if cnt > 0 { edges.push((pre, post, cnt as f32 * 0.5)); }
                }
            }
            let gn = n.min(self.gap_matrix.len());
            for pre in 0..gn {
                let rl = self.gap_matrix[pre].len().min(n);
                for post in 0..rl {
                    let cnt = self.gap_matrix[pre][post];
                    if cnt > 0 { edges.push((pre, post, cnt as f32 * 1.0)); } // gap junctions stronger pull
                }
            }

            // Force-directed iteration.
            let repulsion: f32 = 500.0;    // Coulomb-like repulsion strength
            let ideal_len: f32 = 80.0;    // spring rest length
            let region_anchor: f32 = 0.02; // gentle pull back to region seed position
            let damping: f32 = 0.6;       // strong damping to prevent explosion
            let iterations = 200;
            let max_vel: f32 = 5.0;       // velocity clamp per axis

            // Store region seed positions for anchoring.
            let seed_pos: Vec<[f32; 3]> = pos.clone();

            let mut vel: Vec<[f32; 3]> = vec![[0.0; 3]; n];

            for _iter in 0..iterations {
                let alpha = 1.0 - (_iter as f32) / (iterations as f32); // cooling
                // Repulsion (all pairs, O(n²) is fine for 302 neurons).
                for i in 0..n {
                    for j in (i+1)..n {
                        let dx = pos[i][0] - pos[j][0];
                        let dy = pos[i][1] - pos[j][1];
                        let dz = pos[i][2] - pos[j][2];
                        let dist_sq = (dx*dx + dy*dy + dz*dz).max(1.0); // clamp to avoid div-by-zero
                        let dist = dist_sq.sqrt();
                        let force = repulsion / dist_sq * alpha;
                        let fx = dx / dist * force;
                        let fy = dy / dist * force;
                        let fz = dz / dist * force;
                        vel[i][0] += fx; vel[i][1] += fy; vel[i][2] += fz;
                        vel[j][0] -= fx; vel[j][1] -= fy; vel[j][2] -= fz;
                    }
                }
                // Attraction (springs along edges).
                for &(pre, post, strength) in &edges {
                    let dx = pos[post][0] - pos[pre][0];
                    let dy = pos[post][1] - pos[pre][1];
                    let dz = pos[post][2] - pos[pre][2];
                    let dist = (dx*dx + dy*dy + dz*dz).sqrt().max(1.0);
                    let displacement = dist - ideal_len;
                    let force = displacement * strength * 0.01 * alpha;
                    let fx = dx / dist * force;
                    let fy = dy / dist * force;
                    let fz = dz / dist * force;
                    vel[pre][0] += fx; vel[pre][1] += fy; vel[pre][2] += fz;
                    vel[post][0] -= fx; vel[post][1] -= fy; vel[post][2] -= fz;
                }
                // Region anchor: gently pull back toward seed position.
                for i in 0..n {
                    for ax in 0..3 {
                        vel[i][ax] += (seed_pos[i][ax] - pos[i][ax]) * region_anchor * alpha;
                    }
                }
                // Integrate with damping and velocity clamping.
                for i in 0..n {
                    for ax in 0..3 {
                        vel[i][ax] = (vel[i][ax] * damping).clamp(-max_vel, max_vel);
                        pos[i][ax] += vel[i][ax];
                        // Guard against NaN / infinity from numerical instability.
                        if !pos[i][ax].is_finite() {
                            pos[i][ax] = seed_pos[i][ax];
                            vel[i][ax] = 0.0;
                        }
                    }
                }
            }

            // Write final positions back to builder.
            for i in 0..n {
                builder.set_neuron_pos(i as u64, pos[i][0], pos[i][1], pos[i][2]);
                // Set compartment positions: soma at neuron position, neurite extending outward.
                let soma_comp_id = i as u64 * 2;
                let neurite_comp_id = i as u64 * 2 + 1;
                builder.set_comp_pos(soma_comp_id, pos[i][0], pos[i][1], pos[i][2]);
                // Neurite extends ~100 μm in a direction seeded by neuron index.
                let angle_y = (i as f32 * 2.39996) % (2.0 * std::f32::consts::PI); // golden angle
                let angle_z = ((i as f32 * 1.61803) % 1.0 - 0.5) * std::f32::consts::PI * 0.6;
                let nx = pos[i][0] + 100.0 * angle_z.cos() * angle_y.cos();
                let ny = pos[i][1] + 100.0 * angle_z.cos() * angle_y.sin();
                let nz = pos[i][2] + 100.0 * angle_z.sin();
                builder.set_comp_pos(neurite_comp_id, nx, ny, nz);
            }
        }

        // Write neuron names into builder metadata.
        let mut names = vec![String::new(); self.n_neurons];
        for (id, name) in &self.id_to_name {
            let i = *id as usize;
            if i < self.n_neurons {
                names[i] = name.clone();
            }
        }
        builder.neuron_names = names;

        builder.build(output_path)
    }

    /// Lower-level API: load data into an existing builder without calling
    /// `build()`. Useful for composing multiple loaders.
    pub fn load_into_builder(dir: &Path, _builder: &mut BrainDBBuilder) -> Result<Self> {
        let loader = Self::load_from_dir(dir)?;
        // We return the loader; the caller must call into_braindb() or
        // manually populate the builder. For now this is a convenience
        // that just does the parsing step.
        Ok(loader)
    }
}

// ── Helper functions ─────────────────────────────────────────────────────

/// Register a simple ion channel (no gate vars — kinetics handled by
/// GATE_FN_REGISTRY at runtime). Returns the channel ID.
fn register_channel(builder: &mut BrainDBBuilder, name: &str, ion: IonType, e_rev: f32) -> u32 {
    builder.add_ion_channel(IonChannelDef {
        name: name.to_string(),
        ion,
        e_rev,
        gate_vars: vec![], // placeholder; real kinetics registered at runtime
        ca_dependent: false,
        ca_source_channel: None,
    })
}

/// Determine if a neuron is graded (non-spiking).
///
/// Most *C. elegans* neurons are graded. Only a few motor neurons and
/// interneurons produce classic Na⁺ spikes (AVB, RID, etc.).
fn is_graded_neuron(name: &str) -> bool {
    // Known spiking neurons in C. elegans (producing classical action potentials).
    const SPIKING_NEURONS: &[&str] = &[
        "AVBL", "AVBR", // command interneurons (spike-like)
        "AVDL", "AVDR",
        "AVEL", "AVER",
        "RID",
    ];
    !SPIKING_NEURONS.contains(&name)
}

/// Known inhibitory (GABAergic) neurons in C. elegans.
fn is_excitatory_neuron(name: &str) -> bool {
    // GABAergic motor neurons: VD, DD, RME, AVL, DVB
    const INHIBITORY_PREFIXES: &[&str] = &["VD", "DD", "RME"];
    const INHIBITORY_EXACT: &[&str] = &["AVL", "DVB"];

    for prefix in INHIBITORY_PREFIXES {
        if name.starts_with(prefix) {
            return false;
        }
    }
    if INHIBITORY_EXACT.contains(&name) {
        return false;
    }
    true
}

/// Region definition: (region_name, list_of_name_prefixes).
///
/// This is a data-driven approach — for a different organism (e.g. fruit fly),
/// just replace this table. The last entry acts as the catch-all "Other" bucket.
const REGION_RULES: &[(&str, &[&str])] = &[
    ("Ventral Cord",          &["AV", "PV"]),
    ("Anterior Sensory",      &["AD", "ASE", "ASG", "ASH", "ASI", "ASJ", "ASK", "AW", "AFD", "ADF", "ADL"]),
    ("Anterior Interneuron",  &["RI", "AI", "AU", "AIA", "AIB", "AIZ", "AIM", "AIN", "AIY"]),
    ("Motor Neurons",         &["DA", "DB", "VA", "VB", "VD", "DD", "AS", "VC"]),
    ("Lateral/Head Motor",    &["RM", "RIV", "RIM", "SA", "SAB", "SIA", "SIB", "SMB", "SMD"]),
    ("Other/Tail",           &[]), // catch-all
];

/// Map neuron name to brain region ID using REGION_RULES.
///
/// Returns the index into REGION_RULES. The last rule (empty prefix list)
/// matches everything — it acts as the default.
fn infer_region(name: &str) -> u32 {
    for (rid, (_, prefixes)) in REGION_RULES.iter().enumerate() {
        if prefixes.is_empty() {
            return rid as u32; // catch-all
        }
        if prefixes.iter().any(|p| name.starts_with(p)) {
            return rid as u32;
        }
    }
    // Fallback (should never reach if last rule is catch-all)
    (REGION_RULES.len() - 1) as u32
}

// ── Excel adjacency matrix reader ───────────────────────────────────────

/// Read an adjacency matrix from an Excel sheet.
///
/// The BAAIWorm SI5-302.xlsx has:
/// - Row 0: header row with neuron names
/// - Col 0: neuron names
/// - Cells: synapse/gap count (integer or empty)
///
/// Returns a `Vec<Vec<u32>>` of size `n × n`.
fn read_adjacency_xlsx(
    path: &Path,
    sheet_name: &str,
    n_neurons: usize,
) -> Result<Vec<Vec<u32>>> {
    use calamine::{open_workbook, Reader, Xlsx};

    let mut workbook: Xlsx<_> = open_workbook(path)
        .map_err(|e: calamine::XlsxError| BrainDBError::Spreadsheet(e.to_string()))?;
    let range = workbook.worksheet_range(sheet_name)
        .ok_or_else(|| BrainDBError::Spreadsheet(format!("sheet '{}' not found", sheet_name)))?
        .map_err(|e| BrainDBError::Spreadsheet(e.to_string()))?;

    let mut matrix = vec![vec![0u32; n_neurons]; n_neurons];

    // BAAIWorm's adjacency matrix is ordered by cell ID.
    // Row 0 and column 0 are headers (neuron names).
    // Data cells start at (1,1) and map to neuron IDs 0..n-1.
    for (row_idx, row) in range.rows().enumerate() {
        if row_idx == 0 {
            continue; // skip header row
        }
        let pre_id = row_idx - 1;
        if pre_id >= n_neurons {
            break;
        }
        for (col_idx, cell) in row.iter().enumerate() {
            if col_idx == 0 {
                continue; // skip header column
            }
            let post_id = col_idx - 1;
            if post_id >= n_neurons {
                break;
            }
            let count = cell_data_to_u32(cell);
            matrix[pre_id][post_id] = count;
        }
    }

    Ok(matrix)
}

/// Convert a calamine cell value to u32 (synapse/gap count).
fn cell_data_to_u32(cell: &CalData) -> u32 {
    match cell {
        CalData::Int(i) => *i as u32,
        CalData::Float(f) => *f as u32,
        CalData::String(s) => s.parse::<u32>().unwrap_or(0),
        _ => 0,
    }
}
