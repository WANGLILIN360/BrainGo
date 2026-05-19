//! Region-level pathway queries — inter-region connectivity analysis.
//!
//! Design doc §8.3: brain-region-level queries including pathway strength,
//! region adjacency, and inter-region synapse counts.

use crate::core::region::LongRangePathway;
use crate::storage::mmap_db::BrainDB;

/// Summary of connectivity between two brain regions.
#[derive(Clone, Debug)]
pub struct RegionPathwayInfo {
    pub source_region: u32,
    pub target_region: u32,
    /// Total number of synapses from source → target.
    pub synapse_count: usize,
    /// Total synaptic weight (sum of base_weight).
    pub total_weight: f32,
    /// Mean synaptic weight.
    pub mean_weight: f32,
    /// Number of unique pre-synaptic neurons in source that project to target.
    pub pre_neuron_count: usize,
    /// Number of unique post-synaptic neurons in target that receive from source.
    pub post_neuron_count: usize,
}

/// Compute inter-region pathway info by scanning the CSR.
///
/// For each synapse whose pre-neuron is in `source_region` and post-neuron
/// is in `target_region`, accumulate statistics.
pub fn region_pathway_info(db: &BrainDB, source_region: u32, target_region: u32) -> Option<RegionPathwayInfo> {
    let regions = db.regions();
    let src = regions.get(source_region as usize)?;
    let tgt = regions.get(target_region as usize)?;

    let src_first = src.first_neuron as usize;
    let src_end = src_first + src.neuron_count as usize;
    let tgt_first = tgt.first_neuron as usize;
    let tgt_end = tgt_first + tgt.neuron_count as usize;

    let row_ptr = db.csr_row_ptr();
    let col_idx = db.csr_col_idx();
    let syn_attrs = db.syn_attrs();

    let mut synapse_count: usize = 0;
    let mut total_weight: f32 = 0.0;
    let mut pre_neurons: Vec<u32> = Vec::new();
    let mut post_neurons: Vec<u32> = Vec::new();

    for pre in src_first..src_end {
        let s = row_ptr[pre] as usize;
        let e = row_ptr[pre + 1] as usize;
        for syn_idx in s..e {
            let post = col_idx[syn_idx] as usize;
            if post >= tgt_first && post < tgt_end {
                synapse_count += 1;
                total_weight += syn_attrs[syn_idx].base_weight;
                if !pre_neurons.contains(&(pre as u32)) {
                    pre_neurons.push(pre as u32);
                }
                if !post_neurons.contains(&(post as u32)) {
                    post_neurons.push(post as u32);
                }
            }
        }
    }

    let mean_weight = if synapse_count > 0 {
        total_weight / synapse_count as f32
    } else {
        0.0
    };

    Some(RegionPathwayInfo {
        source_region,
        target_region,
        synapse_count,
        total_weight,
        mean_weight,
        pre_neuron_count: pre_neurons.len(),
        post_neuron_count: post_neurons.len(),
    })
}

/// Return all pathways from `source_region` to any other region.
pub fn outgoing_pathways(db: &BrainDB, source_region: u32) -> Vec<RegionPathwayInfo> {
    let n_regions = db.regions().len();
    let mut result = Vec::new();
    for tgt in 0..n_regions {
        if tgt == source_region as usize { continue; }
        if let Some(info) = region_pathway_info(db, source_region, tgt as u32) {
            if info.synapse_count > 0 {
                result.push(info);
            }
        }
    }
    result
}

/// Return all pathways into `target_region` from any other region.
pub fn incoming_pathways(db: &BrainDB, target_region: u32) -> Vec<RegionPathwayInfo> {
    let n_regions = db.regions().len();
    let mut result = Vec::new();
    for src in 0..n_regions {
        if src == target_region as usize { continue; }
        if let Some(info) = region_pathway_info(db, src as u32, target_region) {
            if info.synapse_count > 0 {
                result.push(info);
            }
        }
    }
    result
}

/// Find all LongRangePathways that connect two regions (either direction).
pub fn pathways_between(db: &BrainDB, region_a: u32, region_b: u32) -> Vec<LongRangePathway> {
    db.pathways()
        .iter()
        .filter(|pw| {
            (pw.source_region == region_a && pw.target_region == region_b)
                || (pw.source_region == region_b && pw.target_region == region_a)
        })
        .copied()
        .collect()
}

/// Build an N×N inter-region connectivity matrix where entry (i,j) is the
/// total synaptic weight from region i → region j.
pub fn region_connectivity_matrix(db: &BrainDB) -> Vec<Vec<f32>> {
    let n = db.regions().len();
    let mut mat = vec![vec![0.0f32; n]; n];

    let row_ptr = db.csr_row_ptr();
    let col_idx = db.csr_col_idx();
    let syn_attrs = db.syn_attrs();
    let regions = db.regions();

    // Build neuron → region lookup.
    let neuron_region: Vec<u32> = {
        let mut nr = vec![u32::MAX; row_ptr.len() - 1];
        for r in regions {
            let first = r.first_neuron as usize;
            let end = first + r.neuron_count as usize;
            for i in first..end.min(nr.len()) {
                nr[i] = r.id;
            }
        }
        nr
    };

    for pre in 0..neuron_region.len() {
        let src_r = neuron_region[pre] as usize;
        if src_r >= n { continue; }
        let s = row_ptr[pre] as usize;
        let e = row_ptr[pre + 1] as usize;
        for syn_idx in s..e {
            let post = col_idx[syn_idx] as usize;
            if post < neuron_region.len() {
                let tgt_r = neuron_region[post] as usize;
                if tgt_r < n {
                    mat[src_r][tgt_r] += syn_attrs[syn_idx].base_weight;
                }
            }
        }
    }
    mat
}
