//! Generic connectome loader (CSV/JSON).
//!
//! Expected CSV format (header row required):
//! ```csv
//! pre_id,post_id,weight,delay_ms,syn_type,receptor_type
//! 0,1,1.0,1.5,1,0
//! ```
//!
//! JSON format: an array of objects with the same fields.

use std::path::Path;

use crate::core::synapse::{
    SynapseAttr, RECEPTOR_AMPA, SYN_EXCITATORY,
    SYN_MODE_EVENT_DRIVEN,
};
use crate::error::{BrainDBError, Result};
use crate::storage::builder::BrainDBBuilder;

/// Connectome loader — reads synapse connectivity from CSV or JSON.
pub struct ConnectomeLoader;

/// A parsed synapse row from CSV/JSON.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SynapseRow {
    pre_id: u32,
    post_id: u32,
    weight: f32,
    delay_ms: f32,
    syn_type: u8,       // SYN_EXCITATORY or SYN_INHIBITORY
    receptor_type: u8,   // RECEPTOR_AMPA, etc.
}

impl ConnectomeLoader {
    /// Load synapses from a CSV file into the builder.
    ///
    /// CSV must have a header row. Recognized columns:
    /// `pre_id`, `post_id`, `weight`, `delay_ms`, `syn_type`, `receptor_type`.
    /// Missing columns get defaults: weight=1.0, delay=1.5ms, excitatory/AMPA.
    pub fn load_csv(path: &Path, builder: &mut BrainDBBuilder) -> Result<()> {
        let data = std::fs::read_to_string(path)
            .map_err(BrainDBError::Io)?;
        let rows = Self::parse_csv(&data)?;
        for row in &rows {
            let delay_ticks = (row.delay_ms / builder.get_dt()).max(1.0) as u16;
            builder.add_synapse(row.pre_id, SynapseAttr {
                post_neuron: row.post_id,
                post_comp: 0,
                pre_comp: 0,
                base_weight: row.weight,
                delay_ticks,
                syn_type: row.syn_type,
                syn_mode: SYN_MODE_EVENT_DRIVEN,
                receptor_type: row.receptor_type,
                _pad0: [0; 3],
                u_se: 0.5,
                u_fac: 0.0,
                tau_rec: 100.0,
            });
        }
        Ok(())
    }

    /// Load synapses from a JSON file into the builder.
    ///
    /// JSON must be an array of objects with the same fields as CSV columns.
    pub fn load_json(path: &Path, builder: &mut BrainDBBuilder) -> Result<()> {
        let data = std::fs::read_to_string(path)
            .map_err(BrainDBError::Io)?;
        let rows: Vec<SynapseRow> = serde_json::from_str(&data)
            .map_err(BrainDBError::Json)?;
        for row in &rows {
            let delay_ticks = (row.delay_ms / builder.get_dt()).max(1.0) as u16;
            builder.add_synapse(row.pre_id, SynapseAttr {
                post_neuron: row.post_id,
                post_comp: 0,
                pre_comp: 0,
                base_weight: row.weight,
                delay_ticks,
                syn_type: row.syn_type,
                syn_mode: SYN_MODE_EVENT_DRIVEN,
                receptor_type: row.receptor_type,
                _pad0: [0; 3],
                u_se: 0.5,
                u_fac: 0.0,
                tau_rec: 100.0,
            });
        }
        Ok(())
    }

    /// Parse CSV text into synapse rows.
    fn parse_csv(data: &str) -> Result<Vec<SynapseRow>> {
        let mut rows = Vec::new();
        let mut lines = data.lines();
        // Parse header to get column indices.
        let header = lines.next().ok_or_else(|| {
            BrainDBError::Validation("CSV: empty file".into())
        })?;
        let cols: Vec<&str> = header.split(',').map(|s| s.trim()).collect();
        let col_pre = cols.iter().position(|c| *c == "pre_id").ok_or_else(|| {
            BrainDBError::Validation("CSV: missing 'pre_id' column".into())
        })?;
        let col_post = cols.iter().position(|c| *c == "post_id").ok_or_else(|| {
            BrainDBError::Validation("CSV: missing 'post_id' column".into())
        })?;
        let col_weight = cols.iter().position(|c| *c == "weight");
        let col_delay = cols.iter().position(|c| *c == "delay_ms");
        let col_type = cols.iter().position(|c| *c == "syn_type");
        let col_receptor = cols.iter().position(|c| *c == "receptor_type");

        for line in lines {
            let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if fields.len() <= col_pre.max(col_post) {
                continue; // skip malformed
            }
            let pre_id: u32 = fields[col_pre].parse().unwrap_or(0);
            let post_id: u32 = fields[col_post].parse().unwrap_or(0);
            let weight = col_weight
                .and_then(|i| fields.get(i))
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0);
            let delay_ms = col_delay
                .and_then(|i| fields.get(i))
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.5);
            let syn_type = col_type
                .and_then(|i| fields.get(i))
                .and_then(|s| s.parse().ok())
                .unwrap_or(SYN_EXCITATORY);
            let receptor_type = col_receptor
                .and_then(|i| fields.get(i))
                .and_then(|s| s.parse().ok())
                .unwrap_or(RECEPTOR_AMPA);
            rows.push(SynapseRow { pre_id, post_id, weight, delay_ms, syn_type, receptor_type });
        }
        Ok(rows)
    }
}

