//! NeuroML / SWC morphology loader.
//!
//! SWC format: one line per sample point:
//! `id type x y z radius parent_id`
//!
//! Each SWC point becomes a compartment. The tree topology is derived
//! from `parent_id` (-1 = root/soma). Compartment length is the
//! Euclidean distance to the parent; diameter = 2 × radius.

use std::path::Path;

use crate::core::compartment::{CompType, CompartmentAttr};
use crate::error::{BrainDBError, Result};
use crate::storage::builder::BrainDBBuilder;

/// NeuroML / SWC morphology loader.
pub struct NeuroMLLoader;

/// A parsed SWC point.
#[derive(Clone, Debug)]
struct SwcPoint {
    id: u32,
    comp_type: u8,   // 1=soma, 2=axon, 3=dendrite, 4=apical_dend
    x: f32,
    y: f32,
    z: f32,
    radius: f32,
    parent_id: i64,  // -1 = root
}

impl NeuroMLLoader {
    /// Load an SWC morphology file and add compartments to the builder.
    ///
    /// The caller must have already added a neuron to the builder.
    /// Returns the first compartment ID.
    pub fn load_swc(
        path: &Path,
        builder: &mut BrainDBBuilder,
        neuron_id: u32,
        ra_ohm_cm: f32,  // axial resistivity (Ohm·cm), typical 150
        cm_u_f_cm2: f32,  // specific membrane capacitance (μF/cm²), typical 1.0
    ) -> Result<u32> {
        let data = std::fs::read_to_string(path)
            .map_err(BrainDBError::Io)?;
        let points = Self::parse_swc(&data)?;

        if points.is_empty() {
            return Err(BrainDBError::Validation("SWC: no points".into()));
        }

        // Build a lookup: swc_id → index in points vec.
        let mut id_to_idx = std::collections::HashMap::new();
        for (idx, p) in points.iter().enumerate() {
            id_to_idx.insert(p.id, idx);
        }

        // Add compartments.
        let first_comp_id = builder.compartment_count() as u32;
        for (i, pt) in points.iter().enumerate() {
            let comp_type = match pt.comp_type {
                1 => CompType::Soma as u8,
                2 => CompType::Axon as u8,
                3 => CompType::BasalDend as u8,
                4 => CompType::ApicalDend as u8,
                _ => CompType::Soma as u8,
            };

            // Compute length and r_axial.
            let (length, r_axial) = if pt.parent_id < 0 {
                (0.0, 1e10) // root: no axial resistance
            } else {
                let parent_idx = id_to_idx.get(&(pt.parent_id as u32));
                match parent_idx {
                    Some(&pidx) => {
                        let pp = &points[pidx];
                        let dx = pt.x - pp.x;
                        let dy = pt.y - pp.y;
                        let dz = pt.z - pp.z;
                        let len = (dx*dx + dy*dy + dz*dz).sqrt();
                        // r_axial = Ra * L / (π * (d/2)²) in Ohm·cm * μm / μm²
                        // = Ra * L / (π * r²)  [Ohm·cm * μm / μm² = Ohm·cm / μm]
                        // Convert to nS: g_int_nS = π * d² * 1e5 / (4 * Ra * L)
                        // We store r_axial = Ra (Ohm·cm) and let the cable solver
                        // compute g_int using d and L.
                        (len.max(0.01), ra_ohm_cm)
                    }
                    None => (1.0, ra_ohm_cm),
                }
            };

            let diameter = 2.0 * pt.radius;

            // Membrane capacitance: C_m = cm_u_f_cm2 * area
            // area = π * d * L (μm²) → cm² = area * 1e-8
            // C_m (pF) = cm_u_f_cm2 * area * 1e-8 * 1e6 = cm_u_f_cm2 * area * 1e-2
            let area_um2 = std::f32::consts::PI * diameter * length;
            let cm_pf = cm_u_f_cm2 * area_um2 * 1e-2;

            // Parent compartment ID.
            let parent_comp_id = if pt.parent_id < 0 {
                u64::MAX
            } else {
                // Map SWC parent to our compartment index.
                let parent_idx = id_to_idx.get(&(pt.parent_id as u32));
                match parent_idx {
                    Some(&pidx) => first_comp_id as u64 + pidx as u64,
                    None => u64::MAX,
                }
            };

            let attr = CompartmentAttr {
                id: first_comp_id as u64 + i as u64,
                neuron_id: neuron_id as u64,
                parent_comp_id,
                comp_type,
                _pad0: [0; 3],
                ion_channel_set: 0, // default, overridden later
                length,
                diameter,
                cm: cm_pf.max(1e-6),
                r_axial,
                x: pt.x,
                y: pt.y,
                z: pt.z,
                g_leak: 0.3,  // mS/cm² → simplified
                e_leak: -65.0,
                _pad1: 0,
                _reserved: [0; 7],
            };
            builder.add_compartment(attr);
        }

        Ok(first_comp_id)
    }

    /// Parse SWC text into points.
    fn parse_swc(data: &str) -> Result<Vec<SwcPoint>> {
        let mut points = Vec::new();
        for line in data.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 7 {
                continue;
            }
            let id: u32 = fields[0].parse().unwrap_or(0);
            let comp_type: u8 = fields[1].parse().unwrap_or(1);
            let x: f32 = fields[2].parse().unwrap_or(0.0);
            let y: f32 = fields[3].parse().unwrap_or(0.0);
            let z: f32 = fields[4].parse().unwrap_or(0.0);
            let radius: f32 = fields[5].parse().unwrap_or(0.5);
            let parent_id: i64 = fields[6].parse().unwrap_or(-1);
            points.push(SwcPoint { id, comp_type, x, y, z, radius, parent_id });
        }
        // Sort by ID for deterministic ordering.
        points.sort_by_key(|p| p.id);
        Ok(points)
    }
}

