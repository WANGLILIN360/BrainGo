//! `BrainDBBuilder` — gather entities in arbitrary order, sort synapses by
//! `pre_neuron`, build CSR, write a `.braindb` v2 file, and return a loaded
//! [`BrainDB`].

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use bytemuck::bytes_of;

use crate::core::circuit_template::CircuitTemplate;
use crate::core::compartment::CompartmentAttr;
use crate::core::gap_junction::GapJunction;
use crate::core::ion_channel::{IonChannelDef, IonChannelSet};
use crate::core::neuron::NeuronAttr;
use crate::core::neuron_type::NeuronTypeParams;
use crate::core::receptor::ReceptorParams;
use crate::core::region::{BrainRegion, LongRangePathway};
use crate::core::synapse::SynapseAttr;
use crate::ensure;
use crate::error::{BrainDBError, Result};
use crate::storage::format::{
    align_up, off, Header, MetaSection, FILE_MAGIC, FILE_VERSION, HEADER_SIZE, SEGMENT_ALIGN,
};
use crate::storage::mmap_db::BrainDB;

/// Builder for a fresh `.braindb` v2 file.
///
/// Add entities in any order, then call [`Self::build`] to:
/// 1. validate ID continuity and synapse references;
/// 2. sort synapses by `pre_neuron` and emit a CSR;
/// 3. compute file offsets and write the `.braindb` v2 file;
/// 4. mmap-open the file and return a fully-initialised [`BrainDB`].
#[derive(Default)]
pub struct BrainDBBuilder {
    // Registries.
    pub(crate) neuron_types: Vec<NeuronTypeParams>,
    pub(crate) receptors: Vec<ReceptorParams>,
    pub(crate) ion_channels: Vec<IonChannelDef>,
    pub(crate) ion_channel_sets: Vec<IonChannelSet>,
    pub(crate) templates: Vec<CircuitTemplate>,

    // Entities (any order).
    pub(crate) neurons: Vec<NeuronAttr>,
    pub(crate) compartments: Vec<CompartmentAttr>,
    pub(crate) synapses: Vec<(u32, SynapseAttr)>,
    pub(crate) gap_junctions: Vec<GapJunction>,
    pub(crate) regions: Vec<BrainRegion>,
    pub(crate) pathways: Vec<LongRangePathway>,

    // Auxiliary metadata.
    pub(crate) sensory_neuron_ids: Vec<u64>,
    pub(crate) motor_neuron_ids: Vec<u64>,
    pub(crate) region_names: Vec<(u32, String)>,
    pub(crate) neuron_names: Vec<String>,

    pub(crate) dt: f32,
    pub(crate) ring_size: u32,
}

impl BrainDBBuilder {
    pub fn new() -> Self {
        Self {
            dt: 0.1,
            ring_size: 10_000,
            ..Default::default()
        }
    }

    pub fn dt(mut self, dt: f32) -> Self { self.dt = dt; self }
    pub fn ring_size(mut self, n: u32) -> Self { self.ring_size = n; self }

    /// Get the configured time step (ms).
    pub fn get_dt(&self) -> f32 { self.dt }

    /// Get the current compartment count (for assigning IDs).
    pub fn compartment_count(&self) -> usize { self.compartments.len() }

    // ── Registration ─────────────────────────────────────────────────────

    pub fn add_neuron_type(&mut self, mut p: NeuronTypeParams) -> u32 {
        let id = self.neuron_types.len() as u32;
        p.type_id = id;
        self.neuron_types.push(p);
        id
    }

    pub fn add_receptor(&mut self, p: ReceptorParams) -> u8 {
        let id = self.receptors.len() as u8;
        self.receptors.push(p);
        id
    }

    pub fn add_ion_channel(&mut self, c: IonChannelDef) -> u32 {
        let id = self.ion_channels.len() as u32;
        self.ion_channels.push(c);
        id
    }

    pub fn add_ion_channel_set(&mut self, s: IonChannelSet) -> u32 {
        let id = self.ion_channel_sets.len() as u32;
        self.ion_channel_sets.push(s);
        id
    }

    pub fn add_template(&mut self, t: CircuitTemplate) {
        self.templates.push(t);
    }

    // ── Entities ─────────────────────────────────────────────────────────

    pub fn add_neuron(&mut self, attr: NeuronAttr) -> u64 {
        let id = attr.id;
        self.neurons.push(attr);
        id
    }

    pub fn neuron_attr(&self, id: u64) -> &NeuronAttr {
        &self.neurons[id as usize]
    }

    pub fn set_neuron_pos(&mut self, id: u64, x: f32, y: f32, z: f32) {
        let n = &mut self.neurons[id as usize];
        n.x = x;
        n.y = y;
        n.z = z;
    }

    pub fn add_compartment(&mut self, attr: CompartmentAttr) {
        self.compartments.push(attr);
    }

    pub fn set_comp_pos(&mut self, id: u64, x: f32, y: f32, z: f32) {
        let c = &mut self.compartments[id as usize];
        c.x = x;
        c.y = y;
        c.z = z;
    }

    pub fn add_synapse(&mut self, pre_id: u32, attr: SynapseAttr) {
        self.synapses.push((pre_id, attr));
    }

    pub fn add_gap_junction(&mut self, gj: GapJunction) {
        self.gap_junctions.push(gj);
    }

    pub fn add_region(&mut self, region: BrainRegion, name: impl Into<String>) {
        self.region_names.push((region.id, name.into()));
        self.regions.push(region);
    }

    pub fn add_pathway(&mut self, p: LongRangePathway) {
        self.pathways.push(p);
    }

    pub fn add_sensory_neuron(&mut self, id: u64) { self.sensory_neuron_ids.push(id); }
    pub fn add_motor_neuron(&mut self, id: u64) { self.motor_neuron_ids.push(id); }

    // ── Build ────────────────────────────────────────────────────────────

    /// Validate, sort, lay out and write the file, then mmap it open.
    pub fn build(self, path: &Path) -> Result<BrainDB> {
        // 1. Validate.
        let n_neurons = self.neurons.len();
        let n_synapses = self.synapses.len();
        let n_compartments = self.compartments.len();
        let n_gap = self.gap_junctions.len();

        for (i, a) in self.neurons.iter().enumerate() {
            ensure!(
                a.id as usize == i,
                "neuron ID not contiguous: expected {i}, got {}",
                a.id
            );
        }
        for (pre_id, a) in &self.synapses {
            ensure!(
                (*pre_id as usize) < n_neurons,
                "invalid pre_neuron: {pre_id} (n_neurons={n_neurons})"
            );
            ensure!(
                (a.post_neuron as usize) < n_neurons,
                "invalid post_neuron: {} (n_neurons={n_neurons})",
                a.post_neuron
            );
        }
        for gj in &self.gap_junctions {
            ensure!(
                (gj.pre_neuron as usize) < n_neurons
                    && (gj.post_neuron as usize) < n_neurons,
                "invalid gap junction endpoints"
            );
        }

        // 2. CSR build.
        let (row_ptr, col_idx, syn_attrs) = build_csr(n_neurons, &self.synapses);

        // 3. Serialize metadata.
        let meta = MetaSection {
            neuron_types: self.neuron_types,
            ion_channels: self.ion_channels,
            ion_channel_sets: self.ion_channel_sets,
            templates: self.templates,
            sensory_neuron_ids: self.sensory_neuron_ids,
            motor_neuron_ids: self.motor_neuron_ids,
            region_names: self.region_names,
            neuron_names: self.neuron_names,
        };
        let meta_bytes = postcard::to_allocvec(&meta)?;

        // 4. Compute segment offsets.
        let mut header = Header {
            magic: FILE_MAGIC,
            version: FILE_VERSION,
            n_neurons: n_neurons as u64,
            n_synapses: n_synapses as u64,
            n_gap_junctions: n_gap as u64,
            n_compartments: n_compartments as u64,
            n_regions: self.regions.len() as u32,
            n_pathways: self.pathways.len() as u32,
            n_ion_channels: meta.ion_channels.len() as u32,
            n_neuron_types: meta.neuron_types.len() as u32,
            n_templates: meta.templates.len() as u32,
            n_receptor_types: self.receptors.len() as u32,
            dt: self.dt,
            ring_size: self.ring_size,
            meta_len: meta_bytes.len() as u64,
            ..Default::default()
        };

        let mut cur = HEADER_SIZE as u64;
        cur = align_up(cur, SEGMENT_ALIGN);
        header.offsets[off::NEURON_ATTR] = cur;
        cur += (n_neurons * std::mem::size_of::<NeuronAttr>()) as u64;

        cur = align_up(cur, SEGMENT_ALIGN);
        header.offsets[off::COMPARTMENT_ATTR] = cur;
        cur += (n_compartments * std::mem::size_of::<CompartmentAttr>()) as u64;

        cur = align_up(cur, SEGMENT_ALIGN);
        header.offsets[off::CSR_ROW_PTR] = cur;
        cur += ((n_neurons + 1) * std::mem::size_of::<u64>()) as u64;

        cur = align_up(cur, SEGMENT_ALIGN);
        header.offsets[off::CSR_COL_IDX] = cur;
        cur += (n_synapses * std::mem::size_of::<u64>()) as u64;

        cur = align_up(cur, SEGMENT_ALIGN);
        header.offsets[off::SYNAPSE_ATTR] = cur;
        cur += (n_synapses * std::mem::size_of::<SynapseAttr>()) as u64;

        cur = align_up(cur, SEGMENT_ALIGN);
        header.offsets[off::GAP] = cur;
        cur += (n_gap * std::mem::size_of::<GapJunction>()) as u64;

        cur = align_up(cur, SEGMENT_ALIGN);
        header.offsets[off::REGION] = cur;
        cur += (self.regions.len() * std::mem::size_of::<BrainRegion>()) as u64;

        cur = align_up(cur, SEGMENT_ALIGN);
        header.offsets[off::PATHWAY] = cur;
        cur += (self.pathways.len() * std::mem::size_of::<LongRangePathway>()) as u64;

        cur = align_up(cur, SEGMENT_ALIGN);
        header.offsets[off::RECEPTOR] = cur;
        cur += (self.receptors.len() * std::mem::size_of::<ReceptorParams>()) as u64;

        cur = align_up(cur, SEGMENT_ALIGN);
        header.offsets[off::META] = cur;
        cur += meta_bytes.len() as u64;

        header.file_size = cur;

        // 5. Write the file.
        let mut file = OpenOptions::new()
            .read(true).write(true).create(true).truncate(true)
            .open(path)?;

        // Header at offset 0.
        file.write_all(bytes_of(&header))?;

        write_segment(&mut file, header.offsets[off::NEURON_ATTR],
                      bytemuck::cast_slice(&self.neurons))?;
        write_segment(&mut file, header.offsets[off::COMPARTMENT_ATTR],
                      bytemuck::cast_slice(&self.compartments))?;
        write_segment(&mut file, header.offsets[off::CSR_ROW_PTR],
                      bytemuck::cast_slice(&row_ptr))?;
        write_segment(&mut file, header.offsets[off::CSR_COL_IDX],
                      bytemuck::cast_slice(&col_idx))?;
        write_segment(&mut file, header.offsets[off::SYNAPSE_ATTR],
                      bytemuck::cast_slice(&syn_attrs))?;
        write_segment(&mut file, header.offsets[off::GAP],
                      bytemuck::cast_slice(&self.gap_junctions))?;
        write_segment(&mut file, header.offsets[off::REGION],
                      bytemuck::cast_slice(&self.regions))?;
        write_segment(&mut file, header.offsets[off::PATHWAY],
                      bytemuck::cast_slice(&self.pathways))?;
        write_segment(&mut file, header.offsets[off::RECEPTOR],
                      bytemuck::cast_slice(&self.receptors))?;
        write_segment(&mut file, header.offsets[off::META], &meta_bytes)?;

        // Pad file to declared size.
        file.set_len(header.file_size)?;
        file.sync_all()?;
        drop(file);

        // 6. Re-open via mmap.
        BrainDB::open(path)
    }
}

/// Build CSR from `(pre_id, attr)` pairs.
fn build_csr(
    n_neurons: usize,
    synapses: &[(u32, SynapseAttr)],
) -> (Vec<u64>, Vec<u64>, Vec<SynapseAttr>) {
    let mut sorted: Vec<(u32, SynapseAttr)> = synapses.to_vec();
    sorted.sort_by_key(|(pre, _)| *pre);

    let mut row_ptr = vec![0u64; n_neurons + 1];
    for (pre_id, _) in &sorted {
        row_ptr[*pre_id as usize + 1] += 1;
    }
    for i in 1..=n_neurons {
        row_ptr[i] += row_ptr[i - 1];
    }

    let col_idx: Vec<u64> = sorted.iter().map(|(_, a)| a.post_neuron as u64).collect();
    let syn_attrs: Vec<SynapseAttr> = sorted.iter().map(|(_, a)| *a).collect();
    (row_ptr, col_idx, syn_attrs)
}

/// Seek to `offset` (padding with zeros if needed) and write `bytes`.
fn write_segment(file: &mut std::fs::File, offset: u64, bytes: &[u8]) -> Result<()> {
    use std::io::Seek;
    let cur = file.stream_position()?;
    if offset < cur {
        return Err(BrainDBError::InvalidFile(format!(
            "segment offset {offset} < current position {cur}"
        )));
    }
    if offset > cur {
        let pad = (offset - cur) as usize;
        file.write_all(&vec![0u8; pad])?;
    }
    file.write_all(bytes)?;
    Ok(())
}
