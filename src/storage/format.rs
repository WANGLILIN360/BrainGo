//! `.braindb` v2 binary file format.
//!
//! Layout (all little-endian, fixed-size segments are mmap-friendly):
//!
//! ```text
//! [0..512]                           Header   (exactly 512 B)
//! [hdr.off_neuron_attr ..]           NeuronAttr        × n_neurons        (64 B each)
//! [hdr.off_compartment_attr ..]      CompartmentAttr   × n_compartments   (128 B each)
//! [hdr.off_csr_row_ptr ..]           u64               × (n_neurons + 1)
//! [hdr.off_csr_col_idx ..]           u64               × n_synapses
//! [hdr.off_synapse_attr ..]          SynapseAttr       × n_synapses       (32 B each)
//! [hdr.off_gap ..]                   GapJunction       × n_gap_junctions  (24 B each)
//! [hdr.off_region ..]                BrainRegion       × n_regions        (Pod)
//! [hdr.off_pathway ..]               LongRangePathway  × n_pathways       (Pod)
//! [hdr.off_receptor ..]              ReceptorParams    × n_receptor_types (Pod)
//! [hdr.off_meta .. off_meta+meta_len] postcard-encoded MetaSection
//! ```
//!
//! Segments are 64-byte aligned. The metadata section holds the non-POD
//! descriptors (ion channels, neuron-type params, templates, naming, etc.).

use bytemuck;
use serde::{Deserialize, Serialize};
use static_assertions::const_assert_eq;

use crate::core::circuit_template::CircuitTemplate;
use crate::core::ion_channel::{IonChannelDef, IonChannelSet};
use crate::core::neuron_type::NeuronTypeParams;

pub const FILE_MAGIC: [u8; 4] = *b"BRDB";
pub const FILE_VERSION: u16 = 2;
pub const HEADER_SIZE: usize = 512;

/// Header offset slot indices (16 of 32 reserved slots are currently used).
pub mod off {
    pub const NEURON_ATTR: usize = 0;
    pub const COMPARTMENT_ATTR: usize = 1;
    pub const CSR_ROW_PTR: usize = 2;
    pub const CSR_COL_IDX: usize = 3;
    pub const SYNAPSE_ATTR: usize = 4;
    pub const GAP: usize = 5;
    pub const REGION: usize = 6;
    pub const PATHWAY: usize = 7;
    pub const RECEPTOR: usize = 8;
    pub const META: usize = 9;
}

/// 512-byte file header (POD).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub magic: [u8; 4],
    pub version: u16,
    pub flags: u16,

    pub n_neurons: u64,
    pub n_synapses: u64,
    pub n_gap_junctions: u64,
    pub n_compartments: u64,

    pub n_regions: u32,
    pub n_pathways: u32,
    pub n_ion_channels: u32,    // count inside the meta section
    pub n_neuron_types: u32,    // count inside the meta section
    pub n_templates: u32,       // count inside the meta section
    pub n_receptor_types: u32,

    pub dt: f32,                // simulation tick (ms)
    pub ring_size: u32,

    pub meta_len: u64,          // length of the postcard metadata section
    pub file_size: u64,         // total file size in bytes

    /// Byte offsets into the mmap'd file. Index using constants in [`off`].
    /// Unused slots are zero.
    pub offsets: [u64; 32],

    /// Padding to reach exactly 512 B (must remain zero on write).
    pub reserved: [u8; HEADER_SIZE
        - 4   // magic
        - 2 - 2
        - 8 * 4
        - 4 * 6
        - 4 - 4
        - 8 * 2
        - 8 * 32],
}

// Compile-time guarantee on header size.
const_assert_eq!(std::mem::size_of::<Header>(), HEADER_SIZE);

// Manual impls: bytemuck derive limits array lengths to ~36 elements;
// [u8; 168] exceeds that, but the type is still bit-wise valid.
unsafe impl bytemuck::Zeroable for Header {}
unsafe impl bytemuck::Pod for Header {}

impl Default for Header {
    fn default() -> Self {
        Self {
            magic: FILE_MAGIC,
            version: FILE_VERSION,
            flags: 0,
            n_neurons: 0,
            n_synapses: 0,
            n_gap_junctions: 0,
            n_compartments: 0,
            n_regions: 0,
            n_pathways: 0,
            n_ion_channels: 0,
            n_neuron_types: 0,
            n_templates: 0,
            n_receptor_types: 0,
            dt: 0.1,
            ring_size: 10_000,
            meta_len: 0,
            file_size: 0,
            offsets: [0; 32],
            reserved: [0; HEADER_SIZE
                - 4 - 2 - 2 - 8 * 4 - 4 * 6 - 4 - 4 - 8 * 2 - 8 * 32],
        }
    }
}

/// Postcard-serialized metadata section (non-POD descriptors).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MetaSection {
    pub neuron_types: Vec<NeuronTypeParams>,
    pub ion_channels: Vec<IonChannelDef>,
    pub ion_channel_sets: Vec<IonChannelSet>,
    pub templates: Vec<CircuitTemplate>,
    pub sensory_neuron_ids: Vec<u64>,
    pub motor_neuron_ids: Vec<u64>,
    /// Map of `BrainRegion.id` → human-readable name (for `name_hash` reverse
    /// lookup).
    pub region_names: Vec<(u32, String)>,
    /// Neuron names indexed by neuron ID (e.g. ["ADAL", "ADAR", ...]).
    pub neuron_names: Vec<String>,
}

// ── Snapshot format ───────────────────────────────────────────────────────

pub const SNAPSHOT_MAGIC: [u8; 4] = *b"BRSN";
pub const SNAPSHOT_VERSION: u16 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SnapshotHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub _pad: u16,
    pub tick: u64,
    pub n_neurons: u64,
    pub n_synapses: u64,
    pub n_compartments: u64,
}

const_assert_eq!(std::mem::size_of::<SnapshotHeader>(), 40);
unsafe impl bytemuck::Zeroable for SnapshotHeader {}
unsafe impl bytemuck::Pod for SnapshotHeader {}

/// 64-byte alignment helper for laying out the file. Returns the smallest
/// `n ≥ offset` such that `n % align == 0`.
#[inline]
pub fn align_up(offset: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (offset + align - 1) & !(align - 1)
}

pub const SEGMENT_ALIGN: u64 = 64;
