//! `BrainDB` — zero-copy mmap loader + dynamic-state snapshots.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use bytemuck::bytes_of;
use memmap2::Mmap;

use crate::core::compartment::{CompartmentAttr, CompartmentState};
use crate::core::gap_junction::GapJunction;
use crate::core::neuron::{NeuronAttr, NeuronState};
use crate::core::receptor::ReceptorParams;
use crate::core::region::{BrainRegion, LongRangePathway};
use crate::core::synapse::{SynapseAttr, SynapseState};
use crate::ensure;
use crate::error::{BrainDBError, Result};
use crate::storage::format::{
    off, Header, MetaSection, SnapshotHeader, FILE_MAGIC, FILE_VERSION, HEADER_SIZE,
    SNAPSHOT_MAGIC, SNAPSHOT_VERSION,
};

/// A loaded BrainDB. Static segments are mapped read-only; dynamic state lives
/// in regular heap-allocated `Vec`s and is persisted via snapshot files.
pub struct BrainDB {
    // Static (mmap'd; the file is kept open via `_file`).
    pub(crate) mmap: Mmap,
    pub(crate) _file: File,
    pub header: Header,
    pub meta: MetaSection,

    // Dynamic state — owned, regular memory.
    pub neuron_states: Vec<NeuronState>,
    pub syn_states: Vec<SynapseState>,
    pub comp_states: Vec<CompartmentState>,

    pub current_tick: u64,
}

impl BrainDB {
    /// Open an existing `.braindb` v2 file (mmap-read static segments,
    /// initialise dynamic state to defaults).
    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new().read(true).open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        ensure!(
            mmap.len() >= HEADER_SIZE,
            "file too small: {} < HEADER_SIZE {HEADER_SIZE}",
            mmap.len()
        );

        let header: Header = *bytemuck::from_bytes::<Header>(&mmap[..HEADER_SIZE]);
        if header.magic != FILE_MAGIC {
            return Err(BrainDBError::InvalidMagic {
                expected: FILE_MAGIC,
                actual: header.magic,
            });
        }
        if header.version != FILE_VERSION {
            return Err(BrainDBError::UnsupportedVersion(header.version));
        }
        ensure!(
            (header.file_size as usize) <= mmap.len(),
            "header file_size {} exceeds mmap length {}",
            header.file_size,
            mmap.len()
        );

        // Decode metadata.
        let meta_off = header.offsets[off::META] as usize;
        let meta_end = meta_off + header.meta_len as usize;
        ensure!(meta_end <= mmap.len(), "metadata section overruns file");
        let meta: MetaSection = postcard::from_bytes(&mmap[meta_off..meta_end])?;

        // Initialise dynamic state to defaults.
        let neuron_states = vec![NeuronState::default(); header.n_neurons as usize];
        let syn_states = vec![SynapseState::default(); header.n_synapses as usize];
        let comp_states = vec![CompartmentState::default(); header.n_compartments as usize];

        let mut db = Self {
            mmap,
            _file: file,
            header,
            meta,
            neuron_states,
            syn_states,
            comp_states,
            current_tick: 0,
        };
        db.initialise_dynamic_defaults();
        Ok(db)
    }

    /// Apply per-type default initial values to the dynamic-state arrays
    /// (e.g. Izhikevich `v_mem = c`, base `weight = base_weight`).
    fn initialise_dynamic_defaults(&mut self) {
        // Neuron states: pull from NeuronTypeParams.default_v_init() when the
        // type is registered. Falls back to the NeuronAttr.e_leak.
        // Copy v_init values first to avoid borrowing self twice.
        let v_inits: Vec<f32> = self.neuron_attrs().iter().map(|attr| {
            self.meta
                .neuron_types
                .get(attr.neuron_type as usize)
                .map(|t| t.default_v_init())
                .unwrap_or(attr.e_leak)
        }).collect();
        for (i, st) in self.neuron_states.iter_mut().enumerate() {
            st.v_mem = v_inits[i];
            st.v_mem_soma = v_inits[i];
        }

        // Synapse states: weight ← base_weight (STDP-writable copy).
        let base_weights: Vec<f32> = self.syn_attrs().iter().map(|a| a.base_weight).collect();
        for (i, st) in self.syn_states.iter_mut().enumerate() {
            st.weight = base_weights[i];
        }

        // Compartment states: v_mem ← e_leak.
        let e_leaks: Vec<f32> = self.compartment_attrs().iter().map(|a| a.e_leak).collect();
        for (i, st) in self.comp_states.iter_mut().enumerate() {
            st.v_mem = e_leaks[i];
        }
    }

    // ── Zero-copy slice accessors into the mmap ──────────────────────────

    fn slice_at<T: bytemuck::Pod>(&self, slot: usize, n: usize) -> &[T] {
        let off = self.header.offsets[slot] as usize;
        let len = n * std::mem::size_of::<T>();
        let end = off + len;
        debug_assert!(end <= self.mmap.len(),
            "slot {slot} out of bounds: {off}..{end} > {}", self.mmap.len());
        bytemuck::cast_slice(&self.mmap[off..end])
    }

    pub fn neuron_attrs(&self) -> &[NeuronAttr] {
        self.slice_at(off::NEURON_ATTR, self.header.n_neurons as usize)
    }
    pub fn compartment_attrs(&self) -> &[CompartmentAttr] {
        self.slice_at(off::COMPARTMENT_ATTR, self.header.n_compartments as usize)
    }
    pub fn csr_row_ptr(&self) -> &[u64] {
        self.slice_at(off::CSR_ROW_PTR, self.header.n_neurons as usize + 1)
    }
    pub fn csr_col_idx(&self) -> &[u64] {
        self.slice_at(off::CSR_COL_IDX, self.header.n_synapses as usize)
    }
    pub fn syn_attrs(&self) -> &[SynapseAttr] {
        self.slice_at(off::SYNAPSE_ATTR, self.header.n_synapses as usize)
    }
    pub fn gap_junctions(&self) -> &[GapJunction] {
        self.slice_at(off::GAP, self.header.n_gap_junctions as usize)
    }
    pub fn regions(&self) -> &[BrainRegion] {
        self.slice_at(off::REGION, self.header.n_regions as usize)
    }
    pub fn pathways(&self) -> &[LongRangePathway] {
        self.slice_at(off::PATHWAY, self.header.n_pathways as usize)
    }
    pub fn receptors(&self) -> &[ReceptorParams] {
        self.slice_at(off::RECEPTOR, self.header.n_receptor_types as usize)
    }

    /// Out-synapse range `[start, end)` into `syn_attrs()` / `csr_col_idx()`
    /// for the given pre-synaptic neuron.
    pub fn out_range(&self, neuron_id: usize) -> std::ops::Range<usize> {
        let r = self.csr_row_ptr();
        r[neuron_id] as usize..r[neuron_id + 1] as usize
    }

    // ── Snapshot I/O ─────────────────────────────────────────────────────

    /// Write the dynamic state to a `.snapshot` file.
    pub fn save_snapshot(&self, path: &Path) -> Result<()> {
        let mut f = OpenOptions::new()
            .write(true).create(true).truncate(true).open(path)?;

        let hdr = SnapshotHeader {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
            _pad: 0,
            tick: self.current_tick,
            n_neurons: self.header.n_neurons,
            n_synapses: self.header.n_synapses,
            n_compartments: self.header.n_compartments,
        };
        f.write_all(bytes_of(&hdr))?;
        f.write_all(bytemuck::cast_slice(&self.neuron_states))?;
        f.write_all(bytemuck::cast_slice(&self.comp_states))?;
        f.write_all(bytemuck::cast_slice(&self.syn_states))?;
        f.sync_all()?;
        Ok(())
    }

    /// Restore dynamic state from a `.snapshot` file. The snapshot's neuron /
    /// synapse / compartment counts must match this database.
    pub fn load_snapshot(&mut self, path: &Path) -> Result<()> {
        let bytes = std::fs::read(path)?;
        let hdr_sz = std::mem::size_of::<SnapshotHeader>();
        ensure!(bytes.len() >= hdr_sz, "snapshot file too small");

        let hdr: SnapshotHeader = *bytemuck::from_bytes::<SnapshotHeader>(&bytes[..hdr_sz]);
        if hdr.magic != SNAPSHOT_MAGIC {
            return Err(BrainDBError::SnapshotMismatch(format!(
                "bad snapshot magic: {:?}", hdr.magic
            )));
        }
        if hdr.version != SNAPSHOT_VERSION {
            return Err(BrainDBError::SnapshotMismatch(format!(
                "unsupported snapshot version {}", hdr.version
            )));
        }
        if hdr.n_neurons != self.header.n_neurons
            || hdr.n_synapses != self.header.n_synapses
            || hdr.n_compartments != self.header.n_compartments
        {
            return Err(BrainDBError::SnapshotMismatch(format!(
                "shape mismatch: snapshot=({},{},{}) db=({},{},{})",
                hdr.n_neurons, hdr.n_synapses, hdr.n_compartments,
                self.header.n_neurons, self.header.n_synapses, self.header.n_compartments,
            )));
        }

        let n_neurons = hdr.n_neurons as usize;
        let n_comp = hdr.n_compartments as usize;
        let n_syn = hdr.n_synapses as usize;

        let n_bytes = n_neurons * std::mem::size_of::<NeuronState>();
        let c_bytes = n_comp * std::mem::size_of::<CompartmentState>();
        let s_bytes = n_syn * std::mem::size_of::<SynapseState>();
        let total = hdr_sz + n_bytes + c_bytes + s_bytes;
        ensure!(
            bytes.len() >= total,
            "snapshot truncated: have {}, need {total}",
            bytes.len()
        );

        let mut p = hdr_sz;
        load_unaligned(&bytes[p..p + n_bytes], &mut self.neuron_states);
        p += n_bytes;
        load_unaligned(&bytes[p..p + c_bytes], &mut self.comp_states);
        p += c_bytes;
        load_unaligned(&bytes[p..p + s_bytes], &mut self.syn_states);

        self.current_tick = hdr.tick;
        Ok(())
    }
}

/// Copy `bytes` into `out` using `pod_read_unaligned`, avoiding the
/// alignment requirements of `cast_slice`.  Snapshot files are read from
/// disk; the first byte after the 40-byte header is not guaranteed to be
/// aligned to the 64-byte requirement of `NeuronState`/`CompartmentState`.
fn load_unaligned<T: bytemuck::Pod>(bytes: &[u8], out: &mut [T]) {
    let sz = std::mem::size_of::<T>();
    assert_eq!(bytes.len(), out.len() * sz);
    for (i, chunk) in bytes.chunks_exact(sz).enumerate() {
        out[i] = bytemuck::pod_read_unaligned(chunk);
    }
}
