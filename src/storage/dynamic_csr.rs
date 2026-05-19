//! Dynamic CSR — runtime synapse insertion/deletion for structural plasticity.
//!
//! Design (from `braindb-design.md` §16.3):
//! - Standard CSR (mmap read-only) + delta area (mutable)
//! - Insertions: `Vec<(u32 pre_id, SynapseAttr)>`
//! - Deletions: `HashSet<u64>` of global synapse indices
//! - Rebuild CSR when `ops_since_rebuild >= threshold`
//! - Low-frequency operation (every 10000 tick = 1 s)

use std::collections::HashSet;

use crate::core::synapse::{SynapseAttr, SynapseState};

/// Dynamic CSR — supports runtime synapse insertion/deletion on top of an
/// immutable CSR base loaded from mmap.
pub struct DynamicCSR {
    // ── Immutable CSR base (copied from mmap at load time) ────────────
    pub row_ptr: Vec<u64>,
    pub col_idx: Vec<u64>,
    pub syn_attrs: Vec<SynapseAttr>,
    pub n_neurons: u32,

    // ── Delta area (mutable) ──────────────────────────────────────────
    /// New synapses awaiting merge: `(pre_neuron_id, attr)`.
    pub insertions: Vec<(u32, SynapseAttr)>,
    /// Running states for newly inserted synapses.
    pub delta_syn_states: Vec<SynapseState>,
    /// Global synapse indices marked for deletion.
    pub deletions: HashSet<u64>,

    // ── Rebuild counter ───────────────────────────────────────────────
    pub ops_since_rebuild: u32,
    pub rebuild_threshold: u32,
}

impl DynamicCSR {
    /// Construct from an existing CSR (row_ptr, col_idx, syn_attrs).
    pub fn new(
        row_ptr: Vec<u64>,
        col_idx: Vec<u64>,
        syn_attrs: Vec<SynapseAttr>,
        n_neurons: u32,
    ) -> Self {
        Self {
            row_ptr,
            col_idx,
            syn_attrs,
            n_neurons,
            insertions: Vec::new(),
            delta_syn_states: Vec::new(),
            deletions: HashSet::new(),
            ops_since_rebuild: 0,
            rebuild_threshold: 1000,
        }
    }

    /// Total synapse count (base + insertions - deletions).
    pub fn total_syn_count(&self) -> usize {
        let base = self.syn_attrs.len();
        let deleted = self.deletions.iter().filter(|&&i| (i as usize) < base).count();
        base - deleted + self.insertions.len()
    }

    /// Insert a new synapse (sprouting). Returns a temporary global ID.
    pub fn insert_synapse(&mut self, pre_id: u32, attr: SynapseAttr) -> u64 {
        let delta_idx = self.insertions.len() as u64;
        self.insertions.push((pre_id, attr));
        self.delta_syn_states.push(SynapseState::default());
        self.ops_since_rebuild += 1;
        // Temporary ID = base_len + delta_idx
        (self.syn_attrs.len() as u64) + delta_idx
    }

    /// Mark a synapse for deletion (pruning). Not immediately removed.
    pub fn remove_synapse(&mut self, global_idx: u64) {
        self.deletions.insert(global_idx);
        self.ops_since_rebuild += 1;
    }

    /// Check if a global synapse index is marked as deleted.
    pub fn is_deleted(&self, global_idx: u64) -> bool {
        self.deletions.contains(&global_idx)
    }

    /// Get the mutable state for a synapse by global index.
    /// Returns `None` if deleted or out of range.
    pub fn syn_state_mut(&mut self, global_idx: u64) -> Option<&mut SynapseState> {
        let base = self.syn_attrs.len() as u64;
        if global_idx < base {
            if self.deletions.contains(&global_idx) {
                None
            } else {
                // Base synapse states are managed externally (in BrainDB.syn_states).
                // This method only handles delta-area states.
                None
            }
        } else {
            let delta_idx = (global_idx - base) as usize;
            self.delta_syn_states.get_mut(delta_idx)
        }
    }

    /// Get the delta-area synapse state by delta index.
    pub fn delta_state(&self, delta_idx: usize) -> Option<&SynapseState> {
        self.delta_syn_states.get(delta_idx)
    }

    /// Get the delta-area synapse state mutably by delta index.
    pub fn delta_state_mut(&mut self, delta_idx: usize) -> Option<&mut SynapseState> {
        self.delta_syn_states.get_mut(delta_idx)
    }

    /// Iterate over outgoing synapses of `pre_id` in the base CSR.
    pub fn csr_out_range(&self, pre_id: u32) -> std::ops::Range<usize> {
        if pre_id as usize >= self.n_neurons as usize {
            return 0..0;
        }
        let s = self.row_ptr[pre_id as usize] as usize;
        let e = self.row_ptr[pre_id as usize + 1] as usize;
        s..e
    }

    /// Iterate over delta-area insertions for `pre_id`.
    pub fn delta_out_synapses(&self, pre_id: u32) -> impl Iterator<Item = (usize, &SynapseAttr)> {
        self.insertions
            .iter()
            .enumerate()
            .filter(move |&(_, (pid, _))| *pid == pre_id)
            .map(|(i, (_, attr))| (i, attr))
    }

    /// Rebuild the CSR — merge base (minus deletions) + insertions.
    /// Call when `ops_since_rebuild >= rebuild_threshold`.
    pub fn rebuild(&mut self) {
        // 1. Collect surviving base synapses.
        let mut all_synapses: Vec<(u32, SynapseAttr)> = Vec::with_capacity(self.total_syn_count());
        for i in 0..self.syn_attrs.len() {
            if !self.deletions.contains(&(i as u64)) {
                let pre_id = self.find_pre_id(i);
                all_synapses.push((pre_id, self.syn_attrs[i]));
            }
        }
        // 2. Add insertions.
        all_synapses.extend(self.insertions.drain(..));

        // 3. Sort by pre_id → rebuild CSR.
        all_synapses.sort_by_key(|(pre, _)| *pre);
        let n = self.n_neurons as usize;
        let mut new_row_ptr = vec![0u64; n + 1];
        for (pre_id, _) in &all_synapses {
            if *pre_id as usize + 1 < new_row_ptr.len() {
                new_row_ptr[*pre_id as usize + 1] += 1;
            }
        }
        for i in 1..=n {
            new_row_ptr[i] += new_row_ptr[i - 1];
        }
        let new_col_idx: Vec<u64> = all_synapses.iter().map(|(_, a)| a.post_neuron as u64).collect();
        let new_syn_attrs: Vec<SynapseAttr> = all_synapses.iter().map(|(_, a)| *a).collect();

        // 4. Replace.
        self.row_ptr = new_row_ptr;
        self.col_idx = new_col_idx;
        self.syn_attrs = new_syn_attrs;
        self.deletions.clear();
        self.delta_syn_states.clear();
        self.ops_since_rebuild = 0;
    }

    /// Reverse-lookup: find the pre_neuron id for a given base CSR index.
    fn find_pre_id(&self, syn_idx: usize) -> u32 {
        // Binary search in row_ptr for the row containing syn_idx.
        let target = syn_idx as u64;
        for i in 0..self.n_neurons as usize {
            if i + 1 < self.row_ptr.len()
                && self.row_ptr[i] <= target
                && target < self.row_ptr[i + 1]
            {
                return i as u32;
            }
        }
        0 // fallback
    }

    /// Should rebuild be triggered?
    pub fn should_rebuild(&self) -> bool {
        self.ops_since_rebuild >= self.rebuild_threshold
    }
}
