//! Graph queries over the static connectome.
//!
//! All routines work on the read-only forward CSR exposed by [`BrainDB`]; for
//! upstream queries we build a transient reverse adjacency list. Edge weight
//! used for path queries is `SynapseAttr.base_weight` (the static, file-resident
//! base weight — *not* the simulation's mutable `SynapseState.weight`).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use crate::storage::mmap_db::BrainDB;

/// `(neuron_id, hop_count)` pairs returned by BFS routines.
pub type Hit = (u64, u32);

/// All neurons reachable from `source` within `max_hops` forward edges,
/// excluding the source itself.
///
/// Returned in BFS order (so the first occurrences have the smallest hop
/// count).
pub fn bfs_downstream(db: &BrainDB, source: u64, max_hops: u32) -> Vec<Hit> {
    let n = db.header.n_neurons as usize;
    if (source as usize) >= n || max_hops == 0 {
        return Vec::new();
    }
    let row = db.csr_row_ptr();
    let col = db.csr_col_idx();

    let mut visited = vec![false; n];
    let mut out: Vec<Hit> = Vec::new();
    let mut q: VecDeque<(u64, u32)> = VecDeque::new();
    visited[source as usize] = true;
    q.push_back((source, 0));

    while let Some((nid, depth)) = q.pop_front() {
        if depth >= max_hops {
            continue;
        }
        let s = row[nid as usize] as usize;
        let e = row[nid as usize + 1] as usize;
        for &target in &col[s..e] {
            if !visited[target as usize] {
                visited[target as usize] = true;
                out.push((target, depth + 1));
                q.push_back((target, depth + 1));
            }
        }
    }
    out
}

/// All neurons that can reach `target` within `max_hops` forward edges.
/// Builds a transient reverse CSR; for repeated queries, prefer caching it
/// (e.g. via [`crate::sim::Simulation::rev_csr_row_ptr`]).
pub fn bfs_upstream(db: &BrainDB, target: u64, max_hops: u32) -> Vec<Hit> {
    let n = db.header.n_neurons as usize;
    if (target as usize) >= n || max_hops == 0 {
        return Vec::new();
    }

    let row = db.csr_row_ptr();
    let col = db.csr_col_idx();
    let n_syn = col.len();

    // Build reverse adjacency.
    let mut rev_row = vec![0u64; n + 1];
    for &t in col {
        rev_row[t as usize + 1] += 1;
    }
    for i in 1..=n {
        rev_row[i] += rev_row[i - 1];
    }
    let mut rev_col = vec![0u32; n_syn];
    let mut cursor = vec![0u64; n];
    for pre in 0..n {
        let s = row[pre] as usize;
        let e = row[pre + 1] as usize;
        for &t in &col[s..e] {
            let post = t as usize;
            let pos = rev_row[post] + cursor[post];
            rev_col[pos as usize] = pre as u32;
            cursor[post] += 1;
        }
    }

    let mut visited = vec![false; n];
    let mut out: Vec<Hit> = Vec::new();
    let mut q: VecDeque<(u64, u32)> = VecDeque::new();
    visited[target as usize] = true;
    q.push_back((target, 0));

    while let Some((nid, depth)) = q.pop_front() {
        if depth >= max_hops {
            continue;
        }
        let s = rev_row[nid as usize] as usize;
        let e = rev_row[nid as usize + 1] as usize;
        for &pre in &rev_col[s..e] {
            if !visited[pre as usize] {
                visited[pre as usize] = true;
                out.push((pre as u64, depth + 1));
                q.push_back((pre as u64, depth + 1));
            }
        }
    }
    out
}

/// Strongest end-to-end path from `source` to `target`, where path strength
/// is the **product** of synapse `base_weight`s. Internally we maximise
/// `Σ log(base_weight)` via Dijkstra (only positive weights are valid).
///
/// Returns `(path_neurons, total_log_weight)` or `None` if unreachable.
pub fn strongest_path(db: &BrainDB, source: u64, target: u64) -> Option<(Vec<u64>, f32)> {
    let n = db.header.n_neurons as usize;
    if (source as usize) >= n || (target as usize) >= n {
        return None;
    }
    if source == target {
        return Some((vec![source], 0.0));
    }

    let row = db.csr_row_ptr();
    let col = db.csr_col_idx();
    let syns = db.syn_attrs();

    // Best total log-weight reaching each node; -inf = unreachable.
    let mut best = vec![f32::NEG_INFINITY; n];
    let mut prev = vec![u64::MAX; n];
    best[source as usize] = 0.0;

    // Max-heap on (total_log_w, node).
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
    heap.push(HeapEntry { score: 0.0, node: source });

    while let Some(HeapEntry { score, node }) = heap.pop() {
        if node == target {
            // Reconstruct path.
            let mut path = Vec::new();
            let mut cur = target;
            while cur != u64::MAX {
                path.push(cur);
                if cur == source { break; }
                cur = prev[cur as usize];
            }
            path.reverse();
            return Some((path, best[target as usize]));
        }
        if score < best[node as usize] {
            continue;
        }
        let s = row[node as usize] as usize;
        let e = row[node as usize + 1] as usize;
        for syn_idx in s..e {
            let nb = col[syn_idx] as usize;
            let w = syns[syn_idx].base_weight;
            if w <= 0.0 {
                continue;
            }
            let cand = score + w.ln();
            if cand > best[nb] {
                best[nb] = cand;
                prev[nb] = node;
                heap.push(HeapEntry { score: cand, node: nb as u64 });
            }
        }
    }
    None
}

// ── Heap entry with f32 score (using total_cmp for Ord) ─────────────────────

#[derive(Clone, Copy, Debug)]
struct HeapEntry {
    score: f32,
    node: u64,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Equal && self.node == other.node
    }
}
impl Eq for HeapEntry {}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.node.cmp(&other.node))
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
