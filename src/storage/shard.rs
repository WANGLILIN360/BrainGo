//! Shard storage — per-region file-level partitioning for distributed simulation.
//!
//! Design (from `braindb-design.md` §10.1):
//! - Each shard stores one brain region's neurons + synapses + gap junctions
//! - Cross-shard communication via async delayed events (chemical synapses)
//! - Gap junctions restricted to single shard (v2.4 design decision)
//! - Long-range pathways carry batched delay events between shards
//!
//! This module defines the shard file format and local simulation context.
//! Full distributed orchestration (dashmap + tokio) is behind the
//! `distributed` feature flag.

/// Shard-local metadata header (written at the start of each shard file).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShardHeader {
    /// File magic: `BRSH`
    pub magic: [u8; 4],
    /// Format version (currently 1).
    pub version: u8,
    /// Reserved padding.
    pub _pad: [u8; 3],
    /// Shard index (0-based).
    pub shard_id: u32,
    /// Number of neurons in this shard.
    pub n_neurons: u32,
    /// Number of synapses (outgoing from this shard's neurons).
    pub n_synapses: u32,
    /// Number of gap junctions (internal to this shard).
    pub n_gap_junctions: u32,
    /// Number of compartments in this shard.
    pub n_compartments: u32,
    /// Region ID this shard covers.
    pub region_id: u32,
    /// Byte offset to neuron attributes segment.
    pub off_neuron_attr: u64,
    /// Byte offset to synapse attributes segment.
    pub off_syn_attr: u64,
    /// Byte offset to gap junction segment.
    pub off_gap: u64,
    /// Byte offset to compartment attributes segment.
    pub off_comp_attr: u64,
    /// Reserved for future use.
    pub _reserved: [u64; 8],
}

const _SHARD_MAGIC: [u8; 4] = [b'B', b'R', b'S', b'H'];
const _SHARD_VERSION: u8 = 1;

/// A cross-shard event — a synaptic event destined for a neuron in another shard.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CrossShardEvent {
    /// Target shard ID.
    pub target_shard: u32,
    /// Target neuron ID within the target shard.
    pub target_neuron: u32,
    /// Synapse index in the source shard's CSR.
    pub source_syn_idx: u32,
    /// Explicit padding before u64 field.
    pub _pad0: u32,
    /// Arrival tick.
    pub arrival_tick: u64,
    /// Conductance delta (nS).
    pub delta_g: f32,
    /// Padding.
    pub _pad: f32,
}

/// Shard routing table — maps region_id → shard_id.
#[derive(Clone, Debug)]
pub struct ShardRouter {
    /// `region_to_shard[region_id] = shard_id`.
    pub region_to_shard: Vec<u32>,
}

impl ShardRouter {
    /// Create a router from a list of `(region_id, shard_id)` pairs.
    pub fn new(mappings: &[(u32, u32)]) -> Self {
        let max_region = mappings.iter().map(|(r, _)| *r).max().unwrap_or(0);
        let mut table = vec![u32::MAX; max_region as usize + 1];
        for &(region, shard) in mappings {
            table[region as usize] = shard;
        }
        Self { region_to_shard: table }
    }

    /// Look up the shard for a given region. Returns `None` if not found.
    pub fn shard_for_region(&self, region_id: u32) -> Option<u32> {
        self.region_to_shard.get(region_id as usize).copied().filter(|&s| s != u32::MAX)
    }

    /// Check if two regions are in the same shard.
    pub fn same_shard(&self, region_a: u32, region_b: u32) -> bool {
        self.shard_for_region(region_a) == self.shard_for_region(region_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_router() {
        let router = ShardRouter::new(&[(0, 0), (1, 0), (2, 1), (3, 1)]);
        assert_eq!(router.shard_for_region(0), Some(0));
        assert_eq!(router.shard_for_region(2), Some(1));
        assert_eq!(router.shard_for_region(99), None);
        assert!(router.same_shard(0, 1));
        assert!(!router.same_shard(0, 2));
    }
}
