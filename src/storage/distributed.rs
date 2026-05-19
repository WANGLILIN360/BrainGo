//! Distributed simulation support — shard routing & cross-shard events.
//!
//! Design doc §16.5: multi-node simulation where each shard owns a subset
//! of brain regions. The [`ShardRouter`] maps regions → shard IDs, and
//! [`CrossShardEvent`] carries synaptic events that cross shard boundaries.
//!
//! This is a **stub module** — full implementation requires `dashmap` +
//! `tokio` (behind the `distributed` feature flag). The data structures
//! below are sufficient for single-node simulation and testing.

use crate::core::synapse::SynapseAttr;

/// A synaptic event that must be delivered to a neuron on a different shard.
#[derive(Clone, Debug)]
pub struct CrossShardEvent {
    /// Source shard ID.
    pub src_shard: u32,
    /// Target shard ID.
    pub tgt_shard: u32,
    /// Global pre-neuron ID.
    pub pre_neuron: u32,
    /// Synapse attributes (needed by the receiving shard).
    pub syn_attr: SynapseAttr,
    /// Arrival tick (simulation time).
    pub arrival_tick: u64,
    /// Conductance delta to apply.
    pub delta_g: f32,
}

/// Per-shard statistics for monitoring distributed simulation progress.
#[derive(Clone, Debug, Default)]
pub struct ShardStats {
    /// Number of neurons owned by this shard.
    pub n_neurons: u32,
    /// Number of synapses owned by this shard.
    pub n_synapses: u32,
    /// Number of cross-shard events sent this tick.
    pub events_sent: u32,
    /// Number of cross-shard events received this tick.
    pub events_received: u32,
    /// Simulation tick of this shard (may lag behind global tick).
    pub local_tick: u64,
}

/// Distributed simulation coordinator (stub).
///
/// In the full implementation, this would manage:
/// - Shard ↔ region mapping
/// - Cross-shard event queues (via tokio channels)
/// - Barrier synchronisation between shards
/// - Snapshot coordination
pub struct DistributedCoordinator {
    /// Number of shards in the simulation.
    pub n_shards: u32,
    /// Per-shard statistics.
    pub stats: Vec<ShardStats>,
}

impl DistributedCoordinator {
    /// Create a single-shard coordinator (no distribution).
    pub fn single_node() -> Self {
        Self {
            n_shards: 1,
            stats: vec![ShardStats::default()],
        }
    }

    /// Create a multi-shard coordinator.
    pub fn new(n_shards: u32) -> Self {
        Self {
            n_shards,
            stats: vec![ShardStats::default(); n_shards as usize],
        }
    }

    /// Check if all shards have reached the given tick (barrier).
    pub fn all_reached_tick(&self, tick: u64) -> bool {
        self.stats.iter().all(|s| s.local_tick >= tick)
    }
}
