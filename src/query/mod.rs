//! Query engine — structural and functional analyses on a [`BrainDB`].
//!
//! See `braindb-design.md` §8. This module currently implements:
//!
//! - N-th order BFS downstream / upstream (`connectivity::bfs_*`)
//! - Strongest-product path between two neurons (`connectivity::strongest_path`)
//! - Region-mean LFP estimate from a `Simulation` snapshot
//!   (`oscillation::region_mean_lfp`)
//!
//! Spectral analysis (FFT, peak detection, functional-connectivity matrix)
//! and graph-theoretic metrics (clustering, modularity) land in M5+.

pub mod connectivity;
pub mod oscillation;
pub mod region_query;

pub use connectivity::{bfs_downstream, bfs_upstream, strongest_path, Hit};
pub use oscillation::region_mean_lfp;
pub use region_query::{region_pathway_info, outgoing_pathways, incoming_pathways, pathways_between, region_connectivity_matrix, RegionPathwayInfo};
