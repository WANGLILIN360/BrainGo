//! Neuron name ↔ ID bidirectional map (for connectome loaders).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NameMap {
    pub name_to_id: HashMap<String, u64>,
    pub id_to_name: HashMap<u64, String>,
}

impl NameMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, id: u64) {
        let name = name.into();
        self.name_to_id.insert(name.clone(), id);
        self.id_to_name.insert(id, name);
    }

    pub fn name(&self, id: u64) -> Option<&str> {
        self.id_to_name.get(&id).map(String::as_str)
    }

    pub fn id(&self, name: &str) -> Option<u64> {
        self.name_to_id.get(name).copied()
    }

    pub fn len(&self) -> usize {
        self.name_to_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.name_to_id.is_empty()
    }
}

/// Stable FNV-1a 64-bit hash for `name_hash` fields (e.g. `BrainRegion.name_hash`).
pub fn name_hash(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}
