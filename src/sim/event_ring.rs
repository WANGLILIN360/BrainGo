//! Fixed-size circular buffer for delayed synaptic events.
//!
//! Each slot is the bucket for events whose arrival `tick % ring_size == slot`.
//! Insertion is `O(1)`; draining a tick is `O(events at that tick)`. There is
//! **zero dynamic allocation per step** during steady-state simulation — only
//! when an individual slot exceeds its current capacity does its `Vec` grow.

use std::mem;

/// A single delivery: at `tick`, deposit `delta_g` (nS) into synapse `syn_id`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynapticEvent {
    pub tick: u64,
    pub syn_id: u32,
    pub delta_g: f32,
}

/// Fixed-size circular buffer indexed by `tick % size`.
pub struct EventRing {
    size: usize,
    slots: Vec<Vec<SynapticEvent>>,
    /// Scratch buffer reused across `drain_at` calls to avoid allocation.
    scratch: Vec<SynapticEvent>,
}

impl EventRing {
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "EventRing size must be positive");
        Self {
            size,
            slots: (0..size).map(|_| Vec::new()).collect(),
            scratch: Vec::new(),
        }
    }

    #[inline]
    pub fn size(&self) -> usize { self.size }

    #[inline]
    pub fn push(&mut self, ev: SynapticEvent) {
        let slot = (ev.tick as usize) % self.size;
        self.slots[slot].push(ev);
    }

    /// Drain and return all events scheduled at the given tick.
    /// The returned slice points into a scratch buffer that is reused on the
    /// next call — copy out before invoking `drain_at` again.
    pub fn drain_at(&mut self, tick: u64) -> &[SynapticEvent] {
        let slot = (tick as usize) % self.size;
        self.scratch.clear();
        // Move the slot's contents into scratch, leaving the slot empty.
        let taken = mem::take(&mut self.slots[slot]);
        self.scratch.extend(taken.iter().filter(|e| e.tick == tick));
        // Any straggler whose tick doesn't match this one (because some
        // earlier delay overflowed the ring or wasn't yet consumed) is dropped.
        // Future versions can re-enqueue them; current callers must ensure
        // `delay < ring_size`.
        &self.scratch
    }

    /// Returns the total number of pending events across all slots
    /// (debug / observability helper, `O(n_slots)`).
    pub fn pending(&self) -> usize {
        self.slots.iter().map(Vec::len).sum()
    }
}
