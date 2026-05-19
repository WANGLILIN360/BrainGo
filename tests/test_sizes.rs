//! Compile-time and runtime size-assertion tests for all POD records.

use std::mem::{align_of, size_of};

use braindb::*;
use braindb::storage::format::{Header, HEADER_SIZE, SnapshotHeader};

#[test]
fn neuron_attr_size_64() {
    assert_eq!(size_of::<NeuronAttr>(), 64);
    assert_eq!(align_of::<NeuronAttr>(), 64);
}

#[test]
fn neuron_state_size_64() {
    assert_eq!(size_of::<NeuronState>(), 64);
    assert_eq!(align_of::<NeuronState>(), 64);
}

#[test]
fn compartment_attr_size_128() {
    assert_eq!(size_of::<CompartmentAttr>(), 128);
    assert_eq!(align_of::<CompartmentAttr>(), 64);
}

#[test]
fn compartment_state_size_64() {
    assert_eq!(size_of::<CompartmentState>(), 64);
    assert_eq!(align_of::<CompartmentState>(), 64);
}

#[test]
fn synapse_attr_size_32() {
    assert_eq!(size_of::<SynapseAttr>(), 32);
}

#[test]
fn synapse_state_size_32() {
    assert_eq!(size_of::<SynapseState>(), 32);
}

#[test]
fn gap_junction_size_24() {
    assert_eq!(size_of::<GapJunction>(), 24);
}

#[test]
fn receptor_params_size_32() {
    assert_eq!(size_of::<ReceptorParams>(), 32);
}

#[test]
fn header_size_512() {
    assert_eq!(size_of::<Header>(), HEADER_SIZE);
    assert_eq!(HEADER_SIZE, 512);
}

#[test]
fn snapshot_header_size_40() {
    assert_eq!(size_of::<SnapshotHeader>(), 40);
}

#[test]
fn brain_region_alignment_ok() {
    // Pod requires no implicit padding; size must be a multiple of alignment.
    assert_eq!(size_of::<BrainRegion>() % align_of::<BrainRegion>(), 0);
}

#[test]
fn long_range_pathway_alignment_ok() {
    assert_eq!(size_of::<LongRangePathway>() % align_of::<LongRangePathway>(), 0);
}
