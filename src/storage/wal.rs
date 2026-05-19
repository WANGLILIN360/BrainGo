//! Write-Ahead Log — crash-recovery for dynamic state mutations.
//!
//! Design (from `braindb-design.md` §4):
//! - Append-only binary log of state mutations (neuron/synapse/compartment)
//! - Checkpoint = full snapshot + WAL truncation
//! - Recovery = load last snapshot + replay WAL entries
//! - Each entry: `[u8 opcode, u64 tick, u32 entity_id, payload...]`
//! - File magic: `BRWL`, version 1

use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

use crate::core::compartment::CompartmentState;
use crate::core::neuron::NeuronState;
use crate::core::synapse::SynapseState;

// ── Opcodes ──────────────────────────────────────────────────────────────

const WAL_MAGIC: [u8; 4] = [b'B', b'R', b'W', b'L'];
const WAL_VERSION: u8 = 1;

/// Opcodes for WAL entries.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalOp {
    /// NeuronState mutation: `[NeuronState]`
    NeuronState = 1,
    /// SynapseState mutation: `[u32 syn_id, SynapseState]`
    SynapseState = 2,
    /// CompartmentState mutation: `[u32 comp_id, CompartmentState]`
    CompartmentState = 3,
    /// Gap junction weight change: `[u32 gj_idx, f32 weight]`
    GapJunctionWeight = 4,
    /// Region modulation change: `[u32 region_id, ModulationLevel]`
    RegionModulation = 5,
    /// Neuron killed (flags change): `[u32 neuron_id, u8 new_flags]`
    NeuronFlags = 6,
    /// Checkpoint marker — all prior entries are captured in a snapshot.
    Checkpoint = 0xFF,
}

impl TryFrom<u8> for WalOp {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            1 => Ok(Self::NeuronState),
            2 => Ok(Self::SynapseState),
            3 => Ok(Self::CompartmentState),
            4 => Ok(Self::GapJunctionWeight),
            5 => Ok(Self::RegionModulation),
            6 => Ok(Self::NeuronFlags),
            0xFF => Ok(Self::Checkpoint),
            other => Err(other),
        }
    }
}

// ── WAL Writer ───────────────────────────────────────────────────────────

/// Append-only WAL writer. Wraps a `BufWriter<File>`.
pub struct WalWriter {
    writer: BufWriter<std::fs::File>,
    tick: u64,
    bytes_written: u64,
}

impl WalWriter {
    /// Create a new WAL file. Writes the 8-byte header `[BRWL, version, 3×pad]`.
    pub fn create(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&WAL_MAGIC)?;
        writer.write_all(&[WAL_VERSION, 0, 0, 0])?; // version + 3 pad bytes
        Ok(Self {
            writer,
            tick: 0,
            bytes_written: 8,
        })
    }

    /// Set the current tick (written into each entry header).
    pub fn set_tick(&mut self, tick: u64) {
        self.tick = tick;
    }

    /// Write a raw WAL entry: `[op, tick(8B), id(4B), payload...]`.
    fn write_entry(&mut self, op: WalOp, id: u32, payload: &[u8]) -> io::Result<()> {
        self.writer.write_all(&[op as u8])?;
        self.writer.write_all(&self.tick.to_le_bytes())?;
        self.writer.write_all(&id.to_le_bytes())?;
        self.writer.write_all(payload)?;
        self.bytes_written += 1 + 8 + 4 + payload.len() as u64;
        Ok(())
    }

    /// Log a neuron state mutation.
    pub fn log_neuron_state(&mut self, neuron_id: u32, state: &NeuronState) -> io::Result<()> {
        let bytes: &[u8] = bytemuck::cast_slice(std::slice::from_ref(state));
        self.write_entry(WalOp::NeuronState, neuron_id, bytes)
    }

    /// Log a synapse state mutation.
    pub fn log_synapse_state(&mut self, syn_id: u32, state: &SynapseState) -> io::Result<()> {
        let bytes: &[u8] = bytemuck::cast_slice(std::slice::from_ref(state));
        self.write_entry(WalOp::SynapseState, syn_id, bytes)
    }

    /// Log a compartment state mutation.
    pub fn log_compartment_state(
        &mut self,
        comp_id: u32,
        state: &CompartmentState,
    ) -> io::Result<()> {
        let bytes: &[u8] = bytemuck::cast_slice(std::slice::from_ref(state));
        self.write_entry(WalOp::CompartmentState, comp_id, bytes)
    }

    /// Log a gap-junction weight change.
    pub fn log_gap_junction_weight(&mut self, gj_idx: u32, weight: f32) -> io::Result<()> {
        self.write_entry(WalOp::GapJunctionWeight, gj_idx, &weight.to_le_bytes())
    }

    /// Log a region modulation change.
    pub fn log_region_modulation(
        &mut self,
        region_id: u32,
        mod_level: &crate::core::neuromodulator::ModulationLevel,
    ) -> io::Result<()> {
        let bytes: &[u8] = bytemuck::cast_slice(std::slice::from_ref(mod_level));
        self.write_entry(WalOp::RegionModulation, region_id, bytes)
    }

    /// Log a neuron flags change.
    pub fn log_neuron_flags(&mut self, neuron_id: u32, flags: u8) -> io::Result<()> {
        self.write_entry(WalOp::NeuronFlags, neuron_id, &[flags])
    }

    /// Write a checkpoint marker. After this, the caller should save a
    /// full snapshot and truncate the WAL.
    pub fn write_checkpoint(&mut self) -> io::Result<()> {
        self.writer.write_all(&[WalOp::Checkpoint as u8])?;
        self.writer.write_all(&self.tick.to_le_bytes())?;
        self.bytes_written += 9;
        Ok(())
    }

    /// Flush buffered writes to disk.
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
}

// ── WAL Reader / Replay ──────────────────────────────────────────────────

/// A single deserialized WAL entry.
#[derive(Clone, Debug)]
pub enum WalEntry {
    NeuronState { tick: u64, neuron_id: u32, state: NeuronState },
    SynapseState { tick: u64, syn_id: u32, state: SynapseState },
    CompartmentState { tick: u64, comp_id: u32, state: CompartmentState },
    GapJunctionWeight { tick: u64, gj_idx: u32, weight: f32 },
    RegionModulation {
        tick: u64,
        region_id: u32,
        modulation: crate::core::neuromodulator::ModulationLevel,
    },
    NeuronFlags { tick: u64, neuron_id: u32, flags: u8 },
    Checkpoint { tick: u64 },
}

/// Read and parse all entries from a WAL file.
pub fn read_wal(path: &Path) -> io::Result<Vec<WalEntry>> {
    let mut file = std::fs::File::open(path)?;
    // Validate header.
    let mut hdr = [0u8; 8];
    file.read_exact(&mut hdr)?;
    if hdr[..4] != WAL_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "WAL: invalid magic",
        ));
    }
    if hdr[4] != WAL_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("WAL: unsupported version {}", hdr[4]),
        ));
    }

    let mut entries = Vec::new();
    let mut buf = [0u8; 8 + 4]; // tick + id

    loop {
        let mut op_byte = [0u8; 1];
        if file.read_exact(&mut op_byte).is_err() {
            break; // EOF
        }
        let op = WalOp::try_from(op_byte[0]).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("WAL: unknown opcode {}", op_byte[0]),
            )
        })?;

        if op == WalOp::Checkpoint {
            file.read_exact(&mut buf[..8])?;
            let tick = u64::from_le_bytes(buf[..8].try_into().unwrap());
            entries.push(WalEntry::Checkpoint { tick });
            continue;
        }

        file.read_exact(&mut buf)?;
        let tick = u64::from_le_bytes(buf[..8].try_into().unwrap());
        let id = u32::from_le_bytes(buf[8..12].try_into().unwrap());

        let entry = match op {
            WalOp::NeuronState => {
                let mut state_bytes = [0u8; std::mem::size_of::<NeuronState>()];
                file.read_exact(&mut state_bytes)?;
                let state: NeuronState = *bytemuck::from_bytes(&state_bytes);
                WalEntry::NeuronState { tick, neuron_id: id, state }
            }
            WalOp::SynapseState => {
                let mut state_bytes = [0u8; std::mem::size_of::<SynapseState>()];
                file.read_exact(&mut state_bytes)?;
                let state: SynapseState = *bytemuck::from_bytes(&state_bytes);
                WalEntry::SynapseState { tick, syn_id: id, state }
            }
            WalOp::CompartmentState => {
                let mut state_bytes = [0u8; std::mem::size_of::<CompartmentState>()];
                file.read_exact(&mut state_bytes)?;
                let state: CompartmentState = *bytemuck::from_bytes(&state_bytes);
                WalEntry::CompartmentState { tick, comp_id: id, state }
            }
            WalOp::GapJunctionWeight => {
                let mut w_bytes = [0u8; 4];
                file.read_exact(&mut w_bytes)?;
                let weight = f32::from_le_bytes(w_bytes);
                WalEntry::GapJunctionWeight { tick, gj_idx: id, weight }
            }
            WalOp::RegionModulation => {
                let mut m_bytes =
                    [0u8; std::mem::size_of::<crate::core::neuromodulator::ModulationLevel>()];
                file.read_exact(&mut m_bytes)?;
                let modulation: crate::core::neuromodulator::ModulationLevel =
                    *bytemuck::from_bytes(&m_bytes);
                WalEntry::RegionModulation { tick, region_id: id, modulation }
            }
            WalOp::NeuronFlags => {
                let mut f = [0u8; 1];
                file.read_exact(&mut f)?;
                WalEntry::NeuronFlags { tick, neuron_id: id, flags: f[0] }
            }
            WalOp::Checkpoint => unreachable!(),
        };
        entries.push(entry);
    }

    Ok(entries)
}
