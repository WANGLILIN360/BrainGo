<div align="center">

# 🧠 BraindGo db

**Brain simulation database → Robot driving engine — from C. elegans to humanoid**

[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

*From-scratch Rust implementation of a brain-network simulation database that bridges biological neural circuits to physical robots — simulating brains, then driving bodies.*

</div>

---

## ✨ Key Features

- **AoS (Array of Structures) layout** — mmap-friendly, cache-aware data design
- **CSR sparse matrix** for synapses — O(1) access to any neuron's full outgoing connections
- **mmap static/dynamic separation** — static data (NeuronAttr/SynapseAttr/CSR) as read-only mmap, dynamic state (NeuronState/SynapseState) in regular memory with periodic snapshots
- **Ring-buffer delay queue** — zero dynamic memory allocation during simulation
- **Custom binary format** `.braindb` + `.braindb.snapshot` — compact, fast, portable
- **5-phase simulation loop** with barrier-synchronized parallelism
- **Python bindings** via PyO3 — seamless integration with scientific Python ecosystem
- **C. elegans 302-neuron dataset** included as reference implementation

## 🏗️ Architecture

### Core Data Structures (v2.4)

| Structure | Size | Alignment | Description |
|-----------|------|-----------|-------------|
| `NeuronAttr` | 64B | 64 | Static neuron attributes |
| `NeuronState` | 64B | 64 | Dynamic neuron state (v, u, i_total) |
| `CompartmentAttr` | 128B | 64 | Multi-compartment attributes |
| `CompartmentState` | 64B | 64 | Compartment dynamic state |
| `SynapseAttr` | 32B | — | Synapse attributes (pre_neuron via CSR) |
| `SynapseState` | 32B | — | Synapse state (g_rise/g_decay) |
| `GapJunction` | 24B | — | Electrical synapse |

### Simulation Loop (5 Phase + Concurrency Safety)

1. **Gap junction** — per-region sharded, sequential update
2. **Chemical synapse event arrival** — delay queue → g_rise/g_decay step
3. **Active synapse conductance decay** — VecDeque list
4. **Neuron/compartment state update** — Izhikevich/LIF point neurons, HH cable equation multi-compartment
5. **STDP plasticity** — batch 100ms, Song2000 form

Concurrency: thread-local current buffers + reduce, barrier synchronization

## 🚀 Quick Start

### Build from Source

```bash
# Clone the repository
git clone https://github.com/wanglilin/BraindGo.git
cd BraindGo

# Build and test
cargo check
cargo test
```

### Python Bindings

```bash
# Install with maturin
pip install maturin
maturin develop --release

# Use in Python
import braindb
db = braindb.BrainDB.open("celegans.braindb")
print(db.neuron_count())  # 302
print(db.get_neuron_name(0))  # I1L
```

### About the `python` Feature

`Cargo.toml` enables the `python` feature by default. Building it requires a
working Python installation (`PYO3_PYTHON` env var or `python` on `PATH`). If
that is inconvenient, set:

```toml
[features]
default = []
```

in `Cargo.toml` and rebuild — the rest of the crate compiles standalone.

## 📁 Project Layout

```
src/
├── core/              — POD records + non-POD descriptors
├── storage/           — .braindb format, builder, mmap loader, snapshots
├── sim/               — simulation loop (5-phase engine)
├── query/             — query engine
├── bin/               — CLI and server binaries
└── pyo3_bindings.rs   — Python bindings (gated by `python` feature)

python/
├── braindb/           — Python package
└── tests/             — Python test suite

tests/
├── test_sizes.rs               — POD size/alignment assertions
├── test_builder_roundtrip.rs   — DB build + round-trip + snapshot
├── test_sim_basic.rs           — Simulation loop tests
├── test_izhikevich.rs          — Izhikevich neuron model
├── test_stdp.rs                — STDP plasticity
└── ...                         — More integration tests
```

## 🗺️ Roadmap — From Brain to Robot

BraindGo db follows a dual-track strategy: **biological simulation ↔ hardware actuation**.
Each phase validates the simulation engine against real neural data, then deploys it to drive a physical robot.

| Phase | 🧬 Biological Circuit | 🔧 Hardware Circuit | Neuron Scale | Timeline |
|-------|----------------------|--------------------|-------------|----------|
| Phase 0-1 | **C. elegans** (nematode worm) | **Caterpillar robot** 🐛 | 302 | 6-12 months |
| Phase 2 | **Drosophila** (fruit fly) | **Insect robot** 🪰 | 140K | 2-4 years |
| Phase 3 | **Mouse** | **Robot dog** 🐕 | 70M | 5-10 years |
| Phase 4+ | **Human** (local / whole brain) | **Humanoid robot** 🤖 | 86B | 15-20+ years |

### How it works

```
Biological Data          BraindGo db Engine           Robot Actuation
──────────────  ──▶  ──────────────────────  ──▶  ─────────────────
Connectome /         Simulation loop:            Motor neuron output →
Electrophysiology    5-phase parallel step        Servo / Actuator / PID
                     ↓                            ↓
                     Spike → Muscle mapping       Real-time control loop
```

- **Phase 0-1** is already underway: C. elegans 302-neuron connectome is loaded,
  motor neuron → muscle mapping (48 muscles) is verified, and the bridge to
  the BAAIWorm 3D rendering engine is functional.
- Each subsequent phase adds neuron count, plasticity complexity, and
  real-time constraints — the database engine scales via mmap + CSR + rayon.

## 🛠️ Dependencies

**Core:** memmap2, bytemuck, thiserror, rayon, serde, postcard, calamine, rand, realfft, nalgebra, static_assertions

**Optional:** pyo3 (Python), cudarc (CUDA), sundials-sys (implicit integration), dashmap + tokio (distributed), clap + axum (CLI/server)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
