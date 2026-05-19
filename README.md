<div align="center">

# 🧠 BrainGo db

**Brain simulation database → Robot driving engine — from C. elegans to humanoid**

[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

*From-scratch Rust implementation of BrainGo db — full-scale brain network simulation database that bridges biological neural circuits to physical robots — simulating brains, then driving bodies.*

English | [中文](README_CN.md)

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

### 1. Build from Source

```bash
# Clone the repository
git clone https://github.com/wanglilin/BrainGo.git
cd BrainGo

# Build and test
cargo check
cargo test
```

### 2. Initialize C. elegans Data

The 302-neuron C. elegans connectome data is **bundled** in this repository under
`data/celegans/`. The import method `load_from_dir()` is generic — it accepts any
directory following the same layout, so you can load your own connectome data too.

Bundled data structure:
```
data/celegans/
├── network/config.json                              — Cell name ↔ ID mapping
├── components/param/cell/<NAME>.json                — Per-neuron 17-channel conductances
└── components/param/connection/SI5-302.xlsx         — Synaptic & gap junction adjacency matrices
```

**Using the CLI** (requires `cli` feature):

```bash
# Build the CLI tool
cargo build --features cli --no-default-features

# Load bundled C. elegans data
braindb-cli load-worm data/celegans/ --output celegans.braindb

# Or load from any directory with the same layout
braindb-cli load-worm /path/to/my/connectome --output my_net.braindb

# Verify the loaded data
braindb-cli info celegans.braindb
# Output:
#   Neurons:        302
#   Synapses:       ~6000+
#   Gap junctions:  ~500+
#   Compartments:   604
```

**Using Rust API**:

```rust
use braindb::storage::loader::BAAIWormLoader;

// Load from bundled data directory
let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("celegans");
let loader = BAAIWormLoader::load_from_dir(&dir)?;
let db = loader.into_braindb(std::path::Path::new("celegans.braindb"))?;
println!("Loaded {} neurons", db.header.n_neurons); // 302

// Or load from any external directory with the same layout
let loader = BAAIWormLoader::load_from_dir(std::path::Path::new("/path/to/my/connectome"))?;
```

**Using Python**:

```python
import braindb

# Open a pre-built .braindb file
db = braindb.BrainDB("celegans.braindb")
print(db.neuron_count())     # 302
print(db.get_neuron_name(0)) # I1L
print(db.get_all_neuron_names())  # ['I1L', 'I1R', 'I2L', ...]
```

### 3. Run Simulation

```bash
# CLI: run 100ms simulation with current injection into neuron 0
braindb-cli run celegans.braindb -d 100 --stimulus "0:30"

# CLI: with STDP plasticity enabled
braindb-cli run celegans.braindb -d 1000 --stdp --snapshot state.snap
```

```python
# Python: run simulation
from braindb import BrainDB, Simulation

db = BrainDB("celegans.braindb")
sim = Simulation("celegans.braindb")
sim.set_neuron_input(0, 30.0)  # inject 30 pA into neuron 0
sim.run(1000)                   # 100 ms (1000 ticks × 0.1ms)
v = sim.get_neuron_voltage(0)   # read membrane voltage
```

### 4. Load Your Own Connectome

For custom networks, use the generic CSV/JSON connectome loader:

```csv
# my_connectome.csv
pre_id,post_id,weight,delay_ms,syn_type,receptor_type
0,1,1.0,1.5,1,0
0,2,0.5,2.0,1,0
1,2,-0.3,1.0,2,1
```

```rust
use braindb::storage::loader::connectome::ConnectomeLoader;
use braindb::storage::builder::BrainDBBuilder;

let mut builder = BrainDBBuilder::new();
// ... add neurons, regions, types ...
ConnectomeLoader::load_csv(std::path::Path::new("my_connectome.csv"), &mut builder)?;
let db = builder.build(std::path::Path::new("my_network.braindb"))?;
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

BrainGo db follows a dual-track strategy: **biological simulation ↔ hardware actuation**.
Each phase validates the simulation engine against real neural data, then deploys it to drive a physical robot.

| Phase | 🧬 Biological Circuit | 🔧 Hardware Circuit | Neuron Scale | Timeline |
|-------|----------------------|--------------------|-------------|----------|
| Phase 0-1 | **C. elegans** (nematode worm) | **Caterpillar robot** 🐛 | 302 | 6-12 months |
| Phase 2 | **Drosophila** (fruit fly) | **Insect robot** 🪰 | 140K | 2-4 years |
| Phase 3 | **Mouse** | **Robot dog** 🐕 | 70M | 5-10 years |
| Phase 4+ | **Human** (local / whole brain) | **Humanoid robot** 🤖 | 86B | 15-20+ years |

### How it works

```
Biological Data          BrainGo db Engine            Robot Actuation
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

### 🔮 Ultimate Vision — Consciousness Carrier

Beyond driving robots, BrainGo db aims to become the **data carrier for brain consciousness**.
When brain-computer interfaces (BCI) one day make it possible to upload human consciousness,
there must exist a database capable of faithfully storing and replaying the full state of
every neuron, every synapse, every spike — the complete neural correlate of a mind.

BrainGo db is designed from the ground up for this eventuality:

- **Mmap + CSR** → petabyte-scale brain state, byte-addressable
- **5-phase simulation** → biophysically accurate replay of uploaded neural dynamics
- **Snapshot / restore** → checkpoint and resume a living connectome
- **Plasticity (STDP / structural)** → the uploaded mind can continue to learn and evolve

> *When the day comes to upload a mind, the database that holds it must be as rigorous
> as the brain itself. BrainGo db intends to be that database.*

## 🛠️ Dependencies

**Core:** memmap2, bytemuck, thiserror, rayon, serde, postcard, calamine, rand, realfft, nalgebra, static_assertions

**Optional:** pyo3 (Python), cudarc (CUDA), sundials-sys (implicit integration), dashmap + tokio (distributed), clap + axum (CLI/server)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
