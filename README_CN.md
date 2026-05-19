<div align="center">

# 🧠 BrainGo db

**大脑仿真数据库 → 机器人驱动引擎 — 从线虫到人形机器人**

[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

*从零自研的 Rust 大脑神经网络仿真数据库，将生物神经回路桥接到物理机器人——仿真大脑，驱动机体。*

[English](README.md) | 中文

</div>

---

## ✨ 核心特性

- **AoS（结构体数组）布局** — mmap 友好，缓存感知的数据设计
- **CSR 稀疏矩阵**存储突触 — O(1) 获取任意神经元的全部出连接
- **mmap 静态/动态分离** — 静态数据（NeuronAttr/SynapseAttr/CSR）mmap 只读，动态状态（NeuronState/SynapseState）常规内存 + 定期快照
- **环形延迟队列** — 仿真时零动态内存分配
- **自定义二进制格式** `.braindb` + `.braindb.snapshot` — 紧凑、快速、可移植
- **5 阶段仿真循环** + barrier 同步并行
- **Python 绑定**（PyO3）— 无缝对接科学计算 Python 生态
- **秀丽隐杆线虫 302 神经元数据集**作为参考实现

## 🏗️ 架构

### 核心数据结构（v2.4）

| 结构体 | 大小 | 对齐 | 说明 |
|--------|------|------|------|
| `NeuronAttr` | 64B | 64 | 静态神经元属性 |
| `NeuronState` | 64B | 64 | 动态神经元状态（v, u, i_total） |
| `CompartmentAttr` | 128B | 64 | 多舱室属性 |
| `CompartmentState` | 64B | 64 | 舱室动态状态 |
| `SynapseAttr` | 32B | — | 突触属性（pre_neuron 由 CSR 隐含） |
| `SynapseState` | 32B | — | 突触状态（g_rise/g_decay） |
| `GapJunction` | 24B | — | 电突触 |

### 仿真循环（5 阶段 + 并发安全）

1. **电突触（Gap Junction）** — 按脑区分片，连续更新
2. **化学突触事件到达** — 延迟队列 → g_rise/g_decay 阶跃
3. **活跃突触电导衰减** — VecDeque 列表
4. **神经元/舱室状态更新** — Izhikevich/LIF 点神经元，HH 电缆方程多舱室
5. **STDP 可塑性** — 批量 100ms，Song2000 形式

并发策略：线程局部电流缓冲 + reduce，barrier 同步

## 🚀 快速开始

### 1. 从源码构建

```bash
# 克隆仓库
git clone https://github.com/wanglilin/BrainGo.git
cd BrainGo

# 构建和测试
cargo check
cargo test
```

### 2. 初始化线虫数据

302 神经元的秀丽隐杆线虫连接组数据**已内置**在本仓库的 `data/celegans/` 目录中。
导入方法 `load_from_dir()` 是通用的——只要目录遵循相同的布局结构，就可以加载任何连接组数据。

内置数据结构：
```
data/celegans/
├── network/config.json                              — 细胞名 ↔ ID 映射
├── components/param/cell/<NAME>.json                — 每个神经元 17 通道电导
└── components/param/connection/SI5-302.xlsx         — 突触和电突触邻接矩阵
```

**使用 CLI**（需要 `cli` 特性）：

```bash
# 构建 CLI 工具
cargo build --features cli --no-default-features

# 加载内置线虫数据
braindb-cli load-worm data/celegans/ --output celegans.braindb

# 或加载任何遵循相同目录结构的外部数据
braindb-cli load-worm /path/to/my/connectome --output my_net.braindb

# 验证加载的数据
braindb-cli info celegans.braindb
# 输出：
#   Neurons:        302
#   Synapses:       ~6000+
#   Gap junctions:  ~500+
#   Compartments:   604
```

**使用 Rust API**：

```rust
use braindb::storage::loader::BAAIWormLoader;

// 从内置数据目录加载
let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("celegans");
let loader = BAAIWormLoader::load_from_dir(&dir)?;
let db = loader.into_braindb(std::path::Path::new("celegans.braindb"))?;
println!("已加载 {} 个神经元", db.header.n_neurons); // 302

// 或从任何遵循相同目录结构的外部目录加载
let loader = BAAIWormLoader::load_from_dir(std::path::Path::new("/path/to/my/connectome"))?;
```

**使用 Python**：

```python
import braindb

# 打开预构建的 .braindb 文件
db = braindb.BrainDB("celegans.braindb")
print(db.neuron_count())     # 302
print(db.get_neuron_name(0)) # I1L
print(db.get_all_neuron_names())  # ['I1L', 'I1R', 'I2L', ...]
```

### 3. 运行仿真

```bash
# CLI：运行 100ms 仿真，向神经元 0 注入电流
braindb-cli run celegans.braindb -d 100 --stimulus "0:30"

# CLI：启用 STDP 可塑性
braindb-cli run celegans.braindb -d 1000 --stdp --snapshot state.snap
```

```python
# Python：运行仿真
from braindb import BrainDB, Simulation

db = BrainDB("celegans.braindb")
sim = Simulation("celegans.braindb")
sim.set_neuron_input(0, 30.0)  # 向神经元 0 注入 30 pA
sim.run(1000)                   # 100 ms（1000 ticks × 0.1ms）
v = sim.get_neuron_voltage(0)   # 读取膜电压
```

### 4. 加载自定义连接组

对于自定义网络，使用通用 CSV/JSON 连接组加载器：

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
// ... 添加神经元、脑区、类型 ...
ConnectomeLoader::load_csv(std::path::Path::new("my_connectome.csv"), &mut builder)?;
let db = builder.build(std::path::Path::new("my_network.braindb"))?;
```

### 关于 `python` 特性

`Cargo.toml` 默认启用 `python` 特性。构建需要有效的 Python 安装（`PYO3_PYTHON` 环境变量或 `PATH` 中的 `python`）。如果不方便，可在 `Cargo.toml` 中设置：

```toml
[features]
default = []
```

重新构建即可——其余部分可独立编译。

## 📁 项目结构

```
src/
├── core/              — POD 记录 + 非 POD 描述符
├── storage/           — .braindb 格式、构建器、mmap 加载器、快照
├── sim/               — 仿真循环（5 阶段引擎）
├── query/             — 查询引擎
├── bin/               — CLI 和服务器二进制
└── pyo3_bindings.rs   — Python 绑定（`python` 特性门控）

python/
├── braindb/           — Python 包
└── tests/             — Python 测试套件

tests/
├── test_sizes.rs               — POD 大小/对齐断言
├── test_builder_roundtrip.rs   — 数据库构建 + 往返 + 快照
├── test_sim_basic.rs           — 仿真循环测试
├── test_izhikevich.rs          — Izhikevich 神经元模型
├── test_stdp.rs                — STDP 可塑性
└── ...                         — 更多集成测试
```

## 🗺️ 路线图 — 从大脑到机器人

BrainGo db 采用双轨策略：**生物仿真 ↔ 硬件驱动**。
每个阶段先用真实神经数据验证仿真引擎，再将其部署到物理机器人。

| 阶段 | 🧬 生物线路 | 🔧 硬件线路 | 神经元规模 | 时间线 |
|------|-----------|-----------|-----------|--------|
| 阶段 0-1 | **秀丽隐杆线虫**（线虫） | **毛毛虫机器人** 🐛 | 302 | 6-12 个月 |
| 阶段 2 | **黑腹果蝇** | **昆虫机器人** 🪰 | 14 万 | 2-4 年 |
| 阶段 3 | **小鼠** | **机器狗** 🐕 | 7000 万 | 5-10 年 |
| 阶段 4+ | **人类**（局部/全脑） | **人形机器人** 🤖 | 860 亿 | 15-20+ 年 |

### 工作原理

```
生物数据                BraindGo db 引擎              机器人执行
────────────  ──▶  ──────────────────  ──▶  ─────────────────
连接组 /              仿真循环：                   运动神经元输出 →
电生理数据            5 阶段并行步进               舵机 / 执行器 / PID
                      ↓                            ↓
                      脉冲 → 肌肉映射              实时控制回路
```

- **阶段 0-1** 已在进行中：秀丽隐杆线虫 302 神经元连接组已加载，
  运动神经元 → 肌肉映射（48 块肌肉）已验证，与 BAAIWorm 3D 渲染引擎的桥接已可用。
- 后续每个阶段增加神经元数量、可塑性复杂度和实时约束——数据库引擎通过 mmap + CSR + rayon 实现扩展。

## 🛠️ 依赖

**核心：** memmap2, bytemuck, thiserror, rayon, serde, postcard, calamine, rand, realfft, nalgebra, static_assertions

**可选：** pyo3（Python）、cudarc（CUDA）、sundials-sys（隐式积分）、dashmap + tokio（分布式）、clap + axum（CLI/服务器）

## 🤝 参与贡献

欢迎贡献代码！请随时提交 Pull Request。

## 📄 许可证

依据 [Apache 许可证 2.0 版](LICENSE-APACHE) 或 [MIT 许可证](LICENSE-MIT) 任选其一授权。

除非您另有明确声明，否则您有意提交以包含在本作品中的任何贡献，均按 Apache-2.0 许可证的定义，以上述双重许可方式授权，不附加任何额外条款或条件。
