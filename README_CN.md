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

## 🧬 大脑 → 肌肉链路

BrainGo db 不仅仅是在隔离状态下仿真神经元——它驱动物理身体。
完整的闭环链路已在 C. elegans 上与 [BAAIWorm](https://github.com/Jessie940611/BAAIWorm) 项目验证通过：

```
┌──────────────────────────────────────────────────────────────────────┐
│                        闭环控制周期                                   │
│                                                                      │
│  感觉输入 ──▶ BrainGo db 神经仿真 ──▶ 运动神经元电压                │
│     ▲                                    │                           │
│     │                                    ▼                           │
│  身体状态 ◀── FEM 身体 (Metaworm) ◀── 肌肉激活                      │
│     ▲                  ▲                  ▲                         │
│     │                  │                  │                           │
│   环境             物理仿真         CNN / Sigmoid                    │
└──────────────────────────────────────────────────────────────────────┘
```

### 逐步流程

1. **感觉输入 → 神经仿真**
   - 感觉神经元（AWA、AWC、ASH、PLM 等）接收刺激电流
   - BrainGo db 运行 5 阶段仿真循环（电突触 → 化学突触 → 衰减 → 更新 → STDP）
   - 302 个神经元通过 ~6000 个化学突触 + ~500 个电突触传播活动

2. **运动神经元电压 → 肌肉激活**
   - 80 个运动神经元（VA、VB、DA、DB、VD、DD、AS、RM\*）产生膜电压
   - 两种转换路径：
     - **Sigmoid**（生物物理）：`activation = σ((V_mem - V_threshold) / scale)` — 直接映射电压到肌肉收缩
     - **CNN**（学习得到）：`Conv1d(80→96, kernel=21)` 在运动神经元电压历史滑动窗口 → 96 个肌肉信号，后处理 `(output + 80) / 100`

3. **肌肉激活 → 身体运动**
   - 48 块体肌（24 背侧 + 24 腹侧）+ 头部肌肉接收激活水平
   - 指令中间神经元（AVB=前进、AVA=后退）沿身体产生行波
   - Metaworm FEM 物理仿真产生身体形态和位置

4. **身体状态 → 感觉反馈**（闭环）
   - 身体位置决定哪些感觉神经元被刺激（如趋化梯度、触觉）
   - 形成闭环：线虫向引诱剂导航

### 训练（eworm_learn）

神经网络**不是天生就能产生正确行为的**——突触权重必须经过训练。
BAAIWorm 的 `eworm_learn` 模块执行此训练：

- **方法**：基于梯度的突触电导优化，使用传递阻抗（核）方法
- **目标**：使仿真运动神经元电压的相关矩阵与实验记录的匹配
- **规模**：跨 100 个 epoch 优化 ~6000+ 个突触权重，使用多 GPU（CuPy）加速
- **结果**：训练后，网络产生真实的锯齿形趋化爬行行为

在 BrainGo db 管线中，训练后的权重通过 `BAAIWormLoader` 直接加载到 `.braindb` 文件中，
因此数据库已包含优化后的连接组。

### 关键模块

| 模块 | 作用 | 来源 |
|------|------|------|
| `BrainDBNeuronSim` | 通过 pyo3 调用 Rust 神经仿真 | `baaiworm_bridge/braindb_sim/` |
| `BrainDBCircuit` | 神经元访问、刺激注入 | `baaiworm_bridge/braindb_sim/` |
| `MuscleInterface` | 运动电压 → 肌肉激活（sigmoid） | `baaiworm_bridge/braindb_sim/` |
| `CNN2Model` | 运动电压 → 肌肉激活（学习 CNN） | `baaiworm_bridge/control/` |
| `BrainDBNeuralDriver` | 完整控制回路：刺激→仿真→肌肉 | `baaiworm_bridge/control/` |
| `neuronXcore` | 3D 可视化 + C++ 桥接 | `BAAIWorm/neuronXcore/` |

### 快速运行

```bash
# 仅神经仿真（无身体）
python -m braindb_sim.run_baaiworm_braindb --braindb celegans.braindb --neural-only --duration 1000

# 完整闭环 + Metaworm 身体
python -m braindb_sim.run_baaiworm_braindb --braindb celegans.braindb --duration 5000

# 带感觉刺激
python -m braindb_sim.run_baaiworm_braindb --braindb celegans.braindb --stimulus AWAL=10 AVBL=5
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

### 🔮 终极愿景 — 意识载体

超越驱动机器人，BrainGo db 的终极目标是成为**大脑意识的数据载体**。
当脑机接口（BCI）有一天使上传人类意识成为可能时，必须存在一个数据库，
能够忠实地存储和回放每一个神经元、每一个突触、每一个脉冲的完整状态——意识的完整神经关联。

BrainGo db 从底层设计就为这一天做准备：

- **Mmap + CSR** → PB 级大脑状态，字节可寻址
- **5 阶段仿真** → 生物物理精确回放上传的神经动力学
- **快照 / 恢复** → 检查点与恢复一个活的连接组
- **可塑性（STDP / 结构性）** → 上传的意识可以继续学习和演化

> *当上传意识的那一天到来，承载它的数据库必须像大脑本身一样严谨。BrainGo db 愿做那个数据库。*

## 🛠️ 依赖

**核心：** memmap2, bytemuck, thiserror, rayon, serde, postcard, calamine, rand, realfft, nalgebra, static_assertions

**可选：** pyo3（Python）、cudarc（CUDA）、sundials-sys（隐式积分）、dashmap + tokio（分布式）、clap + axum（CLI/服务器）

## 🤝 参与贡献

欢迎贡献代码！请随时提交 Pull Request。

## 📄 许可证

依据 [Apache 许可证 2.0 版](LICENSE-APACHE) 或 [MIT 许可证](LICENSE-MIT) 任选其一授权。

除非您另有明确声明，否则您有意提交以包含在本作品中的任何贡献，均按 Apache-2.0 许可证的定义，以上述双重许可方式授权，不附加任何额外条款或条件。
