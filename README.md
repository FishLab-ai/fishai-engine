# 🧠 TinyAI Engine

> 超轻量自研 GPT 推理引擎 — Rust 实现，4-bit 量化，无需 Git LFS

## 概述

TinyAI Engine 是一个**完全从零自研**的 GPT 推理引擎，使用纯 Rust 编写。它实现了完整的 Transformer 架构，包括多头自注意力机制、前馈神经网络、层归一化，以及 4-bit 整数量化。

### 核心特点

| 特性 | 描述 |
|------|------|
| 🦀 语言 | Rust (编译型，高性能，内存安全) |
| 🧮 架构 | GPT-2 风格 Decoder-Only Transformer |
| 📦 量化 | INT4 Per-Channel 量化，权重仅 ~25MB |
| 🚫 无 LFS | 量化权重直接放进 Git 仓库 |
| ⚡ 推理 | 自研前向传播 + 温度采样 |
| 🌐 API | axum HTTP 服务，支持 REST 调用 |

### 模型参数

```
d_model:     512
n_heads:     8
n_layers:    6
d_ff:        2048
vocab_size:  32000
max_seq_len: 512
总参数量:    ~52M
4-bit 量化:  ~25MB
```

## 项目结构

```
src/
├── main.rs       # 入口：启动 HTTP 服务器
├── model.rs      # GPT 模型架构（注意力、FFN、LayerNorm、采样）
├── quantize.rs   # INT4 量化/解量化
├── tokenizer.rs  # BPE 分词器
└── api.rs        # HTTP API (axum)
```

## 自研组件详解

### 1. Multi-Head Self-Attention
- Q, K, V 线性投影
- Scaled Dot-Product Attention
- 因果掩码 (下三角矩阵)
- 多头并行计算 + 输出投影

### 2. Feed-Forward Network
- 两层 MLP: `Linear → GELU → Linear`
- GELU 激活函数近似实现

### 3. Layer Normalization
- 标准 LayerNorm: `(x - μ) / √(σ² + ε) * γ + β`

### 4. INT4 量化方案
- 对称量化: `value = (int4 - zero_point) * scale`
- Per-Channel: 每个输出通道独立 scale 和 zero_point
- 紧凑存储: 每个 u8 存储 2 个 4-bit 值

### 5. BPE 分词器
- Byte-Pair Encoding 算法
- 支持中英文混合文本
- Byte-level 基础词汇 + 可训练的 merge 规则

## 构建 & 运行

```bash
# 编译
cargo build --release

# 运行 (需要先训练模型，见 tinyai-train)
mkdir -p weights
./target/release/tinyai-engine
```

## API 端点

```
POST /api/chat         - 非流式对话
POST /api/chat/stream  - 流式对话
GET  /api/model        - 模型信息
GET  /health           - 健康检查
```

## 许可证

MIT License - FishLab-ai
