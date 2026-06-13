//! TinyAI Engine - 超轻量自研 GPT 推理引擎
//!
//! 完全自研的 Transformer 架构实现：
//! - GPT-2 风格 Decoder-Only Transformer
//! - 多头自注意力机制 (Multi-Head Self-Attention)
//! - 前馈神经网络 (Feed-Forward Network)
//! - 层归一化 (Layer Normalization)
//! - 4-bit 整数量化 (INT4 Quantization)
//! - BPE 分词器
//!
//! 架构参数 (~10M 参数)：
//! - d_model: 512
//! - n_heads: 8
//! - n_layers: 6
//! - d_ff: 2048
//! - vocab_size: 32000
//! - max_seq_len: 512
//!
//! 4-bit 量化后权重大小: ~3MB

pub mod model;
pub mod quantize;
pub mod tokenizer;
pub mod api;
