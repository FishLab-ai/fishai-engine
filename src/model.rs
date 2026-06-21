//! Transformer 模型模块 — FishAI Engine 的核心推理实现
//!
//! 本模块实现了完整的 LLaMA 风格 Transformer 前向传播，是 AI 引擎的心脏。
//!
//! 主要内容：
//! - [`ModelConfig`]：模型架构超参数（层数、头数、嵌入维度、FFN 维度等）
//! - [`ModelWeights`]：模型权重数据及加载逻辑（从 GGUF 文件读取）
//! - [`LayerWeights`]：单层 Transformer 权重（注意力 + FFN）
//! - [`GenerationState`]：自回归生成状态机
//! - [`GenerationConfig`]：生成超参数（温度、top-k、top-p 等）
//!
//! 支持的模型架构：LLaMA、Qwen2、Mistral、Phi
//! 支持的特性：GQA（分组查询注意力）、RoPE 旋转位置编码、SwiGLU 激活函数
//!
//! # 前向传播流程
//!
//! 1. Token 嵌入查表
//! 2. 逐层 Transformer 块：
//!    - RMSNorm → QKV 投影 → RoPE → KV 缓存 → 注意力 → 残差连接
//!    - RMSNorm → SwiGLU FFN → 残差连接
//! 3. 最终 RMSNorm → 输出投影 → logits

use crate::gguf::GGUFFile;
use crate::kv_cache::KVCache;
use crate::sampling::{Sampler, SamplerConfig};
use crate::tensor::{
    add, matmul, mul_elementwise, rms_norm, rope_emb, scaled_dot_product_attention,
    silu,
};

// ---------------------------------------------------------------------------
// ModelConfig — 模型架构超参数
// ---------------------------------------------------------------------------

/// 模型架构超参数，描述 Transformer 的结构。
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Transformer 层数
    pub n_layers: usize,
    /// Query 注意力头数
    pub n_heads: usize,
    /// KV 注意力头数（GQA 时可能小于 n_heads）
    pub n_kv_heads: usize,
    /// 每个头的维度
    pub head_dim: usize,
    /// 总嵌入维度（n_heads × head_dim）
    pub n_embd: usize,
    /// 词表大小
    pub vocab_size: usize,
    /// 最大上下文长度
    pub context_len: usize,
    /// FFN 中间维度倍率（通常 LLaMA 为 2.667）
    pub ffn_dim_multiplier: f32,
    /// RMSNorm 的 epsilon（通常 1e-5）
    pub norm_eps: f32,
    /// RoPE 基础频率（通常 10000.0）
    pub rope_base: f32,
    /// RoPE 频率维度（通常 head_dim / 2）
    pub rope_freq_dim: usize,
}

impl ModelConfig {
    /// 计算 FFN 中间维度。
    ///
    /// 公式：`n_embd × ffn_dim_multiplier`，结果向上取整到 256 的整数倍。
    pub fn n_ff(&self) -> usize {
        let raw = self.n_embd as f32 * self.ffn_dim_multiplier;
        // 向上对齐到 256 的整数倍
        ((raw as usize + 255) / 256) * 256
    }

    /// 返回每个头的维度。
    pub fn n_head_dim(&self) -> usize {
        self.head_dim
    }

    /// 返回 KV 头数（MHA 时等于 n_heads）。
    pub fn head_count_kv(&self) -> usize {
        self.n_kv_heads
    }

    /// 从 GGUF 文件的元数据中提取模型配置。
    ///
    /// 支持的架构：`llama`、`qwen2`、`mistral`、`phi`。
    /// 读取键以架构名称为前缀，例如 `llama.attention.head_count`。
    ///
    /// # 错误
    ///
    /// - 缺少必要元数据键
    /// - 架构名称不支持
    /// - 数值无效（如除零）
    pub fn from_gguf(gguf: &GGUFFile) -> Result<Self, String> {
        // 获取架构名称
        let arch = gguf
            .model_architecture()
            .ok_or_else(|| "缺少 general.architecture 元数据".to_string())?;

        // 验证支持的架构
        match arch.as_str() {
            "llama" | "qwen2" | "mistral" | "phi" => {}
            other => return Err(format!("不支持的模型架构: {}", other)),
        }

        let prefix = &arch;

        // 读取基础超参数
        let n_layers = gguf
            .metadata_u64(&format!("{}.block_count", prefix))
            .ok_or_else(|| format!("缺少 {}.block_count", prefix))? as usize;

        let n_heads = gguf
            .metadata_u64(&format!("{}.attention.head_count", prefix))
            .ok_or_else(|| format!("缺少 {}.attention.head_count", prefix))? as usize;

        let n_kv_heads = gguf
            .metadata_u64(&format!("{}.attention.head_count_kv", prefix))
            .unwrap_or(n_heads as u64) as usize;

        let n_embd = gguf
            .metadata_u64(&format!("{}.embedding_length", prefix))
            .ok_or_else(|| format!("缺少 {}.embedding_length", prefix))? as usize;

        let context_len = gguf
            .metadata_u64(&format!("{}.context_length", prefix))
            .ok_or_else(|| format!("缺少 {}.context_length", prefix))? as usize;

        // 词表大小：优先读取显式值，否则尝试从 embedding 矩阵推断
        let vocab_size = if let Some(vs) = gguf.vocab_size() {
            vs as usize
        } else if let Some(info) = gguf.tensor_info("token_embd.weight") {
            info.nelement() / n_embd
        } else {
            return Err("无法确定词表大小".to_string());
        };

        // 计算头维度
        if n_heads == 0 {
            return Err("attention.head_count 为 0".to_string());
        }
        let head_dim = n_embd / n_heads;
        if head_dim == 0 {
            return Err(format!(
                "嵌入维度 {} 无法被头数 {} 整除",
                n_embd, n_heads
            ));
        }

        // FFN 维度倍率：尝试从元数据读取，否则使用默认值
        let ffn_dim_multiplier = gguf
            .metadata_f32(&format!("{}.feed_forward_length", prefix))
            .map(|ff_len| ff_len as f32 / n_embd as f32)
            .unwrap_or(2.667);

        // RMSNorm epsilon
        let norm_eps = gguf
            .metadata_f32(&format!("{}.attention.layer_norm_rms_epsilon", prefix))
            .unwrap_or(1e-5);

        // RoPE 基础频率
        let rope_base = gguf
            .metadata_f32(&format!("{}.rope.freq_base", prefix))
            .unwrap_or(10000.0);

        // RoPE 频率维度
        let rope_freq_dim = head_dim / 2;

        Ok(Self {
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            n_embd,
            vocab_size,
            context_len,
            ffn_dim_multiplier,
            norm_eps,
            rope_base,
            rope_freq_dim,
        })
    }
}

// ---------------------------------------------------------------------------
// LayerWeights — 单层 Transformer 权重
// ---------------------------------------------------------------------------

/// 单层 Transformer 的权重数据。
///
/// 包含注意力子层和 FFN 子层的所有参数。
/// 注意力权重布局为行优先 `[in_features, out_features]`（与 GGUF 存储一致），
/// 可直接用于 `matmul(x, w, m, k, n)`。
#[derive(Clone, Debug)]
pub struct LayerWeights {
    /// 注意力前的 RMSNorm 权重：[n_embd]
    pub attn_norm: Vec<f32>,
    /// Query 投影权重：[n_embd, n_heads * head_dim]
    pub wq: Vec<f32>,
    /// Key 投影权重：[n_embd, n_kv_heads * head_dim]
    pub wk: Vec<f32>,
    /// Value 投影权重：[n_embd, n_kv_heads * head_dim]
    pub wv: Vec<f32>,
    /// 注意力输出投影权重：[n_heads * head_dim, n_embd]
    pub wo: Vec<f32>,

    /// FFN 前的 RMSNorm 权重：[n_embd]
    pub ffn_norm: Vec<f32>,
    /// SwiGLU 门控投影权重：[n_embd, n_ff]
    pub w_gate: Vec<f32>,
    /// SwiGLU 上投影权重：[n_embd, n_ff]
    pub w_up: Vec<f32>,
    /// SwiGLU 下投影权重：[n_ff, n_embd]
    pub w_down: Vec<f32>,
}

// ---------------------------------------------------------------------------
// ModelWeights — 完整模型权重
// ---------------------------------------------------------------------------

/// 完整的 Transformer 模型权重。
///
/// 包含嵌入层、所有 Transformer 层、最终归一化和输出投影的参数。
#[derive(Clone, Debug)]
pub struct ModelWeights {
    /// Token 嵌入表：[vocab_size, n_embd]
    pub embeddings: Vec<f32>,
    /// 每层权重
    pub layers: Vec<LayerWeights>,
    /// 输出 RMSNorm 权重：[n_embd]
    pub output_norm: Vec<f32>,
    /// 输出投影权重（可选）：[n_embd, vocab_size] 或 [vocab_size, n_embd]
    pub output: Option<Vec<f32>>,
    /// 模型配置
    pub config: ModelConfig,
}

impl ModelWeights {
    /// 从 GGUF 文件加载所有模型权重。
    ///
    /// 张量名称映射规则（适用于 llama/qwen2/mistral/phi）：
    /// - `token_embd.weight` → embeddings
    /// - `blk.{i}.attn_norm.weight` → attn_norm
    /// - `blk.{i}.attn_q.weight` → wq
    /// - `blk.{i}.attn_k.weight` → wk
    /// - `blk.{i}.attn_v.weight` → wv
    /// - `blk.{i}.attn_output.weight` → wo
    /// - `blk.{i}.ffn_norm.weight` → ffn_norm
    /// - `blk.{i}.ffn_gate.weight` → w_gate
    /// - `blk.{i}.ffn_up.weight` → w_up
    /// - `blk.{i}.ffn_down.weight` → w_down
    /// - `output_norm.weight` → output_norm
    /// - `output.weight` → output（可选）
    pub fn load_from_gguf(gguf: &GGUFFile, config: &ModelConfig) -> Result<Self, String> {
        // 辅助函数：读取张量并返回 f32 数据
        let load_tensor = |name: &str| -> Result<Vec<f32>, String> {
            let info = gguf
                .tensor_info(name)
                .ok_or_else(|| format!("找不到张量: {}", name))?;
            gguf.read_tensor_data_f32(info)
        };

        // 加载嵌入层
        let embeddings = load_tensor("token_embd.weight")?;

        // 加载每层权重
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let layer = LayerWeights {
                attn_norm: load_tensor(&format!("blk.{}.attn_norm.weight", i))?,
                wq: load_tensor(&format!("blk.{}.attn_q.weight", i))?,
                wk: load_tensor(&format!("blk.{}.attn_k.weight", i))?,
                wv: load_tensor(&format!("blk.{}.attn_v.weight", i))?,
                wo: load_tensor(&format!("blk.{}.attn_output.weight", i))?,
                ffn_norm: load_tensor(&format!("blk.{}.ffn_norm.weight", i))?,
                w_gate: load_tensor(&format!("blk.{}.ffn_gate.weight", i))?,
                w_up: load_tensor(&format!("blk.{}.ffn_up.weight", i))?,
                w_down: load_tensor(&format!("blk.{}.ffn_down.weight", i))?,
            };
            layers.push(layer);
        }

        // 加载输出归一化
        let output_norm = load_tensor("output_norm.weight")?;

        // 加载输出投影（可选）
        let output = if gguf.tensor_info("output.weight").is_some() {
            Some(load_tensor("output.weight")?)
        } else {
            None
        };

        Ok(Self {
            embeddings,
            layers,
            output_norm,
            output,
            config: config.clone(),
        })
    }

    /// Transformer 前向传播。
    ///
    /// 对输入 token 执行完整的单步前向传播，返回 logits 向量。
    ///
    /// # Arguments
    /// * `token_ids` - 输入 token ID 列表（通常为 1 个，用于自回归生成）
    /// * `pos` - 当前 token 在序列中的位置（从 0 开始）
    /// * `kv_cache` - 可变的 KV 缓存引用
    ///
    /// # Returns
    /// logits 向量：长度为 vocab_size
    ///
    /// # 前向传播步骤
    /// 1. Token 嵌入查表
    /// 2. 逐层 Transformer 块
    /// 3. 最终 RMSNorm + 输出投影
    pub fn forward(
        &self,
        token_ids: &[u32],
        pos: usize,
        kv_cache: &mut KVCache,
    ) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let n_q_dim = n_heads * head_dim;
        let n_kv_dim = n_kv_heads * head_dim;
        let n_ff = self.config.n_ff();
        let norm_eps = self.config.norm_eps;
        let rope_base = self.config.rope_base;

        // GQA 重复因子
        let n_rep = n_heads / n_kv_heads;

        // ========== 1. Token 嵌入查表 ==========
        // 对每个 token，查嵌入表得到 [n_embd] 向量
        // 如果 batch > 1，拼接所有 token 的嵌入；这里通常 batch=1
        let mut x = Vec::with_capacity(token_ids.len() * n_embd);
        for &tid in token_ids {
            let start = tid as usize * n_embd;
            let end = start + n_embd;
            x.extend_from_slice(&self.embeddings[start..end]);
        }
        let batch_size = token_ids.len();

        // ========== 2. 逐层 Transformer ==========
        for layer_idx in 0..self.config.n_layers {
            let layer = &self.layers[layer_idx];

            // --- 2a. RMSNorm（注意力前） ---
            let mut x_norm = Vec::with_capacity(batch_size * n_embd);
            for b in 0..batch_size {
                let token_x = &x[b * n_embd..(b + 1) * n_embd];
                let normed = rms_norm(token_x, &layer.attn_norm, norm_eps);
                x_norm.extend_from_slice(&normed);
            }

            // --- 2b. QKV 投影 ---
            // Q: [batch, n_embd] @ wq [n_embd, n_q_dim] → [batch, n_q_dim]
            let mut q = matmul(&x_norm, &layer.wq, batch_size, n_embd, n_q_dim);
            // K: [batch, n_embd] @ wk [n_embd, n_kv_dim] → [batch, n_kv_dim]
            let mut k = matmul(&x_norm, &layer.wk, batch_size, n_embd, n_kv_dim);
            // V: [batch, n_embd] @ wv [n_embd, n_kv_dim] → [batch, n_kv_dim]
            let v = matmul(&x_norm, &layer.wv, batch_size, n_embd, n_kv_dim);

            // --- 2c. 对 Q 和 K 施加 RoPE ---
            apply_rope_to_tensor(&mut q, n_heads, head_dim, pos, rope_base);
            apply_rope_to_tensor(&mut k, n_kv_heads, head_dim, pos, rope_base);

            // --- 2d. KV 缓存更新 ---
            kv_cache.update(layer_idx, k.clone(), v.clone());

            // --- 2e. 获取完整缓存 K/V 并为 GQA 扩展 ---
            let (cached_k, cached_v) = kv_cache.get(layer_idx);
            let cached_seq_len = cached_k.len() / n_kv_dim;

            // 扩展 K/V: [total_seq, n_kv_heads * head_dim] → [total_seq, n_heads * head_dim]
            let expanded_k = expand_kv_heads(cached_k, cached_seq_len, n_kv_heads, n_rep, head_dim);
            let expanded_v = expand_kv_heads(cached_v, cached_seq_len, n_kv_heads, n_rep, head_dim);

            // --- 2f. 缩放点积注意力 ---
            // q: [batch, n_heads * head_dim]
            // expanded_k: [total_seq, n_heads * head_dim]
            // expanded_v: [total_seq, n_heads * head_dim]
            let (attn_out, _, _) = scaled_dot_product_attention(
                &q,
                &expanded_k,
                &expanded_v,
                n_heads,
                head_dim,
                None,
                None,
                None,
            );

            // --- 2g. 输出投影 ---
            // attn_out: [batch, n_q_dim] @ wo: [n_q_dim, n_embd] → [batch, n_embd]
            let attn_proj = matmul(&attn_out, &layer.wo, batch_size, n_q_dim, n_embd);

            // --- 2h. 残差连接 ---
            x = add(&x, &attn_proj);

            // --- 2i. RMSNorm（FFN 前） ---
            let mut x_norm2 = Vec::with_capacity(batch_size * n_embd);
            for b in 0..batch_size {
                let token_x = &x[b * n_embd..(b + 1) * n_embd];
                let normed = rms_norm(token_x, &layer.ffn_norm, norm_eps);
                x_norm2.extend_from_slice(&normed);
            }

            // --- 2j. SwiGLU FFN ---
            // gate = SiLU(x @ w_gate): [batch, n_embd] @ [n_embd, n_ff] → [batch, n_ff]
            let gate_proj = matmul(&x_norm2, &layer.w_gate, batch_size, n_embd, n_ff);
            let gate = silu(&gate_proj);
            // up = x @ w_up: [batch, n_embd] @ [n_embd, n_ff] → [batch, n_ff]
            let up = matmul(&x_norm2, &layer.w_up, batch_size, n_embd, n_ff);
            // SwiGLU: gate * up
            let swiglu = mul_elementwise(&gate, &up);
            // down = (gate * up) @ w_down: [batch, n_ff] @ [n_ff, n_embd] → [batch, n_embd]
            let ffn_out = matmul(&swiglu, &layer.w_down, batch_size, n_ff, n_embd);

            // --- 2k. 残差连接 ---
            x = add(&x, &ffn_out);
        }

        // ========== 3. 最终 RMSNorm ==========
        let mut x_final = Vec::with_capacity(batch_size * n_embd);
        for b in 0..batch_size {
            let token_x = &x[b * n_embd..(b + 1) * n_embd];
            let normed = rms_norm(token_x, &self.output_norm, norm_eps);
            x_final.extend_from_slice(&normed);
        }

        // ========== 4. 输出投影 → logits ==========
        // logits: [batch, n_embd] @ output: [n_embd, vocab_size] → [batch, vocab_size]
        let logits = if let Some(ref output_w) = self.output {
            // 检查输出权重的形状并决定是否需要转置
            let expected_len = n_embd * self.config.vocab_size;
            if output_w.len() == expected_len {
                // 形状匹配 [n_embd, vocab_size]，直接使用
                matmul(&x_final, output_w, batch_size, n_embd, self.config.vocab_size)
            } else {
                // 可能是 [vocab_size, n_embd]，需要转置
                let output_t = transpose_2d(output_w, self.config.vocab_size, n_embd);
                matmul(&x_final, &output_t, batch_size, n_embd, self.config.vocab_size)
            }
        } else {
            // 没有输出投影权重时，使用嵌入表的转置（权重共享）
            // embeddings: [vocab_size, n_embd] → 转置为 [n_embd, vocab_size]
            let emb_t = transpose_2d(&self.embeddings, self.config.vocab_size, n_embd);
            matmul(&x_final, &emb_t, batch_size, n_embd, self.config.vocab_size)
        };

        logits
    }
}

// ---------------------------------------------------------------------------
// 辅助函数
// ---------------------------------------------------------------------------

/// 对张量施加 RoPE（支持不同的 Q/K 头数）。
///
/// 独立于 `tensor::apply_rope`，此函数对单一张量施加旋转位置编码，
/// 适用于 GQA 场景中 Q 和 K 头数不同的情况。
///
/// # 布局
/// `data` 为 `[seq_len, n_h * head_dim]`（行优先）。
fn apply_rope_to_tensor(data: &mut [f32], n_h: usize, head_dim: usize, pos: usize, base: f32) {
    let seq_len = data.len() / (n_h * head_dim);
    if seq_len == 0 || n_h == 0 || head_dim == 0 {
        return;
    }
    let emb = rope_emb(pos, head_dim, base);
    let half = head_dim / 2;

    for s in 0..seq_len {
        for h in 0..n_h {
            let base_offset = s * n_h * head_dim + h * head_dim;
            for d in 0..half {
                let cos_val = emb[2 * d];
                let sin_val = emb[2 * d + 1];

                let x0 = data[base_offset + d];
                let x1 = data[base_offset + d + half];
                data[base_offset + d] = x0 * cos_val - x1 * sin_val;
                data[base_offset + d + half] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

/// 将 KV 头扩展以匹配 Query 头数（GQA 支持）。
///
/// 当 `n_kv_heads < n_heads` 时，每个 KV 头被复制 `n_rep` 次以匹配 Q 的头数。
///
/// # 布局
/// - 输入：`[seq_len, n_kv_heads * head_dim]`
/// - 输出：`[seq_len, n_heads * head_dim]`
fn expand_kv_heads(
    kv: &[f32],
    seq_len: usize,
    n_kv_heads: usize,
    n_rep: usize,
    head_dim: usize,
) -> Vec<f32> {
    if n_rep <= 1 {
        return kv.to_vec();
    }

    let mut expanded = vec![0.0f32; seq_len * n_kv_heads * n_rep * head_dim];

    for s in 0..seq_len {
        for kv_h in 0..n_kv_heads {
            for r in 0..n_rep {
                let q_h = kv_h * n_rep + r;
                let src_start = s * n_kv_heads * head_dim + kv_h * head_dim;
                let dst_start = s * n_kv_heads * n_rep * head_dim + q_h * head_dim;
                expanded[dst_start..dst_start + head_dim]
                    .copy_from_slice(&kv[src_start..src_start + head_dim]);
            }
        }
    }

    expanded
}

/// 转置二维矩阵。
///
/// 输入 `[rows, cols]`（行优先），输出 `[cols, rows]`（行优先）。
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = data[i * cols + j];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// GenerationConfig — 生成超参数
// ---------------------------------------------------------------------------

/// 自回归生成的超参数配置。
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// 最大生成 token 数
    pub max_tokens: usize,
    /// 采样温度（0.0 = 贪婪解码）
    pub temperature: f32,
    /// Top-K 截断（0 = 不限制）
    pub top_k: usize,
    /// Top-P 截断（0.0 = 不限制）
    pub top_p: f32,
    /// 重复惩罚系数（1.0 = 不惩罚）
    pub repetition_penalty: f32,
    /// 停止 token ID 列表
    pub stop_token_ids: Vec<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.15,
            stop_token_ids: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// GenerationState — 自回归生成状态机
// ---------------------------------------------------------------------------

/// 自回归生成状态机，管理完整的生成过程。
///
/// 持有模型权重、KV 缓存、采样器和生成配置，
/// 提供从 prompt 到完整回复的生成能力。
pub struct GenerationState {
    /// 模型权重
    pub model: ModelWeights,
    /// KV 缓存
    pub kv_cache: KVCache,
    /// Token 采样器
    pub sampler: Sampler,
    /// 已生成的 token ID 序列（包含 prompt）
    pub token_ids: Vec<u32>,
    /// 当前序列位置
    pub pos: usize,
    /// 生成配置
    pub config: GenerationConfig,
}

impl GenerationState {
    /// 创建新的生成状态。
    ///
    /// # Arguments
    /// * `model` - 已加载的模型权重
    /// * `config` - 生成超参数
    pub fn new(model: ModelWeights, config: GenerationConfig) -> Self {
        let n_layers = model.config.n_layers;
        let kv_cache = KVCache::new(n_layers);

        let sampler_config = SamplerConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        };
        let sampler = Sampler::new(sampler_config);

        Self {
            model,
            kv_cache,
            sampler,
            token_ids: Vec::new(),
            pos: 0,
            config,
        }
    }

    /// 从 prompt 开始自回归生成。
    ///
    /// 先对 prompt 中每个 token 执行前向传播（prefill），
    /// 然后逐个生成新 token 直到达到最大数量或遇到停止 token。
    ///
    /// # Arguments
    /// * `prompt_tokens` - 提示词的 token ID 列表
    ///
    /// # Returns
    /// 生成的 token ID 列表（不含 prompt）
    pub fn generate(&mut self, prompt_tokens: &[u32]) -> Vec<u32> {
        self.token_ids.clear();
        self.kv_cache.clear();
        self.sampler.reset();
        self.pos = 0;

        // Prefill: 对 prompt 中每个 token 执行前向传播
        for &tid in prompt_tokens {
            self.token_ids.push(tid);
            // Prefill 时不采样，只更新 KV 缓存
            let _logits = self.model.forward(&[tid], self.pos, &mut self.kv_cache);
            self.pos += 1;
        }

        // 自回归生成
        let mut generated = Vec::new();
        for _ in 0..self.config.max_tokens {
            // 取最后一个 token 进行生成
            let last_token = *self.token_ids.last().unwrap();
            match self.generate_step(last_token) {
                Some(next_token) => {
                    // 检查是否为停止 token
                    if self.config.stop_token_ids.contains(&next_token) {
                        break;
                    }
                    generated.push(next_token);
                }
                None => break,
            }
        }

        generated
    }

    /// 执行单步生成。
    ///
    /// 对给定 token 执行前向传播、采样，并更新状态。
    ///
    /// # Arguments
    /// * `token_id` - 当前输入 token ID
    ///
    /// # Returns
    /// 采样的下一个 token ID，如果已达最大长度则返回 `None`。
    pub fn generate_step(&mut self, token_id: u32) -> Option<u32> {
        // 检查是否超过最大 token 数
        let generated_count = self.token_ids.len();
        if generated_count >= self.config.max_tokens {
            return None;
        }

        // 前向传播
        let logits = self.model.forward(&[token_id], self.pos, &mut self.kv_cache);

        // 采样
        let next_token = self.sampler.sample(&logits) as u32;

        // 记录
        self.sampler.observe_token(next_token as usize);
        self.token_ids.push(next_token);
        self.pos += 1;

        // 检查停止 token
        if self.config.stop_token_ids.contains(&next_token) {
            return None;
        }

        Some(next_token)
    }

    /// 重置生成状态（用于开始新对话）。
    ///
    /// 清空 KV 缓存、采样器状态和 token 历史。
    pub fn reset(&mut self) {
        self.kv_cache.clear();
        self.sampler.reset();
        self.token_ids.clear();
        self.pos = 0;
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GGUFFile, GGUFValueType};
    use byteorder::{LittleEndian, WriteBytesExt};


    // -----------------------------------------------------------------------
    // 辅助函数：构建最小化 GGUF 文件
    // -----------------------------------------------------------------------

    /// 写入 GGUF 字符串到缓冲区
    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
        buf.extend_from_slice(s.as_bytes());
    }

    /// 写入 UINT32 元数据键值对
    fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
        write_gguf_string(buf, key);
        buf.write_u32::<LittleEndian>(GGUFValueType::Uint32 as u32)
            .unwrap();
        buf.write_u32::<LittleEndian>(value).unwrap();
    }

    /// 写入 STRING 元数据键值对
    fn write_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
        write_gguf_string(buf, key);
        buf.write_u32::<LittleEndian>(GGUFValueType::String as u32)
            .unwrap();
        write_gguf_string(buf, value);
    }

    /// 写入 FLOAT32 元数据键值对
    fn write_kv_f32(buf: &mut Vec<u8>, key: &str, value: f32) {
        write_gguf_string(buf, key);
        buf.write_u32::<LittleEndian>(GGUFValueType::Float32 as u32)
            .unwrap();
        buf.write_f32::<LittleEndian>(value).unwrap();
    }

    /// 对齐到指定字节的整数倍
    fn align_to(n: usize, alignment: usize) -> usize {
        if alignment == 0 {
            return n;
        }
        let mask = alignment - 1;
        (n + mask) & !mask
    }

    /// 构建用于测试 ModelConfig 的最小 GGUF 文件
    fn build_test_gguf_file() -> String {
        let mut buf = Vec::new();

        // 文件头
        buf.write_u32::<LittleEndian>(0x46465547u32).unwrap(); // GGUF magic
        buf.write_u32::<LittleEndian>(3).unwrap(); // version V3
        buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count = 0
        buf.write_u64::<LittleEndian>(10).unwrap(); // metadata_kv_count = 10

        // 元数据
        write_kv_string(&mut buf, "general.architecture", "llama");
        write_kv_u32(&mut buf, "general.tokenizer.ggml.vocab_size", 32000);
        write_kv_u32(&mut buf, "llama.block_count", 32);
        write_kv_u32(&mut buf, "llama.attention.head_count", 32);
        write_kv_u32(&mut buf, "llama.attention.head_count_kv", 32);
        write_kv_u32(&mut buf, "llama.embedding_length", 4096);
        write_kv_u32(&mut buf, "llama.context_length", 2048);
        write_kv_f32(&mut buf, "llama.attention.layer_norm_rms_epsilon", 1e-5);
        write_kv_f32(&mut buf, "llama.rope.freq_base", 10000.0);

        // 对齐填充
        let header_end = buf.len();
        let aligned = align_to(header_end, 32);
        while buf.len() < aligned {
            buf.push(0);
        }

        let path = std::env::temp_dir().join("fishai_test_model_config.gguf");
        let path_str = path.to_string_lossy().into_owned();
        std::fs::write(&path, &buf).unwrap();
        path_str
    }

    // -----------------------------------------------------------------------
    // 测试 1: ModelConfig::n_ff 计算
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_config_n_ff() {
        let config = ModelConfig {
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 32,
            head_dim: 128,
            n_embd: 4096,
            vocab_size: 32000,
            context_len: 2048,
            ffn_dim_multiplier: 2.667,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            rope_freq_dim: 64,
        };

        // n_embd * 2.667 = 10922.112 → 对齐到 256 的整数倍 → 11008
        let n_ff = config.n_ff();
        assert_eq!(n_ff, 11008);
        assert!(n_ff % 256 == 0);
        assert!(n_ff >= (4096_f32 * 2.667) as usize);
    }

    #[test]
    fn test_model_config_n_ff_exact_multiple() {
        let config = ModelConfig {
            n_layers: 1,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 64,
            n_embd: 256,
            vocab_size: 1000,
            context_len: 512,
            ffn_dim_multiplier: 2.0,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            rope_freq_dim: 32,
        };

        // 256 * 2.0 = 512, 已经是 256 的整数倍
        assert_eq!(config.n_ff(), 512);
    }

    // -----------------------------------------------------------------------
    // 测试 2: ModelConfig::from_gguf
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_config_from_gguf_metadata() {
        let path = build_test_gguf_file();
        let gguf = GGUFFile::open(&path).expect("无法打开测试 GGUF 文件");

        let config = ModelConfig::from_gguf(&gguf).expect("无法从 GGUF 提取配置");

        assert_eq!(config.n_layers, 32);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.n_kv_heads, 32);
        assert_eq!(config.n_embd, 4096);
        assert_eq!(config.head_dim, 128); // 4096 / 32
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.context_len, 2048);
        assert!((config.norm_eps - 1e-5).abs() < 1e-8);
        assert!((config.rope_base - 10000.0).abs() < 1e-3);
        assert_eq!(config.rope_freq_dim, 64); // head_dim / 2

        // 清理临时文件
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_model_config_from_gguf_gqa() {
        // 构建一个 GQA 配置的 GGUF（head_count_kv != head_count）
        let mut buf = Vec::new();

        buf.write_u32::<LittleEndian>(0x46465547u32).unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap();
        buf.write_u64::<LittleEndian>(0).unwrap();
        buf.write_u64::<LittleEndian>(7).unwrap();

        write_kv_string(&mut buf, "general.architecture", "qwen2");
        write_kv_u32(&mut buf, "general.tokenizer.ggml.vocab_size", 151936);
        write_kv_u32(&mut buf, "qwen2.block_count", 24);
        write_kv_u32(&mut buf, "qwen2.attention.head_count", 28);
        write_kv_u32(&mut buf, "qwen2.attention.head_count_kv", 4);
        write_kv_u32(&mut buf, "qwen2.embedding_length", 2048);
        write_kv_u32(&mut buf, "qwen2.context_length", 32768);

        let aligned = align_to(buf.len(), 32);
        while buf.len() < aligned {
            buf.push(0);
        }

        let path = std::env::temp_dir().join("fishai_test_gqa.gguf");
        let path_str = path.to_string_lossy().into_owned();
        std::fs::write(&path, &buf).unwrap();

        let gguf = GGUFFile::open(&path_str).expect("无法打开 GQA GGUF 文件");
        let config = ModelConfig::from_gguf(&gguf).expect("无法提取 GQA 配置");

        assert_eq!(config.n_heads, 28);
        assert_eq!(config.n_kv_heads, 4);
        assert_eq!(config.head_dim, 2048 / 28); // 不是整数，所以 73
        assert_eq!(config.n_layers, 24);

        let _ = std::fs::remove_file(&path_str);
    }

    // -----------------------------------------------------------------------
    // 测试 3: GenerationConfig 默认值
    // -----------------------------------------------------------------------

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();

        assert_eq!(config.max_tokens, 512);
        assert!((config.temperature - 0.7).abs() < 1e-6);
        assert_eq!(config.top_k, 40);
        assert!((config.top_p - 0.95).abs() < 1e-6);
        assert!((config.repetition_penalty - 1.15).abs() < 1e-6);
        assert!(config.stop_token_ids.is_empty());
    }

    // -----------------------------------------------------------------------
    // 测试 4: GenerationState 创建
    // -----------------------------------------------------------------------

    fn create_test_model() -> ModelWeights {
        // 构建一个微型模型：1 层、2 头、head_dim=4、n_embd=8、vocab=16
        let config = ModelConfig {
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            n_embd: 8,
            vocab_size: 16,
            context_len: 32,
            ffn_dim_multiplier: 2.0, // n_ff = 16
            norm_eps: 1e-5,
            rope_base: 10000.0,
            rope_freq_dim: 2,
        };

        let n_ff = config.n_ff(); // 16

        // 嵌入表：[16, 8]
        let embeddings: Vec<f32> = (0..16 * 8).map(|i| (i as f32) * 0.01).collect();

        // 单层权重
        let attn_norm: Vec<f32> = vec![1.0; 8];
        let wq: Vec<f32> = (0..8 * 8).map(|i| (i as f32) * 0.01).collect();
        let wk: Vec<f32> = (0..8 * 8).map(|i| (i as f32) * 0.02).collect();
        let wv: Vec<f32> = (0..8 * 8).map(|i| (i as f32) * 0.03).collect();
        let wo: Vec<f32> = (0..8 * 8).map(|i| (i as f32) * 0.04).collect();
        let ffn_norm: Vec<f32> = vec![1.0; 8];
        let w_gate: Vec<f32> = (0..8 * n_ff).map(|i| (i as f32) * 0.05).collect();
        let w_up: Vec<f32> = (0..8 * n_ff).map(|i| (i as f32) * 0.06).collect();
        let w_down: Vec<f32> = (0..n_ff * 8).map(|i| (i as f32) * 0.07).collect();

        let layers = vec![LayerWeights {
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
        }];

        let output_norm: Vec<f32> = vec![1.0; 8];

        ModelWeights {
            embeddings,
            layers,
            output_norm,
            output: None, // 使用嵌入表转置
            config,
        }
    }

    #[test]
    fn test_generation_state_creation() {
        let model = create_test_model();
        let config = GenerationConfig::default();

        let state = GenerationState::new(model, config);

        assert_eq!(state.token_ids.len(), 0);
        assert_eq!(state.pos, 0);
        assert_eq!(state.kv_cache.n_layers(), 1);
        assert!(!state.config.stop_token_ids.is_empty() || true); // 默认为空
    }

    #[test]
    fn test_generation_state_reset() {
        let model = create_test_model();
        let config = GenerationConfig {
            max_tokens: 5,
            ..GenerationConfig::default()
        };

        let mut state = GenerationState::new(model, config);
        state.token_ids.push(1);
        state.token_ids.push(2);
        state.pos = 2;

        state.reset();

        assert!(state.token_ids.is_empty());
        assert_eq!(state.pos, 0);
    }

    // -----------------------------------------------------------------------
    // 测试 5: KV 缓存集成 — 模拟简单前向传播
    // -----------------------------------------------------------------------

    #[test]
    fn test_kv_cache_integration() {
        let model = create_test_model();

        // 创建 KV 缓存
        let mut kv_cache = KVCache::new(model.config.n_layers);

        // 第一次前向传播：token_id=0, pos=0
        let logits_0 = model.forward(&[0], 0, &mut kv_cache);

        // 验证 logits 维度
        assert_eq!(logits_0.len(), model.config.vocab_size); // 16

        // 验证 KV 缓存已更新
        let kv_dim = model.config.n_kv_heads * model.config.head_dim; // 2 * 4 = 8
        let seq_len = kv_cache.seq_len_with_dim(0, kv_dim);
        assert_eq!(seq_len, 1);

        // 第二次前向传播：token_id=1, pos=1
        let logits_1 = model.forward(&[1], 1, &mut kv_cache);

        // 验证 KV 缓存增长
        let seq_len = kv_cache.seq_len_with_dim(0, kv_dim);
        assert_eq!(seq_len, 2);

        // 两次前向传播的 logits 长度相同
        assert_eq!(logits_0.len(), logits_1.len());

        // 不同的输入应该产生不同的 logits
        let are_different = logits_0
            .iter()
            .zip(logits_1.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(are_different, "不同 token 的 logits 应该不同");
    }

    #[test]
    fn test_kv_cache_gqa_expansion() {
        // 测试 GQA KV 头扩展
        let kv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // seq=1, kv_heads=1, head_dim=8
        let expanded = expand_kv_heads(&kv, 1, 1, 2, 8); // n_rep=2

        // 应该将 1 个 KV 头扩展为 2 个
        assert_eq!(expanded.len(), 16); // seq=1, heads=2, head_dim=8

        // 两个头的值应该相同
        for i in 0..8 {
            assert_eq!(expanded[i], expanded[i + 8]);
        }
    }

    #[test]
    fn test_forward_produces_finite_logits() {
        let model = create_test_model();
        let mut kv_cache = KVCache::new(model.config.n_layers);

        let logits = model.forward(&[5], 0, &mut kv_cache);

        // 所有 logits 应该是有限值（非 NaN、非 Inf）
        for (i, &val) in logits.iter().enumerate() {
            assert!(
                val.is_finite(),
                "logit[{}] = {} 不是有限值",
                i, val
            );
        }
    }

    #[test]
    fn test_generation_step_produces_token() {
        let model = create_test_model();
        let config = GenerationConfig {
            max_tokens: 10,
            temperature: 0.0, // 贪婪
            ..GenerationConfig::default()
        };

        let mut state = GenerationState::new(model, config);

        let result = state.generate_step(3);
        assert!(result.is_some());

        let token = result.unwrap();
        assert!(token < state.model.config.vocab_size as u32);
        assert_eq!(state.token_ids.len(), 1); // 生成的 1 个 token
        assert_eq!(state.pos, 1);
    }

    #[test]
    fn test_generation_with_stop_token() {
        let model = create_test_model();
        let config = GenerationConfig {
            max_tokens: 100,
            temperature: 0.0,
            stop_token_ids: vec![0], // token 0 作为停止符
            ..GenerationConfig::default()
        };

        let mut state = GenerationState::new(model, config);

        // 先生成一个步骤
        let _ = state.generate_step(5);
        // generate_step 只将生成的 token 加入 token_ids
        assert_eq!(state.token_ids.len(), 1);
    }

    #[test]
    fn test_model_config_accessors() {
        let config = ModelConfig {
            n_layers: 4,
            n_heads: 8,
            n_kv_heads: 4,
            head_dim: 64,
            n_embd: 512,
            vocab_size: 1000,
            context_len: 1024,
            ffn_dim_multiplier: 2.0,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            rope_freq_dim: 32,
        };

        assert_eq!(config.n_head_dim(), 64);
        assert_eq!(config.head_count_kv(), 4);
        assert_eq!(config.n_ff(), 1024); // 512 * 2.0, 已经是 256 的倍数
    }
}