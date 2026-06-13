//! GPT 模型架构 - 完全自研
//!
//! 从零实现 Transformer 的每一个组件：
//! 1. Token Embedding + Positional Embedding
//! 2. Multi-Head Self-Attention (带因果掩码)
//! 3. Feed-Forward Network (两层 MLP + GELU)
//! 4. Layer Normalization
//! 5. Residual Connection
//! 6. 最终的 LM Head

use rand::Rng;
use std::f64::consts::SQRT_2;

/// 模型超参数配置
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub dropout: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            max_seq_len: 512,
            d_model: 512,
            n_heads: 8,
            n_layers: 6,
            d_ff: 2048,
            dropout: 0.1,
        }
    }
}

impl ModelConfig {
    /// 计算模型总参数量
    pub fn total_params(&self) -> usize {
        // Token Embedding
        let tok_emb = self.vocab_size * self.d_model;
        // Position Embedding
        let pos_emb = self.max_seq_len * self.d_model;

        let mut layer_params = 0usize;
        // Multi-Head Attention: Q, K, V projections + output projection
        layer_params += 4 * self.d_model * self.d_model;
        // Attention biases
        layer_params += 4 * self.d_model;
        // Feed-Forward Network
        layer_params += self.d_model * self.d_ff + self.d_ff; // W1 + b1
        layer_params += self.d_ff * self.d_model + self.d_model; // W2 + b2
        // Layer Norm (2 per layer)
        layer_params += 4 * self.d_model;

        let transformer_params = layer_params * self.n_layers;
        // LM Head
        let lm_head = self.d_model * self.vocab_size;
        // Final LayerNorm
        let final_ln = 2 * self.d_model;

        tok_emb + pos_emb + transformer_params + lm_head + final_ln
    }

    /// 量化后的模型大小 (4-bit)
    pub fn quantized_size_mb(&self) -> f64 {
        let total_params = self.total_params();
        // 4-bit = 0.5 bytes per parameter
        let bytes = total_params as f64 * 0.5;
        bytes / (1024.0 * 1024.0)
    }
}

/// 4-bit 量化权重存储格式
/// 每个 u8 存储 2 个 4-bit 权重
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantizedWeight {
    /// 量化后的数据 (每个 byte 存 2 个 4-bit 值)
    pub data: Vec<u8>,
    /// 缩放因子 (per-channel)
    pub scale: Vec<f32>,
    /// 零点 (per-channel)
    pub zero_point: Vec<i8>,
    /// 原始形状
    pub shape: Vec<usize>,
}

impl QuantizedWeight {
    /// 创建空的量化权重
    pub fn new(shape: Vec<usize>, channels: usize) -> Self {
        let total_elements: usize = shape.iter().product();
        let data_len = (total_elements + 1) / 2;
        Self {
            data: vec![0u8; data_len],
            scale: vec![1.0f32; channels],
            zero_point: vec![0i8; channels],
            shape,
        }
    }

    /// 解量化为 f32 向量
    pub fn dequantize(&self, channel: usize) -> Vec<f32> {
        let scale = self.scale.get(channel).copied().unwrap_or(1.0);
        let zp = self.zero_point.get(channel).copied().unwrap_or(0) as f32;
        self.data
            .iter()
            .flat_map(|&byte| {
                let low = (byte & 0x0F) as f32;
                let high = ((byte >> 4) & 0x0F) as f32;
                [(low - zp) * scale, (high - zp) * scale]
            })
            .collect()
    }
}

/// 单个 Transformer 层的权重
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransformerLayerWeights {
    pub wq: QuantizedWeight,
    pub wk: QuantizedWeight,
    pub wv: QuantizedWeight,
    pub wo: QuantizedWeight,
    pub w1: QuantizedWeight,
    pub w2: QuantizedWeight,
    pub ln1_gamma: Vec<f32>,
    pub ln1_beta: Vec<f32>,
    pub ln2_gamma: Vec<f32>,
    pub ln2_beta: Vec<f32>,
}

/// 完整的 GPT 模型权重
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GPTWeights {
    pub config: ModelConfig,
    pub token_embedding: QuantizedWeight,
    pub position_embedding: QuantizedWeight,
    pub layers: Vec<TransformerLayerWeights>,
    pub final_ln_gamma: Vec<f32>,
    pub final_ln_beta: Vec<f32>,
    pub lm_head: QuantizedWeight,
}

impl GPTWeights {
    /// 随机初始化模型权重 (用于训练起点或 demo 模式)
    pub fn random_init(config: &ModelConfig) -> Self {
        let d = config.d_model;
        let ff = config.d_ff;
        let v = config.vocab_size;
        let s = config.max_seq_len;

        let mut rng = rand::thread_rng();

        let make_qw = |shape: Vec<usize>, channels: usize, rng: &mut rand::rngs::ThreadRng| -> QuantizedWeight {
            let total: usize = shape.iter().product();
            let data_len = (total + 1) / 2;
            let mut data = vec![0u8; data_len];
            for byte in data.iter_mut() {
                let low: u8 = rng.gen_range(0..16u8);
                let high: u8 = rng.gen_range(0..16u8);
                *byte = (high << 4) | low;
            }
            let scale = vec![0.02f32; channels];
            let zero_point = vec![8i8; channels];
            QuantizedWeight { data, scale, zero_point, shape }
        };

        let layers: Vec<TransformerLayerWeights> = (0..config.n_layers)
            .map(|_| {
                let wq = make_qw(vec![d, d], d, &mut rng);
                let wk = make_qw(vec![d, d], d, &mut rng);
                let wv = make_qw(vec![d, d], d, &mut rng);
                let wo = make_qw(vec![d, d], d, &mut rng);
                let w1 = make_qw(vec![d, ff], d, &mut rng);
                let w2 = make_qw(vec![ff, d], ff, &mut rng);
                TransformerLayerWeights {
                    wq, wk, wv, wo, w1, w2,
                    ln1_gamma: vec![1.0f32; d],
                    ln1_beta: vec![0.0f32; d],
                    ln2_gamma: vec![1.0f32; d],
                    ln2_beta: vec![0.0f32; d],
                }
            })
            .collect();

        Self {
            config: config.clone(),
            token_embedding: make_qw(vec![v, d], v, &mut rng),
            position_embedding: make_qw(vec![s, d], s, &mut rng),
            layers,
            final_ln_gamma: vec![1.0f32; d],
            final_ln_beta: vec![0.0f32; d],
            lm_head: make_qw(vec![d, v], d, &mut rng),
        }
    }

    /// 保存量化权重到文件
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// 从文件加载量化权重
    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let weights: GPTWeights = serde_json::from_str(&data)?;
        Ok(weights)
    }
}

// ============ 推理核心 ============

/// Layer Normalization
fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std = (var + eps).sqrt();

    for i in 0..n {
        x[i] = gamma[i] * (x[i] - mean) / std + beta[i];
    }
}

/// GELU 激活函数
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (x / SQRT_2 as f32).tanh())
}

/// Softmax (in-place)
fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter().map(|v| (*v - max).exp()).sum();
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v = (*v - max).exp() / sum;
        }
    }
}

/// 矩阵向量乘法 (简化解量化后)
fn mat_vec(weight: &QuantizedWeight, input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let w_data: Vec<f32> = weight.dequantize(0);
    let mut output = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        for j in 0..in_dim {
            if i * in_dim + j < w_data.len() && j < input.len() {
                output[i] += input[j] * w_data[i * in_dim + j];
            }
        }
    }
    output
}

/// 多头自注意力 (带因果掩码)
fn multi_head_attention(
    x: &[Vec<f32>],
    wq: &QuantizedWeight,
    wk: &QuantizedWeight,
    wv: &QuantizedWeight,
    wo: &QuantizedWeight,
    n_heads: usize,
    d_model: usize,
) -> Vec<Vec<f32>> {
    let seq_len = x.len();
    let head_dim = d_model / n_heads;

    // 计算 Q, K, V
    let q: Vec<Vec<f32>> = x.iter().map(|xi| mat_vec(wq, xi, d_model, d_model)).collect();
    let k: Vec<Vec<f32>> = x.iter().map(|xi| mat_vec(wk, xi, d_model, d_model)).collect();
    let v: Vec<Vec<f32>> = x.iter().map(|xi| mat_vec(wv, xi, d_model, d_model)).collect();

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_output = vec![vec![0.0f32; d_model]; seq_len];

    for h in 0..n_heads {
        let start = h * head_dim;

        for i in 0..seq_len {
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            for j in 0..=i.min(seq_len - 1) {
                let dot: f32 = (0..head_dim)
                    .map(|d| q[i][start + d] * k[j][start + d])
                    .sum();
                scores[j] = dot * scale;
            }
            softmax(&mut scores);

            for j in 0..=i.min(seq_len - 1) {
                for d in 0..head_dim {
                    attn_output[i][start + d] += scores[j] * v[j][start + d];
                }
            }
        }
    }

    // 输出投影
    attn_output
        .iter()
        .map(|ai| mat_vec(wo, ai, d_model, d_model))
        .collect()
}

/// 前馈神经网络
fn feed_forward(
    x: &[f32],
    w1: &QuantizedWeight,
    w2: &QuantizedWeight,
    d_ff: usize,
    d_model: usize,
) -> Vec<f32> {
    // 第一层: x @ W1 + GELU
    let mut hidden = vec![0.0f32; d_ff];
    let w1_data: Vec<f32> = w1.dequantize(0);
    for i in 0..d_ff {
        for j in 0..d_model {
            if i * d_model + j < w1_data.len() && j < x.len() {
                hidden[i] += x[j] * w1_data[j * d_ff + i];
            }
        }
        hidden[i] = gelu(hidden[i]);
    }

    // 第二层: hidden @ W2
    let mut output = vec![0.0f32; d_model];
    let w2_data: Vec<f32> = w2.dequantize(0);
    for i in 0..d_model {
        for j in 0..d_ff {
            if j * d_model + i < w2_data.len() && j < hidden.len() {
                output[i] += hidden[j] * w2_data[j * d_model + i];
            }
        }
    }
    output
}

/// 单个 Transformer 层前向传播
fn transformer_layer_forward(
    x: &mut Vec<Vec<f32>>,
    weights: &TransformerLayerWeights,
    n_heads: usize,
    d_ff: usize,
) {
    let d_model = x[0].len();

    // Multi-Head Self-Attention + Residual + LayerNorm
    let attn_out = multi_head_attention(
        x, &weights.wq, &weights.wk, &weights.wv, &weights.wo, n_heads, d_model,
    );
    for i in 0..x.len() {
        for j in 0..d_model {
            x[i][j] += attn_out[i][j];
        }
        layer_norm(&mut x[i], &weights.ln1_gamma, &weights.ln1_beta, 1e-5);
    }

    // Feed-Forward + Residual + LayerNorm
    for i in 0..x.len() {
        let ff_out = feed_forward(&x[i], &weights.w1, &weights.w2, d_ff, d_model);
        for j in 0..d_model {
            x[i][j] += ff_out[j];
        }
        layer_norm(&mut x[i], &weights.ln2_gamma, &weights.ln2_beta, 1e-5);
    }
}

/// GPT 模型前向传播
pub fn gpt_forward(
    token_ids: &[usize],
    weights: &GPTWeights,
) -> Vec<Vec<f32>> {
    let config = &weights.config;
    let d_model = config.d_model;
    let seq_len = token_ids.len();

    // Token Embedding + Position Embedding
    let tok_emb: Vec<f32> = weights.token_embedding.dequantize(0);
    let pos_emb: Vec<f32> = weights.position_embedding.dequantize(0);

    let mut x: Vec<Vec<f32>> = (0..seq_len)
        .map(|pos| {
            let token_id = token_ids[pos].min(config.vocab_size - 1);
            (0..d_model)
                .map(|d| {
                    let t_idx = token_id * d_model + d;
                    let p_idx = pos * d_model + d;
                    tok_emb.get(t_idx).copied().unwrap_or(0.0)
                        + pos_emb.get(p_idx).copied().unwrap_or(0.0)
                })
                .collect()
        })
        .collect();

    // 逐层 Transformer
    for layer_weights in &weights.layers {
        transformer_layer_forward(&mut x, layer_weights, config.n_heads, config.d_ff);
    }

    // 最终 LayerNorm
    for i in 0..seq_len {
        layer_norm(&mut x[i], &weights.final_ln_gamma, &weights.final_ln_beta, 1e-5);
    }

    // LM Head: 计算 logits
    let lm_data: Vec<f32> = weights.lm_head.dequantize(0);
    x.iter()
        .map(|xi| {
            (0..config.vocab_size)
                .map(|v| {
                    let mut sum = 0.0f32;
                    for d in 0..d_model {
                        let idx = d * config.vocab_size + v;
                        if idx < lm_data.len() {
                            sum += xi[d] * lm_data[idx];
                        }
                    }
                    sum
                })
                .collect()
        })
        .collect()
}

/// 从 logits 采样下一个 token
pub fn sample_token(logits: &[f32], temperature: f32) -> usize {
    let vocab_size = logits.len();
    if vocab_size == 0 {
        return 0;
    }

    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature.max(0.01)).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();

    if sum <= 0.0 {
        return 0;
    }

    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    let mut rng = rand::thread_rng();
    let mut r: f32 = rng.gen::<f32>();
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i;
        }
    }
    vocab_size - 1
}

/// 自回归生成
pub fn generate(
    prompt_tokens: &[usize],
    weights: &GPTWeights,
    max_new_tokens: usize,
    temperature: f32,
) -> Vec<usize> {
    let mut tokens = prompt_tokens.to_vec();
    let config = &weights.config;

    for _ in 0..max_new_tokens {
        let context_len = tokens.len().min(config.max_seq_len);
        let context = &tokens[tokens.len() - context_len..];
        let logits = gpt_forward(context, weights);
        let last_logits = &logits[logits.len() - 1];
        let next_token = sample_token(last_logits, temperature);
        tokens.push(next_token);
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config() {
        let config = ModelConfig::default();
        let params = config.total_params();
        let size_mb = config.quantized_size_mb();
        println!("Total parameters: {}", params);
        println!("Quantized size: {:.2} MB", size_mb);
        assert!(params > 0);
        assert!(size_mb < 10.0);
    }

    #[test]
    fn test_layer_norm() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];
        let beta = vec![0.0f32; 4];
        layer_norm(&mut x, &gamma, &beta, 1e-5);
        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        assert!(mean.abs() < 0.01);
    }

    #[test]
    fn test_gelu() {
        assert!(gelu(0.0).abs() < 0.001);
        assert!(gelu(1.0) > 0.0);
        assert!(gelu(-1.0) < 0.0);
    }
}
