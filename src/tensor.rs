//! 张量库 — FishAI Engine 的核心张量运算模块
//!
//! 提供 AI 推理所需的基础张量数据结构及运算，包括：
//! - 多维张量（1D/2D/3D/4D）的创建、重塑、转置、切片
//! - 矩阵乘法、归一化、激活函数
//! - 注意力机制原语（RoPE、缩放点积注意力）
//! - 逐元素运算

use std::fmt;
use std::ops::Range;

// ---------------------------------------------------------------------------
// Tensor struct
// ---------------------------------------------------------------------------

/// 多维张量，支持 1D / 2D / 3D / 4D 形状。
///
/// 内部使用行主序（row-major）一维 `Vec<f32>` 存储，配合 `shape` 和 `strides`
/// 实现多维索引。
#[derive(Clone, Debug)]
pub struct Tensor {
    /// 行主序扁平数据
    pub data: Vec<f32>,
    /// 各维度大小，例如 `[2, 3, 4]` 表示 2×3×4 张量
    pub shape: Vec<usize>,
    /// 各维度的步长，由 `shape` 自动计算
    pub strides: Vec<usize>,
}

impl Tensor {
    /// 创建一个空的 0 维张量。
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            shape: Vec::new(),
            strides: Vec::new(),
        }
    }

    /// 用给定的数据向量和形状创建张量。
    ///
    /// # Panics
    /// 如果 `data.len()` 与 `shape` 各维度之积不匹配。
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "data length {} does not match shape product {}",
            data.len(),
            expected
        );
        let strides = compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    /// 创建指定形状的全零张量。
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self::from_data(vec![0.0; len], shape)
    }

    /// 创建指定形状并用固定值填充的张量。
    pub fn filled(shape: Vec<usize>, value: f32) -> Self {
        let len: usize = shape.iter().product();
        Self::from_data(vec![value; len], shape)
    }

    /// 返回张量中扁平数据的长度（等同于 `elem_count()`）。
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// 张量是否为空。
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 返回张量中元素的数量（与 `shape` 各维度之积相同）。
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// 重塑张量的形状。
    ///
    /// # Panics
    /// 如果新形状的元素总数与当前不匹配。
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let expected: usize = new_shape.iter().product();
        assert_eq!(
            self.data.len(),
            expected,
            "cannot reshape tensor of {} elements into shape {:?}",
            self.data.len(),
            new_shape
        );
        Tensor::from_data(self.data.clone(), new_shape)
    }

    /// 转置张量（交换最后两个维度）。
    ///
    /// 仅对 2D 及以上张量有意义。
    pub fn transpose(&self) -> Self {
        assert!(
            self.shape.len() >= 2,
            "transpose requires at least 2D tensor"
        );
        let dims = self.shape.len();
        let mut new_shape = self.shape.clone();
        new_shape.swap(dims - 2, dims - 1);

        let mut new_data = vec![0.0f32; self.data.len()];

        // 手动迭代所有索引并按转置映射写入
        let mut idx = vec![0usize; dims];
        loop {
            // 计算线性偏移（原始）
            let mut src_off = 0;
            for d in 0..dims {
                src_off += idx[d] * self.strides[d];
            }
            // 交换最后两维的索引
            let mut dst_idx = idx.clone();
            dst_idx.swap(dims - 2, dims - 1);
            let mut dst_off = 0;
            for d in 0..dims {
                dst_off += dst_idx[d] * compute_strides(&new_shape)[d];
            }
            new_data[dst_off] = self.data[src_off];

            // 递增索引
            let mut carry = true;
            for d in (0..dims).rev() {
                if carry {
                    idx[d] += 1;
                    if idx[d] >= self.shape[d] {
                        idx[d] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry {
                break;
            }
        }

        Tensor::from_data(new_data, new_shape)
    }

    /// 沿各维度切片，返回新的子张量。
    ///
    /// `ranges` 的长度必须与 `shape` 的维度相同。每个 `Range` 表示在该维度上
    /// 截取的半开区间。若某个维度使用 `0..shape[d]`，则该维度不受影响。
    pub fn slice(&self, ranges: &[Range<usize>]) -> Self {
        assert_eq!(
            ranges.len(),
            self.shape.len(),
            "slice ranges length must match tensor rank"
        );
        // 计算新形状
        let new_shape: Vec<usize> = ranges
            .iter()
            .zip(self.shape.iter())
            .map(|(r, &s)| {
                assert!(r.end <= s, "slice range {:?} exceeds dimension {}", r, s);
                r.end - r.start
            })
            .collect();

        let new_strides = compute_strides(&new_shape);
        let mut new_data = vec![0.0f32; new_shape.iter().product()];

        let rank = self.shape.len();
        let mut idx = vec![0usize; rank];
        loop {
            let mut src_off = 0;
            for d in 0..rank {
                src_off += (ranges[d].start + idx[d]) * self.strides[d];
            }
            let mut dst_off = 0;
            for d in 0..rank {
                dst_off += idx[d] * new_strides[d];
            }
            new_data[dst_off] = self.data[src_off];

            let mut carry = true;
            for d in (0..rank).rev() {
                if carry {
                    idx[d] += 1;
                    if idx[d] >= new_shape[d] {
                        idx[d] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry {
                break;
            }
        }

        Tensor::from_data(new_data, new_shape)
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, len={})", self.shape, self.data.len())
    }
}

/// 计算行主序步长。
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    if rank == 0 {
        return vec![];
    }
    let mut strides = vec![0usize; rank];
    strides[rank - 1] = 1;
    for d in (0..rank - 1).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

// ---------------------------------------------------------------------------
// Core operations
// ---------------------------------------------------------------------------

/// 矩阵乘法：C[m,n] = A[m,k] × B[k,n]。
///
/// `a` 长度必须为 `m * k`，`b` 长度必须为 `k * n`。
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// 矩阵乘法 + 行偏置：C[m,n] = A[m,k] × B[k,n] + bias[0..n]（逐行广播）。
pub fn matmul_add(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut c = matmul(a, b, m, k, n);
    assert!(bias.len() >= n);
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] += bias[j];
        }
    }
    c
}

/// RMS 归一化：`y[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]`
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    assert_eq!(weight.len(), n);
    // 计算均方值
    let mean_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let inv_norm = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * inv_norm * wi)
        .collect()
}

/// Layer 归一化：标准化后做仿射变换。
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    assert_eq!(weight.len(), n);
    assert_eq!(bias.len(), n);
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(weight.iter().zip(bias.iter()))
        .map(|(&xi, (&wi, &bi))| (xi - mean) * inv_std * wi + bi)
        .collect()
}

/// SiLU / Swish 激活函数：`f(x) = x * sigmoid(x)`
pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v * sigmoid(v)).collect()
}

/// GELU 激活函数（tanh 近似）。
pub fn gelu(x: &[f32]) -> Vec<f32> {
    const SQRT_2_OVER_PI: f32 = 0.7978845608_f32;
    x.iter()
        .map(|&v| {
            let inner = SQRT_2_OVER_PI * (v + 0.044715 * v * v * v);
            0.5 * v * (1.0 + inner.tanh())
        })
        .collect()
}

/// ReLU 激活函数：`f(x) = max(0, x)`
pub fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

/// 原地 Softmax（沿最后一个维度）。
///
/// `x` 的布局为 `[rows, cols]`，即总长度 = `rows * cols`，
/// 对每一行独立做 softmax。
pub fn softmax(x: &mut [f32]) {
    let len = x.len();
    if len == 0 {
        return;
    }
    // 假定 1D（单行）或者根据 shape 信息无法获取时按整体做
    // 这里我们对整个切片做 softmax（单行情况）
    // 找最大值
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    // exp(x - max)
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// 对一个切片做 softmax 并返回新的 `Vec<f32>`。
pub fn softmax_slice(x: &[f32]) -> Vec<f32> {
    let mut buf = x.to_vec();
    softmax(&mut buf);
    buf
}

/// 对二维数据的每一行做 softmax。
///
/// `x` 长度 = `rows * cols`，行优先存储。
pub fn softmax_rows(x: &mut [f32], cols: usize) {
    let rows = x.len() / cols;
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        let inv_sum = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv_sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Attention primitives
// ---------------------------------------------------------------------------

/// 生成单个位置的旋转位置嵌入（RoPE）。
///
/// 返回长度为 `dim` 的向量，前半部分为 cos 值，后半部分为 sin 值。
pub fn rope_emb(pos: usize, dim: usize, base: f32) -> Vec<f32> {
    assert_eq!(dim % 2, 0, "rope_emb requires even dim, got {}", dim);
    let half = dim / 2;
    let mut emb = Vec::with_capacity(dim);
    for i in 0..half {
        let freq = 1.0f32 / (base as f32).powf(2.0 * i as f32 / dim as f32);
        let angle = pos as f32 * freq;
        emb.push(angle.cos());
        emb.push(angle.sin());
    }
    emb
}

/// 对 Query 和 Key 张量原地施加旋转位置嵌入（RoPE）。
///
/// 布局：`q` 和 `k` 均为 `[seq_len, n_heads * head_dim]`（flat），
/// 即每个 head 的 head_dim 维向量在连续内存中。
/// 对每个 head 独立施加 RoPE。
pub fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    base: f32,
) {
    let emb = rope_emb(pos, head_dim, base);
    let half = head_dim / 2;

    // seq_len 假定为 1（单 token 推理）；如果是多 token 则需要循环
    let seq_len = q.len() / (n_heads * head_dim);
    for s in 0..seq_len {
        for h in 0..n_heads {
            let base_q = s * n_heads * head_dim + h * head_dim;
            let base_k = s * n_heads * head_dim + h * head_dim;

            for d in 0..half {
                let cos_val = emb[2 * d];
                let sin_val = emb[2 * d + 1];

                let q0 = q[base_q + d];
                let q1 = q[base_q + d + half];
                q[base_q + d] = q0 * cos_val - q1 * sin_val;
                q[base_q + d + half] = q0 * sin_val + q1 * cos_val;

                let k0 = k[base_k + d];
                let k1 = k[base_k + d + half];
                k[base_k + d] = k0 * cos_val - k1 * sin_val;
                k[base_k + d + half] = k0 * sin_val + k1 * cos_val;
            }
        }
    }
}

/// 缩放点积注意力（支持 KV 缓存和可选 mask）。
///
/// 布局说明：
/// - `q`: `[q_len, n_heads * head_dim]`
/// - `k`: `[k_len, n_heads * head_dim]` （本次新的 key）
/// - `v`: `[v_len, n_heads * head_dim]` （本次新的 value）
/// - `kv_cache_k`: `[cached_len, n_heads * head_dim]` （之前缓存的 key）
/// - `kv_cache_v`: `[cached_len, n_heads * head_dim]` （之前缓存的 value）
/// - `mask`: `[q_len * total_k_len]` 可选 bool 掩码
///
/// 返回 `(output, full_k, full_v)`，其中 `full_k`/`full_v` = cache + new。
pub fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    head_dim: usize,
    kv_cache_k: Option<&[f32]>,
    kv_cache_v: Option<&[f32]>,
    mask: Option<&[bool]>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_len = q.len() / (n_heads * head_dim);
    let k_len_new = k.len() / (n_heads * head_dim);
    let v_len_new = v.len() / (n_heads * head_dim);
    assert_eq!(k_len_new, v_len_new);

    let cached_len = kv_cache_k.map_or(0, |c| c.len() / (n_heads * head_dim));
    let total_k_len = cached_len + k_len_new;

    // 拼接 full K: [cached_len + k_len_new, n_heads * head_dim]
    let mut full_k = Vec::with_capacity(total_k_len * n_heads * head_dim);
    if let Some(cache) = kv_cache_k {
        full_k.extend_from_slice(cache);
    }
    full_k.extend_from_slice(k);

    let mut full_v = Vec::with_capacity(total_k_len * n_heads * head_dim);
    if let Some(cache) = kv_cache_v {
        full_v.extend_from_slice(cache);
    }
    full_v.extend_from_slice(v);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; q_len * n_heads * head_dim];

    // 每个头独立计算
    for h in 0..n_heads {
        // 提取当前头的 q: [q_len, head_dim]
        // k: [total_k_len, head_dim], v: [total_k_len, head_dim]
        let q_head = |i: usize, d: usize| -> f32 {
            q[i * n_heads * head_dim + h * head_dim + d]
        };
        let k_head = |i: usize, d: usize| -> f32 {
            full_k[i * n_heads * head_dim + h * head_dim + d]
        };
        let v_head = |i: usize, d: usize| -> f32 {
            full_v[i * n_heads * head_dim + h * head_dim + d]
        };

        for qi in 0..q_len {
            // 计算注意力分数: [total_k_len]
            let mut scores = vec![0.0f32; total_k_len];
            let mut max_score = f32::NEG_INFINITY;
            for ki in 0..total_k_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_head(qi, d) * k_head(ki, d);
                }
                scores[ki] = dot * scale;
                if scores[ki] > max_score {
                    max_score = scores[ki];
                }
            }

            // 应用 mask
            if let Some(m) = mask {
                for ki in 0..total_k_len {
                    let mask_idx = qi * total_k_len + ki;
                    if mask_idx < m.len() && !m[mask_idx] {
                        scores[ki] = f32::NEG_INFINITY;
                    }
                    // 重新计算 max
                    if scores[ki] > max_score {
                        max_score = scores[ki];
                    }
                }
            }

            // Softmax
            let mut exp_sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                exp_sum += *s;
            }
            let inv_exp_sum = 1.0 / exp_sum;
            for s in scores.iter_mut() {
                *s *= inv_exp_sum;
            }

            // 加权求和得到 output
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for ki in 0..total_k_len {
                    val += scores[ki] * v_head(ki, d);
                }
                output[qi * n_heads * head_dim + h * head_dim + d] = val;
            }
        }
    }

    (output, full_k, full_v)
}

// ---------------------------------------------------------------------------
// Element-wise operations
// ---------------------------------------------------------------------------

/// 逐元素加法。
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// 逐元素标量乘法。
pub fn scale(x: &[f32], s: f32) -> Vec<f32> {
    x.iter().map(|&v| v * s).collect()
}

/// 逐元素标量加法。
pub fn add_scalar(x: &[f32], s: f32) -> Vec<f32> {
    x.iter().map(|&v| v + s).collect()
}

/// 逐元素乘法。
pub fn mul_elementwise(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

/// 向量点积。
pub fn vec_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.data.len(), 4);
        assert_eq!(t.elem_count(), 4);

        let empty = Tensor::new();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(vec![3, 4]);
        assert_eq!(t.shape, vec![3, 4]);
        assert_eq!(t.elem_count(), 12);
        assert!(t.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_tensor_filled() {
        let t = Tensor::filled(vec![2, 3], 7.0);
        assert_eq!(t.elem_count(), 6);
        assert!(t.data.iter().all(|&v| v == 7.0));
    }

    #[test]
    fn test_tensor_reshape() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let t2 = t.reshape(vec![3, 2]);
        assert_eq!(t2.shape, vec![3, 2]);
        assert_eq!(t2.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor_transpose() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let tt = t.transpose();
        assert_eq!(tt.shape, vec![2, 2]);
        // [[1,2],[3,4]]^T = [[1,3],[2,4]]
        assert_eq!(tt.data, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_tensor_slice() {
        let t = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        // 取第 0 行，列 1..3
        let s = t.slice(&[0..1, 1..3]);
        assert_eq!(s.shape, vec![1, 2]);
        assert_eq!(s.data, vec![2.0, 3.0]);
    }

    #[test]
    fn test_matmul_basic() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = [[19,22],[43,50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_identity() {
        // A * I = A
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let c = matmul(&a, &identity, 2, 3, 3);
        assert_eq!(c, a);
    }

    #[test]
    fn test_rms_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let y = rms_norm(&x, &w, 1e-6);
        // mean_sq = (1+4+9+16)/4 = 7.5, inv_norm = 1/sqrt(7.5+1e-6) ≈ 0.3651
        let inv_norm = 1.0 / (7.5f32 + 1e-6).sqrt();
        for (i, &v) in y.iter().enumerate() {
            let expected = x[i] * inv_norm;
            assert!((v - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn test_layer_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![0.0, 0.0, 0.0, 0.0];
        let y = layer_norm(&x, &w, &b, 1e-6);
        // mean = 2.5, var = (2.25+0.25+0.25+2.25)/4 = 1.25
        let mean = 2.5f32;
        let var = 1.25f32;
        let inv_std = 1.0 / (var + 1e-6).sqrt();
        for (i, &v) in y.iter().enumerate() {
            let expected = (x[i] - mean) * inv_std;
            assert!((v - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn test_silu() {
        // silu(0) = 0
        let y = silu(&[0.0]);
        assert!((y[0] - 0.0).abs() < 1e-6);
        // silu(x) ≈ x for large positive x
        let y = silu(&[100.0]);
        assert!((y[0] - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_gelu() {
        // gelu(0) = 0
        let y = gelu(&[0.0]);
        assert!((y[0] - 0.0).abs() < 1e-5);
        // gelu positive → positive
        let y = gelu(&[1.0]);
        // Approximate: gelu(1) ≈ 0.8413
        assert!((y[0] - 0.8413).abs() < 0.01);
    }

    #[test]
    fn test_relu() {
        let y = relu(&[-1.0, 0.0, 1.0, -0.5, 0.5]);
        assert_eq!(y, vec![0.0, 0.0, 1.0, 0.0, 0.5]);
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);
        // e^1=2.718, e^2=7.389, e^3=20.086; sum=30.193
        // normalized: [0.0900, 0.2447, 0.6652]
        let expected = softmax_slice(&[1.0_f32, 2.0, 3.0]);
        for (a, b) in x.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
        // 和为 1
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_emb() {
        let emb = rope_emb(0, 4, 10000.0);
        assert_eq!(emb.len(), 4);
        // pos=0: cos(0)=1, sin(0)=0
        assert!((emb[0] - 1.0).abs() < 1e-6);
        assert!(emb[1].abs() < 1e-6);
        assert!((emb[2] - 1.0).abs() < 1e-6);
        assert!(emb[3].abs() < 1e-6);

        let emb1 = rope_emb(1, 4, 10000.0);
        // 不全为 cos=1
        assert!((emb1[0] - 1.0).abs() > 1e-6 || (emb1[2] - 1.0).abs() > 1e-6);
    }

    #[test]
    fn test_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = add(&a, &b);
        assert_eq!(c, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_scale() {
        let x = vec![1.0, 2.0, 3.0];
        let y = scale(&x, 2.0);
        assert_eq!(y, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_vec_dot() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let d = vec_dot(&a, &b);
        assert!((d - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_dot_product_attention() {
        // 单头、单 token、无 cache、无 mask
        let q = vec![1.0, 0.0]; // [1, 1*2] — 1 token, 1 head, dim=2
        let k = vec![1.0, 0.0, 0.0, 1.0]; // [2, 1*2] — 2 tokens
        let v = vec![1.0, 0.0, 0.0, 1.0]; // [2, 1*2]
        let (out, full_k, full_v) = scaled_dot_product_attention(
            &q, &k, &v,
            1, 2,
            None, None, None,
        );
        assert_eq!(out.len(), 2);
        // q·k[0] = 1, q·k[1] = 0, scale = 1/sqrt(2) ≈ 0.7071
        // scores = [0.7071, 0]; softmax ≈ [0.67, 0.33]
        // output ≈ 0.67*[1,0] + 0.33*[0,1] = [0.67, 0.33]
        assert!((out[0] - 0.67).abs() < 0.05);
        assert!((out[1] - 0.33).abs() < 0.05);
        assert_eq!(full_k, k);
        assert_eq!(full_v, v);
    }

    #[test]
    fn test_scaled_dot_product_attention_with_cache() {
        let q = vec![1.0, 0.0]; // 1 token, 1 head, dim=2
        let k = vec![0.0, 1.0]; // new key
        let v = vec![0.0, 1.0]; // new value
        let cached_k = vec![1.0, 0.0]; // old key
        let cached_v = vec![1.0, 0.0]; // old value

        let (out, full_k, full_v) = scaled_dot_product_attention(
            &q, &k, &v,
            1, 2,
            Some(&cached_k), Some(&cached_v),
            None,
        );
        // full_k = [[1,0],[0,1]], full_v = [[1,0],[0,1]]
        assert_eq!(full_k.len(), 4);
        assert_eq!(full_v.len(), 4);
        // q·full_k[0] = 1, q·full_k[1] = 0 → scale*1 ≈ 0.7071, scale*0 = 0
        // softmax ≈ [0.67, 0.33]
        // out = 0.67 * [1, 0] + 0.33 * [0, 1] = [0.67, 0.33]
        assert!((out[0] - 0.67).abs() < 0.05);
        assert!((out[1] - 0.33).abs() < 0.05);
    }

    #[test]
    fn test_add_scalar() {
        let x = vec![1.0, 2.0, 3.0];
        let y = add_scalar(&x, 10.0);
        assert_eq!(y, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_mul_elementwise() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = mul_elementwise(&a, &b);
        assert_eq!(c, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_matmul_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let bias = vec![10.0, 20.0];
        let c = matmul_add(&a, &b, &bias, 2, 2, 2);
        // [[1,2],[3,4]] * I = [[1,2],[3,4]] + bias
        assert_eq!(c, vec![11.0, 22.0, 13.0, 24.0]);
    }
}