//! KV 缓存模块 — Transformer 自回归推理的键值缓存
//!
//! 在自回归生成过程中，每一步只需要计算最新 token 的 Query、Key、Value，
//! 而之前所有 token 的 Key 和 Value 需要被缓存以便注意力计算复用。
//!
//! 本模块提供了分层 KV 缓存，按 Transformer 层索引存储历史 K/V 向量，
//! 支持 GQA（分组查询注意力）中 KV 头数与 Query 头数不同的情况。

/// Transformer KV 缓存，按层存储历史 Key 和 Value 向量。
///
/// 每层维护一个动态增长的 K 向量列表和 V 向量列表，
/// 布局为 `[seq_len, n_kv_heads * head_dim]`（行优先）。
pub struct KVCache {
    /// 每层缓存的 Key 向量（拼接后的完整历史）
    cache_k: Vec<Vec<f32>>,
    /// 每层缓存的 Value 向量（拼接后的完整历史）
    cache_v: Vec<Vec<f32>>,
    /// Transformer 层数
    n_layers: usize,
}

impl KVCache {
    /// 创建空的 KV 缓存。
    ///
    /// # Arguments
    /// * `n_layers` - Transformer 层数
    pub fn new(n_layers: usize) -> Self {
        let mut cache_k = Vec::with_capacity(n_layers);
        let mut cache_v = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            cache_k.push(Vec::new());
            cache_v.push(Vec::new());
        }
        Self {
            cache_k,
            cache_v,
            n_layers,
        }
    }

    /// 向指定层追加新的 Key 和 Value 向量。
    ///
    /// # Arguments
    /// * `layer` - 层索引（0-based）
    /// * `k` - 新的 Key 向量，长度应为 `n_kv_heads * head_dim`
    /// * `v` - 新的 Value 向量，长度应为 `n_kv_heads * head_dim`
    ///
    /// # Panics
    /// 如果 `layer >= n_layers`
    pub fn update(&mut self, layer: usize, k: Vec<f32>, v: Vec<f32>) {
        assert!(layer < self.n_layers, "KV 缓存层索引越界: {} >= {}", layer, self.n_layers);
        self.cache_k[layer].extend_from_slice(&k);
        self.cache_v[layer].extend_from_slice(&v);
    }

    /// 获取指定层的完整缓存 K 和 V（只读引用）。
    ///
    /// # Arguments
    /// * `layer` - 层索引（0-based）
    ///
    /// # Returns
    /// 元组 `(k_slice, v_slice)`，布局均为 `[seq_len, n_kv_heads * head_dim]`。
    ///
    /// # Panics
    /// 如果 `layer >= n_layers`
    pub fn get(&self, layer: usize) -> (&[f32], &[f32]) {
        assert!(layer < self.n_layers, "KV 缓存层索引越界: {} >= {}", layer, self.n_layers);
        (&self.cache_k[layer], &self.cache_v[layer])
    }

    /// 获取指定层当前缓存的序列长度（已存储的 token 数）。
    ///
    /// # Panics
    /// 如果 `layer >= n_layers`
    pub fn seq_len(&self, layer: usize) -> usize {
        assert!(layer < self.n_layers, "KV 缓存层索引越界: {} >= {}", layer, self.n_layers);
        if self.cache_k[layer].is_empty() {
            0
        } else {
            // 暂时返回 0，实际长度由外部根据 head 维度计算
            0
        }
    }

    /// 获取指定层的序列长度（基于 K 缓存和每 token 的 KV 维度计算）。
    ///
    /// # Arguments
    /// * `layer` - 层索引
    /// * `kv_dim` - 每个 token 的 KV 维度 = `n_kv_heads * head_dim`
    pub fn seq_len_with_dim(&self, layer: usize, kv_dim: usize) -> usize {
        assert!(layer < self.n_layers, "KV 缓存层索引越界: {} >= {}", layer, self.n_layers);
        if kv_dim == 0 {
            return 0;
        }
        self.cache_k[layer].len() / kv_dim
    }

    /// 清空所有层的 KV 缓存（用于新对话重置）。
    pub fn clear(&mut self) {
        for i in 0..self.n_layers {
            self.cache_k[i].clear();
            self.cache_v[i].clear();
        }
    }

    /// 返回层数。
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new(4);
        assert_eq!(cache.n_layers(), 4);
        for i in 0..4 {
            let (k, v) = cache.get(i);
            assert!(k.is_empty());
            assert!(v.is_empty());
        }
    }

    #[test]
    fn test_kv_cache_update_and_get() {
        let mut cache = KVCache::new(2);

        // 向第 0 层追加一个 token 的 K/V（假设 kv_dim = 8）
        let k0 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v0 = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        cache.update(0, k0.clone(), v0.clone());

        let (k, v) = cache.get(0);
        assert_eq!(k, &k0);
        assert_eq!(v, &v0);

        // 再追加一个 token
        let k1 = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let v1 = vec![80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0];
        cache.update(0, k1.clone(), v1.clone());

        let (k, _v) = cache.get(0);
        let mut expected_k = k0.clone();
        expected_k.extend_from_slice(&k1);
        assert_eq!(k, &expected_k);
        assert_eq!(k.len(), 16); // 2 tokens × 8 dim

        // 第 1 层应该仍然为空
        let (k1_empty, _v) = cache.get(1);
        assert!(k1_empty.is_empty());
    }

    #[test]
    fn test_kv_cache_seq_len() {
        let mut cache = KVCache::new(1);
        let kv_dim = 16;

        assert_eq!(cache.seq_len_with_dim(0, kv_dim), 0);

        cache.update(0, vec![0.0; kv_dim], vec![0.0; kv_dim]);
        assert_eq!(cache.seq_len_with_dim(0, kv_dim), 1);

        cache.update(0, vec![0.0; kv_dim], vec![0.0; kv_dim]);
        assert_eq!(cache.seq_len_with_dim(0, kv_dim), 2);
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KVCache::new(2);
        cache.update(0, vec![1.0; 8], vec![2.0; 8]);
        cache.update(1, vec![3.0; 8], vec![4.0; 8]);

        cache.clear();

        for i in 0..2 {
            let (k, v) = cache.get(i);
            assert!(k.is_empty());
            assert!(v.is_empty());
        }
    }

    #[test]
    #[should_panic(expected = "KV 缓存层索引越界")]
    fn test_kv_cache_out_of_bounds() {
        let cache = KVCache::new(1);
        cache.get(5);
    }
}