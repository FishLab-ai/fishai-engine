//! FishAI Token 采样器
//!
//! 在模型输出 logits（每个 token 的概率分布）上进行采样，决定下一个 token。
//!
//! 支持策略：
//! - Temperature：控制随机性（0 = 贪婪，>0 越大越随机）
//! - Top-K：只从概率最高的 K 个 token 中采样
//! - Top-P（nucleus）：从累计概率 ≤ P 的最小 token 集合中采样
//! - Repetition Penalty：对已出现过的 token 降低概率
//! - Frequency Penalty：对高频 token 额外降权

use rand::Rng;

/// 采样器配置
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 温度（0.0~2.0，默认 0.7）
    pub temperature: f32,
    /// Top-K 截断（0 = 不限制，默认 40）
    pub top_k: usize,
    /// Top-P 截断（0.0~1.0，0 = 不限制，默认 0.95）
    pub top_p: f32,
    /// 重复惩罚系数（1.0 = 不惩罚，>1 越大越惩罚，默认 1.15）
    pub repetition_penalty: f32,
    /// 频率惩罚（0.0 = 不惩罚，默认 0.0）
    pub frequency_penalty: f32,
    /// 存在惩罚（0.0 = 不惩罚，默认 0.0）
    pub presence_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.15,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

impl SamplerConfig {
    /// 贪婪解码（temperature=0）
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }

    /// 创意模式（高温度、低惩罚）
    pub fn creative() -> Self {
        Self {
            temperature: 1.2,
            top_k: 80,
            top_p: 0.98,
            repetition_penalty: 1.05,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }

    /// 精确模式（低温度、高惩罚）
    pub fn precise() -> Self {
        Self {
            temperature: 0.3,
            top_k: 20,
            top_p: 0.9,
            repetition_penalty: 1.3,
            frequency_penalty: 0.1,
            presence_penalty: 0.1,
        }
    }
}

/// Token 采样器
///
/// 接收模型输出的 logits 和已生成的 token 序列，
/// 按配置策略选出下一个 token。
pub struct Sampler {
    config: SamplerConfig,
    /// 每个 token 的出现次数（用于频率惩罚）
    token_freq: Vec<usize>,
    /// 已出现过的 token 集合（用于存在惩罚和重复惩罚）
    token_present: Vec<bool>,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            config,
            token_freq: Vec::new(),
            token_present: Vec::new(),
        }
    }

    /// 从 logits 中采样一个 token
    ///
    /// # Arguments
    /// * `logits` - 模型输出的原始 logits（未归一化的分数）
    ///
    /// # Returns
    /// 采样的 token 索引
    pub fn sample(&mut self, logits: &[f32]) -> usize {
        let vocab_size = logits.len();

        // 确保 token 计数器足够大
        self.ensure_capacity(vocab_size);

        // 1. 应用重复惩罚
        let mut adjusted = self.apply_repetition_penalty(logits);

        // 2. 应用频率/存在惩罚（在 logit 空间减去惩罚值）
        self.apply_frequency_presence_penalty(&mut adjusted);

        // 3. Temperature 缩放
        if self.config.temperature > 0.0 && self.config.temperature != 1.0 {
            let temp = self.config.temperature;
            for v in adjusted.iter_mut() {
                *v /= temp;
            }
        }

        // 4. Softmax 归一化 → 概率
        let probs = softmax(&adjusted);

        // 5. Top-K 过滤
        let mut filtered = if self.config.top_k > 0 && self.config.top_k < vocab_size {
            top_k_filter(&probs, self.config.top_k)
        } else {
            probs
        };

        // 6. Top-P (nucleus) 过滤
        if self.config.top_p > 0.0 && self.config.top_p < 1.0 {
            top_p_filter(&mut filtered);
        }

        // 7. 采样
        if self.config.temperature == 0.0 || filtered.iter().all(|&p| p <= 0.0) {
            // 贪婪模式或全部概率为 0 → 取最高
            argmax(&filtered)
        } else {
            categorical_sample(&filtered)
        }
    }

    /// 记录一个 token 被选中（更新频率/存在状态）
    pub fn observe_token(&mut self, token_id: usize) {
        self.ensure_capacity(token_id + 1);
        self.token_freq[token_id] += 1;
        self.token_present[token_id] = true;
    }

    /// 重置采样器状态（新对话时调用）
    pub fn reset(&mut self) {
        self.token_freq.clear();
        self.token_present.clear();
    }

    fn ensure_capacity(&mut self, size: usize) {
        if self.token_freq.len() < size {
            self.token_freq.resize(size, 0);
            self.token_present.resize(size, false);
        }
    }

    fn apply_repetition_penalty(&self, logits: &[f32]) -> Vec<f32> {
        let penalty = self.config.repetition_penalty;
        if penalty == 1.0 || self.token_present.is_empty() {
            return logits.to_vec();
        }

        logits
            .iter()
            .enumerate()
            .map(|(i, &logit)| {
                if i < self.token_present.len() && self.token_present[i] {
                    // 正 logit → 除以 penalty（降低概率）
                    // 负 logit → 乘以 penalty（降低概率）
                    if logit > 0.0 {
                        logit / penalty
                    } else {
                        logit * penalty
                    }
                } else {
                    logit
                }
            })
            .collect()
    }

    fn apply_frequency_presence_penalty(&self, logits: &mut [f32]) {
        let freq_p = self.config.frequency_penalty;
        let pres_p = self.config.presence_penalty;

        if freq_p == 0.0 && pres_p == 0.0 {
            return;
        }

        for (i, logit) in logits.iter_mut().enumerate() {
            if i >= self.token_freq.len() {
                break;
            }
            if self.token_freq[i] > 0 {
                *logit -= freq_p * self.token_freq[i] as f32;
                *logit -= pres_p;
            }
        }
    }
}

// ==================== 数学工具函数 ====================

/// Softmax：将 logits 转为概率分布
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        exps
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// Top-K 过滤：只保留概率最高的 K 个 token
pub fn top_k_filter(probs: &[f32], k: usize) -> Vec<f32> {
    if k == 0 {
        return probs.to_vec();
    }
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = vec![0.0; probs.len()];
    for (i, &(idx, _)) in indexed.iter().enumerate() {
        if i < k {
            result[idx] = probs[idx];
        }
    }
    result
}

/// Top-P (nucleus) 过滤：只保留累计概率 ≤ P 的最小集合
pub fn top_p_filter(probs: &mut [f32]) {
    let total: f32 = probs.iter().sum();
    if total <= 0.0 {
        return;
    }

    // 按 probability 排序的索引
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0_f32;
    let mut to_zero = Vec::new();

    for &(idx, prob) in &indexed {
        cumsum += prob / total;
        if cumsum > 0.999 {
            to_zero.push(idx);
        }
    }

    // 不在 top-p 内的全部置零
    let top_indices: std::collections::HashSet<usize> =
        indexed.iter().take_while(|(_, p)| *p > 0.0).map(|(i, _)| *i).collect();

    for (i, prob) in probs.iter_mut().enumerate() {
        if to_zero.contains(&i) || !top_indices.contains(&i) {
            *prob = 0.0;
        }
    }
}

/// 分类采样：按概率分布随机选择
pub fn categorical_sample(probs: &[f32]) -> usize {
    let mut rng = rand::rng();
    let mut r: f32 = rng.random_range(0.0..1.0);
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i;
        }
    }
    // 兜底：返回概率最高的
    argmax(probs)
}

/// 返回第一个最大值的索引
pub fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .reduce(|acc, (i, v)| {
            if v > acc.1 {
                (i, v)
            } else {
                acc
            }
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_single_element() {
        let logits = vec![5.0];
        let probs = softmax(&logits);
        assert!((probs[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_empty() {
        let logits: Vec<f32> = vec![];
        let probs = softmax(&logits);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_softmax_negative_values() {
        let logits = vec![-10.0, 0.0, 10.0];
        let probs = softmax(&logits);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(probs[2] > 0.9); // 10 is way larger
    }

    #[test]
    fn test_top_k_filter() {
        let probs = vec![0.1, 0.7, 0.15, 0.05];
        let filtered = top_k_filter(&probs, 2);
        assert_eq!(filtered[0], 0.0); // 0.1 < top 2
        assert_eq!(filtered[1], 0.7); // 0.7 in top 2
        assert_eq!(filtered[2], 0.15); // 0.15 in top 2
        assert_eq!(filtered[3], 0.0); // 0.05 < top 2
    }

    #[test]
    fn test_top_k_filter_preserves_order() {
        let probs = vec![0.4, 0.3, 0.2, 0.1];
        let filtered = top_k_filter(&probs, 3);
        assert_eq!(filtered, vec![0.4, 0.3, 0.2, 0.0]);
    }

    #[test]
    fn test_top_k_zero_means_no_filter() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let filtered = top_k_filter(&probs, 0);
        // k=0 意味着不限制
        assert_eq!(filtered, probs);
    }

    #[test]
    fn test_top_p_filter() {
        let mut probs = vec![0.6, 0.25, 0.1, 0.05];
        top_p_filter(&mut probs);
        let sum: f32 = probs.iter().sum();
        assert!(sum > 0.0);
        // top 2 tokens 累计 0.85 > 0.8 所以大约保留前 2-3 个
        assert!(probs[0] > 0.0);
        assert!(probs[1] > 0.0);
    }

    #[test]
    fn test_argmax() {
        let values = vec![0.1, 0.9, 0.3];
        assert_eq!(argmax(&values), 1);
    }

    #[test]
    fn test_argmax_empty() {
        let values: Vec<f32> = vec![];
        assert_eq!(argmax(&values), 0);
    }

    #[test]
    fn test_argmax_tie_returns_first() {
        let values = vec![0.5, 0.5, 0.3];
        assert_eq!(argmax(&values), 0);
    }

    #[test]
    fn test_categorical_sample_deterministic_for_extreme() {
        let probs = vec![0.0, 0.0, 1.0, 0.0];
        // 概率为 1.0 的应该总被选中
        for _ in 0..100 {
            let idx = categorical_sample(&probs);
            assert_eq!(idx, 2);
        }
    }

    #[test]
    fn test_sampler_greedy() {
        let config = SamplerConfig::greedy();
        let mut sampler = Sampler::new(config);
        let logits = vec![0.1, 5.0, 0.3, 0.2];
        let token = sampler.sample(&logits);
        assert_eq!(token, 1); // logit 最大
    }

    #[test]
    fn test_sampler_default() {
        let config = SamplerConfig::default();
        let mut sampler = Sampler::new(config);
        // 768 个模拟 logits，其中一个远大于其他
        let mut logits = vec![0.0; 768];
        logits[100] = 10.0;
        logits[200] = 8.0;
        logits[300] = 6.0;
        // 大多数采样应该落到前几个
        let mut counts = vec![0usize; 768];
        for _ in 0..100 {
            let t = sampler.sample(&logits);
            counts[t] += 1;
        }
        // token 100 应该被选中最多
        assert!(counts[100] > 50);
    }

    #[test]
    fn test_repetition_penalty_reduces_seen_tokens() {
        let config = SamplerConfig {
            temperature: 0.0,
            repetition_penalty: 10.0, // 极端惩罚
            ..SamplerConfig::greedy()
        };
        let mut sampler = Sampler::new(config);
        sampler.observe_token(0); // token 0 已出现

        let logits = vec![5.0, 3.0];
        let token = sampler.sample(&logits);
        // token 0 的 logit 被 5.0/10.0=0.5，token 1 的 3.0 不变
        assert_eq!(token, 1); // 惩罚后 token 1 概率更大
    }

    #[test]
    fn test_sampler_observe_and_reset() {
        let config = SamplerConfig::default();
        let mut sampler = Sampler::new(config);
        sampler.observe_token(42);
        sampler.observe_token(42);
        sampler.observe_token(100);

        assert_eq!(sampler.token_freq[42], 2);
        assert_eq!(sampler.token_freq[100], 1);

        sampler.reset();
        assert!(sampler.token_freq.is_empty());
        assert!(sampler.token_present.is_empty());
    }

    #[test]
    fn test_frequency_penalty() {
        let config = SamplerConfig {
            temperature: 0.0,
            frequency_penalty: 100.0, // 极端
            ..SamplerConfig::greedy()
        };
        let mut sampler = Sampler::new(config);
        sampler.observe_token(0);
        sampler.observe_token(0);

        let logits = vec![200.0, 3.0]; // 即使 logit 极大，频率惩罚后应小于 token 1
        let token = sampler.sample(&logits);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_presence_penalty() {
        let config = SamplerConfig {
            temperature: 0.0,
            presence_penalty: 5.0,
            ..SamplerConfig::greedy()
        };
        let mut sampler = Sampler::new(config);
        sampler.observe_token(0);

        let logits = vec![4.0, 3.0];
        // token 0: 4.0 - 5.0 = -1.0, token 1: 3.0
        let token = sampler.sample(&logits);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_precise_config() {
        let config = SamplerConfig::precise();
        assert_eq!(config.temperature, 0.3);
        assert_eq!(config.top_k, 20);
        assert!(config.repetition_penalty > 1.0);
    }

    #[test]
    fn test_creative_config() {
        let config = SamplerConfig::creative();
        assert!(config.temperature > 1.0);
        assert_eq!(config.top_k, 80);
    }
}
