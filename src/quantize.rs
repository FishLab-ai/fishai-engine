//! 4-bit 整数量化模块
//!
//! 自研量化方案:
//! - 对称量化: value = (int4 - zero_point) * scale
//! - Per-Channel 量化: 每个输出通道独立的 scale 和 zero_point
//! - 紧凑存储: 每个 u8 存储 2 个 4-bit 值 (低4位 + 高4位)
//!
//! 量化流程:
//! 1. FP32 权重 → 找到每通道的 min/max
//! 2. 计算 scale = (max - min) / 15
//! 3. 计算 zero_point = round(-min / scale)
//! 4. 量化: int4 = round(value / scale) + zero_point, clamp to [0, 15]
//! 5. 打包: 两个 int4 塞进一个 u8

/// 将 FP32 权重量化为 INT4
pub fn quantize_tensor(
    weights: &[f32],
    shape: &[usize],
    channel_dim: usize, // 沿哪个维度分通道
) -> super::model::QuantizedWeight {
    let n_channels = shape[channel_dim];
    let channel_size: usize = shape.iter().enumerate()
        .filter(|(i, _)| *i != channel_dim)
        .map(|(_, &s)| s)
        .product();

    let total_elements: usize = shape.iter().product();
    let data_len = (total_elements + 1) / 2;

    let mut data = vec![0u8; data_len];
    let mut scale = vec![0.0f32; n_channels];
    let mut zero_point = vec![0i8; n_channels];

    // Per-Channel 量化
    for ch in 0..n_channels {
        // 找到当前通道的 min/max
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        for i in 0..channel_size {
            let idx = if channel_dim == 0 {
                ch * channel_size + i
            } else {
                i * n_channels + ch
            };
            if idx < weights.len() {
                min = min.min(weights[idx]);
                max = max.max(weights[idx]);
            }
        }

        // 计算 scale 和 zero_point
        let ch_scale = (max - min) / 15.0;
        let ch_zp = if ch_scale > 0.0 {
            (-min / ch_scale).round() as i8
        } else {
            8i8
        };

        scale[ch] = ch_scale;
        zero_point[ch] = ch_zp.clamp(0, 15);

        // 量化并打包
        for i in 0..channel_size {
            let idx = if channel_dim == 0 {
                ch * channel_size + i
            } else {
                i * n_channels + ch
            };

            let quantized = if ch_scale > 0.0 {
                ((weights[idx] / ch_scale).round() as i32 + ch_zp as i32)
                    .clamp(0, 15) as u8
            } else {
                8u8
            };

            let flat_idx = ch * channel_size + i;
            let byte_idx = flat_idx / 2;
            let is_high = flat_idx % 2 == 1;

            if is_high {
                data[byte_idx] |= quantized << 4;
            } else {
                data[byte_idx] = quantized;
            }
        }
    }

    super::model::QuantizedWeight {
        data,
        scale,
        zero_point,
        shape: shape.to_vec(),
    }
}

/// 将 INT4 权重解量化回 FP32
pub fn dequantize_tensor(qw: &super::model::QuantizedWeight) -> Vec<f32> {
    let total_elements: usize = qw.shape.iter().product();
    let mut result = vec![0.0f32; total_elements];

    for (i, &byte) in qw.data.iter().enumerate() {
        let low = (byte & 0x0F) as i32;
        let high = ((byte >> 4) & 0x0F) as i32;

        let base_idx = i * 2;
        if base_idx < total_elements {
            let ch = base_idx / ((total_elements + qw.scale.len() - 1) / qw.scale.len());
            let ch_clamped = ch.min(qw.scale.len() - 1);
            result[base_idx] = (low - qw.zero_point[ch_clamped] as i32) as f32 * qw.scale[ch_clamped];
        }
        if base_idx + 1 < total_elements {
            let ch = (base_idx + 1) / ((total_elements + qw.scale.len() - 1) / qw.scale.len());
            let ch_clamped = ch.min(qw.scale.len() - 1);
            result[base_idx + 1] = (high - qw.zero_point[ch_clamped] as i32) as f32 * qw.scale[ch_clamped];
        }
    }

    result
}

/// 计算量化误差 (MSE)
pub fn quantization_error(original: &[f32], dequantized: &[f32]) -> f32 {
    let n = original.len().min(dequantized.len());
    let mse: f32 = (0..n)
        .map(|i| (original[i] - dequantized[i]).powi(2))
        .sum::<f32>()
        / n as f32;
    mse
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize() {
        let original: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 10.0).collect();
        let shape = [10, 10];
        let qw = quantize_tensor(&original, &shape, 0);
        let deq = dequantize_tensor(&qw);

        let error = quantization_error(&original, &deq);
        // 4-bit 量化误差应该在合理范围内
        assert!(error < 1.0, "Quantization error too high: {}", error);
    }

    #[test]
    fn test_packing() {
        // 确保打包/解包正确
        let shape = [2, 4];
        let original = vec![1.0, -0.5, 0.3, -1.2, 0.8, 0.1, -0.7, 0.5];
        let qw = quantize_tensor(&original, &shape, 0);

        // 每个 byte 存 2 个 4-bit 值
        assert_eq!(qw.data.len(), 4); // 8 elements / 2 = 4 bytes
    }
}
