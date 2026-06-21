//! 量化支持 — GGUF 模型加载用的量化/反量化模块
//!
//! 本模块实现了与 llama.cpp GGUF 格式兼容的量化类型和反量化算法，包括：
//! - GGML 量化类型常量（F32 / F16 / Q4_0 / Q4_1 / Q5_0 / Q8_0 / Qn_K）
//! - 各量化格式的块结构定义
//! - 将量化数据反量化回 f32 的函数
//! - 辅助函数（块大小、类型字节大小）

use byteorder::{LittleEndian, ReadBytesExt};
use half::f16;
use std::io::Cursor;

// ---------------------------------------------------------------------------
// GGML Quantization Type Constants
// ---------------------------------------------------------------------------

pub const GGML_TYPE_F32: i32 = 0;
pub const GGML_TYPE_F16: i32 = 1;
pub const GGML_TYPE_Q4_0: i32 = 2;
pub const GGML_TYPE_Q4_1: i32 = 3;
pub const GGML_TYPE_Q5_0: i32 = 6;
pub const GGML_TYPE_Q5_1: i32 = 7;
pub const GGML_TYPE_Q8_0: i32 = 8;
pub const GGML_TYPE_Q8_1: i32 = 9;
pub const GGML_TYPE_Q2_K: i32 = 10;
pub const GGML_TYPE_Q3_K: i32 = 11;
pub const GGML_TYPE_Q4_K: i32 = 12;
pub const GGML_TYPE_Q5_K: i32 = 13;
pub const GGML_TYPE_Q6_K: i32 = 14;

/// 从 `u32` 值转换为 GGML 量化类型常量。
///
/// 如果值不对应已知的类型，返回 `None`。
pub fn from_u32(v: u32) -> Option<i32> {
    match v {
        0 => Some(GGML_TYPE_F32),
        1 => Some(GGML_TYPE_F16),
        2 => Some(GGML_TYPE_Q4_0),
        3 => Some(GGML_TYPE_Q4_1),
        6 => Some(GGML_TYPE_Q5_0),
        7 => Some(GGML_TYPE_Q5_1),
        8 => Some(GGML_TYPE_Q8_0),
        9 => Some(GGML_TYPE_Q8_1),
        10 => Some(GGML_TYPE_Q2_K),
        11 => Some(GGML_TYPE_Q3_K),
        12 => Some(GGML_TYPE_Q4_K),
        13 => Some(GGML_TYPE_Q5_K),
        14 => Some(GGML_TYPE_Q6_K),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Block quantization structures
// ---------------------------------------------------------------------------

/// Q4_0 量化块：每个块包含 32 个 4-bit 值。
///
/// 内存布局（18 字节，打包格式）：
/// - 2 字节：f16 缩放因子 `d`
/// - 16 字节：32 个 4-bit 值（每个 nibble 范围 0~15）
///
/// 反量化公式：`value = (nibble - 8) * d`
///
/// 注意：此结构体仅用于文档说明，实际反量化通过字节级读取完成。
#[derive(Clone, Copy, Debug)]
pub struct Q4_0Block {
    pub d: f16,       // 2 bytes
    pub qs: [u8; 16], // 16 bytes = 32 nibbles
}

/// Q4_1 量化块：每个块包含 32 个 4-bit 值 + 最小值。
///
/// 内存布局（20 字节，打包格式）：
/// - 2 字节：f16 缩放因子 `d`
/// - 2 字节：f16 最小值 `m`
/// - 16 字节：32 个 4-bit 值
///
/// 反量化公式：`value = nibble * d + m`
#[derive(Clone, Copy, Debug)]
pub struct Q4_1Block {
    pub d: f16,       // 2 bytes
    pub m: f16,       // 2 bytes
    pub qs: [u8; 16], // 16 bytes = 32 nibbles
}

/// Q5_0 量化块：每个块包含 32 个 5-bit 值。
///
/// 内存布局（22 字节，打包格式）：
/// - 2 字节：f16 缩放因子 `d`
/// - 4 字节：`qh`（u32，每 bit 存储一个值的高位，低 32 bit 各对应一个值）
/// - 16 字节：32 个 4-bit 低位部分
///
/// 反量化公式：`value = ((low_nibble | (high_bit << 4)) - 16) * d`
#[derive(Clone, Copy, Debug)]
pub struct Q5_0Block {
    pub d: f16,        // 2 bytes
    pub qh: u32,       // 4 bytes
    pub qs: [u8; 16],  // 16 bytes = 32 nibbles (low 4 bits)
}

/// Q8_0 量化块：每个块包含 32 个 8-bit 量化值。
///
/// 内存布局（34 字节，打包格式）：
/// - 2 字节：f16 缩放因子 `d`
/// - 32 字节：32 个 i8 值
///
/// 反量化公式：`value = i8_value * d`
#[derive(Clone, Copy, Debug)]
pub struct Q8_0Block {
    pub d: f16,        // 2 bytes
    pub qs: [i8; 32],  // 32 bytes
}

// ---------------------------------------------------------------------------
// Dequantization functions
// ---------------------------------------------------------------------------

/// 将 Q4_0 量化数据反量化为 f32。
///
/// `data` 为原始字节流，`count` 为需要反量化的 f32 值数量。
/// 每 18 字节对应 32 个 f32 值。
pub fn dequantize_q4_0(data: &[u8], count: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(count);
    let block_size = 32usize;
    let num_blocks = (count + block_size - 1) / block_size;
    let mut offset = 0usize;

    for _ in 0..num_blocks {
        if offset + 18 > data.len() {
            break;
        }
        // 读取 f16 缩放因子
        let d_val = f16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let d = f32::from(d_val);
        offset += 2;

        // 读取 16 字节 = 32 个 nibble
        for i in 0..32 {
            if result.len() >= count {
                break;
            }
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                data[offset + byte_idx] >> 4
            } else {
                data[offset + byte_idx] & 0x0F
            };
            // value = (nibble - 8) * d
            let value = (nibble as i32 - 8) as f32 * d;
            result.push(value);
        }
        offset += 16;
    }

    result
}

/// 将 Q4_1 量化数据反量化为 f32。
///
/// `data` 为原始字节流，`count` 为需要反量化的 f32 值数量。
/// 每 20 字节对应 32 个 f32 值。
pub fn dequantize_q4_1(data: &[u8], count: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(count);
    let block_size = 32usize;
    let num_blocks = (count + block_size - 1) / block_size;
    let mut offset = 0usize;

    for _ in 0..num_blocks {
        if offset + 20 > data.len() {
            break;
        }
        // 读取 f16 缩放因子
        let d_val = f16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let d = f32::from(d_val);
        offset += 2;

        // 读取 f16 最小值
        let m_val = f16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let m = f32::from(m_val);
        offset += 2;

        // 读取 16 字节 = 32 个 nibble
        for i in 0..32 {
            if result.len() >= count {
                break;
            }
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                data[offset + byte_idx] >> 4
            } else {
                data[offset + byte_idx] & 0x0F
            };
            // value = nibble * d + m
            let value = nibble as f32 * d + m;
            result.push(value);
        }
        offset += 16;
    }

    result
}

/// 将 Q5_0 量化数据反量化为 f32。
///
/// `data` 为原始字节流，`count` 为需要反量化的 f32 值数量。
/// 每 22 字节对应 32 个 f32 值。
pub fn dequantize_q5_0(data: &[u8], count: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(count);
    let block_size = 32usize;
    let num_blocks = (count + block_size - 1) / block_size;
    let mut offset = 0usize;

    for _ in 0..num_blocks {
        if offset + 22 > data.len() {
            break;
        }
        // 读取 f16 缩放因子
        let d_val = f16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let d = f32::from(d_val);
        offset += 2;

        // 读取 u32 qh（高位 bits）
        let qh = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        // 读取 16 字节 = 32 个 nibble（低位 4 bits）
        for i in 0..32 {
            if result.len() >= count {
                break;
            }
            let byte_idx = i / 2;
            let low_nibble = if i % 2 == 0 {
                data[offset + byte_idx] >> 4
            } else {
                data[offset + byte_idx] & 0x0F
            };
            // 从 qh 取第 i 个高位 bit
            let high_bit = (qh >> i) & 1;
            // 组合为 5-bit 值，然后减去 16 偏移
            let val_5bit = (low_nibble as u32 | (high_bit << 4)) as i32 - 16;
            // value = val_5bit * d
            let value = val_5bit as f32 * d;
            result.push(value);
        }
        offset += 16;
    }

    result
}

/// 将 Q8_0 量化数据反量化为 f32。
///
/// `data` 为原始字节流，`count` 为需要反量化的 f32 值数量。
/// 每 34 字节对应 32 个 f32 值。
pub fn dequantize_q8_0(data: &[u8], count: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(count);
    let block_size = 32usize;
    let num_blocks = (count + block_size - 1) / block_size;
    let mut offset = 0usize;

    for _ in 0..num_blocks {
        if offset + 34 > data.len() {
            break;
        }
        // 读取 f16 缩放因子
        let d_val = f16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let d = f32::from(d_val);
        offset += 2;

        // 读取 32 个 i8 值
        for i in 0..32 {
            if result.len() >= count {
                break;
            }
            let qi = data[offset + i] as i8;
            // value = i8 * d
            let value = qi as f32 * d;
            result.push(value);
        }
        offset += 32;
    }

    result
}

/// 将 f16 数据反量化为 f32。
///
/// 每个值占 2 字节，直接转换为 f32。
pub fn dequantize_f16(data: &[u8], count: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(count);
    let mut cursor = Cursor::new(data);
    for _ in 0..count {
        if let Ok(bits) = cursor.read_u16::<LittleEndian>() {
            let val = f16::from_bits(bits);
            result.push(f32::from(val));
        } else {
            break;
        }
    }
    result
}

/// 反量化调度器：根据 `ggml_type` 调用对应的反量化函数。
///
/// 返回包含 `count` 个 f32 值的向量，或者在不支持的类型时返回错误字符串。
pub fn dequantize(data: &[u8], ggml_type: i32, count: usize) -> Result<Vec<f32>, String> {
    match ggml_type {
        GGML_TYPE_F32 => {
            if data.len() < count * 4 {
                return Err("Not enough data for F32".into());
            }
            let mut result = Vec::with_capacity(count);
            let mut cursor = Cursor::new(data);
            for _ in 0..count {
                match cursor.read_f32::<LittleEndian>() {
                    Ok(v) => result.push(v),
                    Err(_) => break,
                }
            }
            Ok(result)
        }
        GGML_TYPE_F16 => Ok(dequantize_f16(data, count)),
        GGML_TYPE_Q4_0 => Ok(dequantize_q4_0(data, count)),
        GGML_TYPE_Q4_1 => Ok(dequantize_q4_1(data, count)),
        GGML_TYPE_Q5_0 => Ok(dequantize_q5_0(data, count)),
        GGML_TYPE_Q8_0 => Ok(dequantize_q8_0(data, count)),
        _ => Err(format!(
            "Unsupported ggml type for dequantization: {}",
            ggml_type
        )),
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// 返回每种量化类型的块大小（每个块包含的量化值数量）。
pub fn block_size(ggml_type: i32) -> usize {
    match ggml_type {
        GGML_TYPE_F32 => 1,
        GGML_TYPE_F16 => 1,
        GGML_TYPE_Q4_0 => 32,
        GGML_TYPE_Q4_1 => 32,
        GGML_TYPE_Q5_0 => 32,
        GGML_TYPE_Q5_1 => 32,
        GGML_TYPE_Q8_0 => 32,
        GGML_TYPE_Q8_1 => 32,
        GGML_TYPE_Q2_K => 256,
        GGML_TYPE_Q3_K => 256,
        GGML_TYPE_Q4_K => 256,
        GGML_TYPE_Q5_K => 256,
        GGML_TYPE_Q6_K => 256,
        _ => 1,
    }
}

/// 返回每种量化类型每个块占用的字节数。
pub fn type_size(ggml_type: i32) -> usize {
    match ggml_type {
        GGML_TYPE_F32 => 4,
        GGML_TYPE_F16 => 2,
        GGML_TYPE_Q4_0 => 18,   // 2 (f16 d) + 16 (32 nibbles)
        GGML_TYPE_Q4_1 => 20,   // 2 (f16 d) + 2 (f16 m) + 16 (32 nibbles)
        GGML_TYPE_Q5_0 => 22,   // 2 (f16 d) + 4 (u32 qh) + 16 (32 low nibbles)
        GGML_TYPE_Q5_1 => 24,   // 2 (f16 d) + 2 (f16 m) + 4 (u32 qh) + 16 (32 low nibbles)
        GGML_TYPE_Q8_0 => 34,   // 2 (f16 d) + 32 (32 x i8)
        GGML_TYPE_Q8_1 => 40,   // 2 (f16 d) + 2 (f16 s) + 32 (32 x i8) + 4 (sums)
        GGML_TYPE_Q2_K => 84,   // llama.cpp ggml_type_size
        GGML_TYPE_Q3_K => 110,
        GGML_TYPE_Q4_K => 144,
        GGML_TYPE_Q5_K => 176,
        GGML_TYPE_Q6_K => 210,
        _ => 4,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 辅助函数：将 f32 向量量化为 Q4_0 格式的字节流（用于 round-trip 测试）。
    fn quantize_q4_0(values: &[f32]) -> Vec<u8> {
        let block_size = 32usize;
        let num_blocks = (values.len() + block_size - 1) / block_size;
        let mut bytes = Vec::with_capacity(num_blocks * 18);

        for b in 0..num_blocks {
            let start = b * block_size;
            let end = std::cmp::min(start + block_size, values.len());
            let block_vals = &values[start..end];

            // 计算 scale: max abs value / 8
            let max_abs = block_vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = if max_abs == 0.0 { 1.0f32 } else { max_abs / 8.0 };

            // 写入 f16 d
            let d_f16 = f16::from_f32(d);
            let d_bits = d_f16.to_bits();
            bytes.extend_from_slice(&d_bits.to_le_bytes());

            // 量化每个值为 nibble
            let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };
            let mut nibbles = Vec::with_capacity(block_size);
            for &v in block_vals.iter() {
                let scaled = (v * inv_d + 8.0).round() as i32;
                let clamped = std::cmp::max(0, std::cmp::min(15, scaled)) as u8;
                nibbles.push(clamped);
            }

            // 填充到 32 个
            while nibbles.len() < 32 {
                nibbles.push(8); // 零值对应的 nibble
            }

            // 打包为 16 字节
            for i in (0..32).step_by(2) {
                bytes.push((nibbles[i] << 4) | nibbles[i + 1]);
            }
        }

        bytes
    }

    /// 辅助函数：将 f32 向量量化为 Q8_0 格式的字节流。
    fn quantize_q8_0(values: &[f32]) -> Vec<u8> {
        let block_size = 32usize;
        let num_blocks = (values.len() + block_size - 1) / block_size;
        let mut bytes = Vec::with_capacity(num_blocks * 34);

        for b in 0..num_blocks {
            let start = b * block_size;
            let end = std::cmp::min(start + block_size, values.len());
            let block_vals = &values[start..end];

            // 计算 scale: max abs value / 127
            let max_abs = block_vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = if max_abs == 0.0 { 1.0f32 } else { max_abs / 127.0 };

            // 写入 f16 d
            let d_f16 = f16::from_f32(d);
            let d_bits = d_f16.to_bits();
            bytes.extend_from_slice(&d_bits.to_le_bytes());

            // 量化每个值为 i8
            let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };
            for &v in block_vals.iter() {
                let scaled = (v * inv_d).round() as i16;
                let clamped = std::cmp::max(-128, std::cmp::min(127, scaled)) as i8;
                bytes.push(clamped as u8);
            }

            // 填充到 32 字节
            let padding = 32 - (end - start);
            for _ in 0..padding {
                bytes.push(0);
            }
        }

        bytes
    }

    #[test]
    fn test_q4_0_roundtrip() {
        // 用已知向量做 round-trip 测试
        let values: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 3.0).collect();
        let quantized = quantize_q4_0(&values);
        let dequantized = dequantize_q4_0(&quantized, 64);

        assert_eq!(dequantized.len(), 64);
        for (orig, &deq) in values.iter().zip(dequantized.iter()) {
            // 4-bit 量化精度有限，最大量化步长 = max_abs / 8 ≈ 0.41
            assert!(
                (orig - deq).abs() < 0.5,
                "Q4_0 roundtrip: expected {}, got {}, diff {}",
                orig,
                deq,
                (orig - deq).abs()
            );
        }
    }

    #[test]
    fn test_q8_0_roundtrip() {
        let values: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 3.0).collect();
        let quantized = quantize_q8_0(&values);
        let dequantized = dequantize_q8_0(&quantized, 64);

        assert_eq!(dequantized.len(), 64);
        for (orig, &deq) in values.iter().zip(dequantized.iter()) {
            // 8-bit 量化精度更高，误差应更小
            assert!(
                (orig - deq).abs() < 0.05,
                "Q8_0 roundtrip: expected {}, got {}, diff {}",
                orig,
                deq,
                (orig - deq).abs()
            );
        }
    }

    #[test]
    fn test_f16_dequantize() {
        // 测试几个已知 f16 → f32 转换
        let val_f32 = 1.5f32;
        let val_f16 = f16::from_f32(val_f32);
        let bits = val_f16.to_bits();
        let bytes = bits.to_le_bytes();
        let result = dequantize_f16(&bytes, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - val_f32).abs() < 1e-3);

        // 测试零
        let zero_f16 = f16::from_f32(0.0);
        let zero_bytes = zero_f16.to_bits().to_le_bytes();
        let zero_result = dequantize_f16(&zero_bytes, 1);
        assert!((zero_result[0] - 0.0).abs() < 1e-6);

        // 测试负数
        let neg_f32 = -3.14f32;
        let neg_f16 = f16::from_f32(neg_f32);
        let neg_bytes = neg_f16.to_bits().to_le_bytes();
        let neg_result = dequantize_f16(&neg_bytes, 1);
        assert!((neg_result[0] - neg_f32).abs() < 0.01);
    }

    #[test]
    fn test_block_size() {
        assert_eq!(block_size(GGML_TYPE_F32), 1);
        assert_eq!(block_size(GGML_TYPE_F16), 1);
        assert_eq!(block_size(GGML_TYPE_Q4_0), 32);
        assert_eq!(block_size(GGML_TYPE_Q4_1), 32);
        assert_eq!(block_size(GGML_TYPE_Q5_0), 32);
        assert_eq!(block_size(GGML_TYPE_Q8_0), 32);
        assert_eq!(block_size(GGML_TYPE_Q2_K), 256);
        assert_eq!(block_size(GGML_TYPE_Q3_K), 256);
        assert_eq!(block_size(GGML_TYPE_Q4_K), 256);
        assert_eq!(block_size(GGML_TYPE_Q5_K), 256);
        assert_eq!(block_size(GGML_TYPE_Q6_K), 256);
    }

    #[test]
    fn test_type_size() {
        assert_eq!(type_size(GGML_TYPE_F32), 4);
        assert_eq!(type_size(GGML_TYPE_F16), 2);
        assert_eq!(type_size(GGML_TYPE_Q4_0), 18);
        assert_eq!(type_size(GGML_TYPE_Q4_1), 20);
        assert_eq!(type_size(GGML_TYPE_Q5_0), 22);
        assert_eq!(type_size(GGML_TYPE_Q8_0), 34);
        assert_eq!(type_size(GGML_TYPE_Q2_K), 84);
        assert_eq!(type_size(GGML_TYPE_Q3_K), 110);
        assert_eq!(type_size(GGML_TYPE_Q4_K), 144);
        assert_eq!(type_size(GGML_TYPE_Q5_K), 176);
        assert_eq!(type_size(GGML_TYPE_Q6_K), 210);
    }

    #[test]
    fn test_dequantize_dispatcher() {
        // F32 调度
        let f32_val = 3.14f32;
        let mut f32_bytes = Vec::new();
        f32_bytes.extend_from_slice(&f32_val.to_le_bytes());
        let result = dequantize(f32_bytes.as_slice(), GGML_TYPE_F32, 1).unwrap();
        assert!((result[0] - f32_val).abs() < 1e-6);

        // F16 调度
        let f16_val = f16::from_f32(2.71f32);
        let f16_bytes = f16_val.to_bits().to_le_bytes();
        let result = dequantize(f16_bytes.as_slice(), GGML_TYPE_F16, 1).unwrap();
        assert!((result[0] - 2.71f32).abs() < 0.01);

        // Q4_0 调度（使用 round-trip 数据）
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.5 - 8.0).collect();
        let quantized = quantize_q4_0(&values);
        let result = dequantize(quantized.as_slice(), GGML_TYPE_Q4_0, 32).unwrap();
        assert_eq!(result.len(), 32);
        for (orig, &deq) in values.iter().zip(result.iter()) {
            assert!(
                (orig - deq).abs() <= 0.5,
                "dispatcher Q4_0: orig={}, deq={}, diff={}",
                orig,
                deq,
                (orig - deq).abs()
            );
        }

        // 不支持的类型
        let err = dequantize(&[], GGML_TYPE_Q4_K, 10);
        assert!(err.is_err());
    }

    #[test]
    fn test_from_u32_conversion() {
        assert_eq!(from_u32(0), Some(GGML_TYPE_F32));
        assert_eq!(from_u32(1), Some(GGML_TYPE_F16));
        assert_eq!(from_u32(2), Some(GGML_TYPE_Q4_0));
        assert_eq!(from_u32(3), Some(GGML_TYPE_Q4_1));
        assert_eq!(from_u32(6), Some(GGML_TYPE_Q5_0));
        assert_eq!(from_u32(7), Some(GGML_TYPE_Q5_1));
        assert_eq!(from_u32(8), Some(GGML_TYPE_Q8_0));
        assert_eq!(from_u32(9), Some(GGML_TYPE_Q8_1));
        assert_eq!(from_u32(10), Some(GGML_TYPE_Q2_K));
        assert_eq!(from_u32(11), Some(GGML_TYPE_Q3_K));
        assert_eq!(from_u32(12), Some(GGML_TYPE_Q4_K));
        assert_eq!(from_u32(13), Some(GGML_TYPE_Q5_K));
        assert_eq!(from_u32(14), Some(GGML_TYPE_Q6_K));
        assert_eq!(from_u32(99), None);
    }

    #[test]
    fn test_q4_1_basic() {
        // 构造一个简单的 Q4_1 块：d=1.0, m=0.0, 所有 nibble=5
        let mut data = Vec::new();
        // d = 1.0
        let d_f16 = f16::from_f32(1.0);
        data.extend_from_slice(&d_f16.to_bits().to_le_bytes());
        // m = 0.0
        let m_f16 = f16::from_f32(0.0);
        data.extend_from_slice(&m_f16.to_bits().to_le_bytes());
        // 32 个 nibble 全为 5：每个字节高 nibble=5, 低 nibble=5 → 0x55
        for _ in 0..16 {
            data.push(0x55);
        }

        let result = dequantize_q4_1(&data, 32);
        assert_eq!(result.len(), 32);
        for &v in &result {
            // value = 5 * 1.0 + 0.0 = 5.0
            assert!((v - 5.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_q5_0_basic() {
        // 构造一个简单的 Q5_0 块：d=1.0, 所有 5-bit 值=16（即 0）
        // low nibble=0, high bit=1 → 0 | (1<<4) = 16, 16-16=0
        let mut data = Vec::new();
        // d = 1.0
        let d_f16 = f16::from_f32(1.0);
        data.extend_from_slice(&d_f16.to_bits().to_le_bytes());
        // qh = 0xFFFFFFFF（所有高位 bit=1）
        data.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        // qs: 所有 low nibble=0 → 0x00
        for _ in 0..16 {
            data.push(0x00);
        }

        let result = dequantize_q5_0(&data, 32);
        assert_eq!(result.len(), 32);
        for &v in &result {
            // value = (0 | 16) - 16 = 0, 0 * 1.0 = 0.0
            assert!(v.abs() < 1e-5);
        }
    }

    #[test]
    fn test_q8_0_basic() {
        // 构造一个 Q8_0 块：d=0.5, 所有 i8=10
        let mut data = Vec::new();
        // d = 0.5
        let d_f16 = f16::from_f32(0.5);
        data.extend_from_slice(&d_f16.to_bits().to_le_bytes());
        // 32 个 i8 = 10
        for _ in 0..32 {
            data.push(10u8);
        }

        let result = dequantize_q8_0(&data, 32);
        assert_eq!(result.len(), 32);
        for &v in &result {
            // value = 10 * 0.5 = 5.0
            assert!((v - 5.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_block_struct_sizes() {
        // 验证 type_size 函数返回的打包格式字节大小
        // 注意：Rust struct 可能因对齐有额外填充，因此我们验证 type_size()
        // 而非 std::mem::size_of。打包格式中 Q5_0 为 22 字节，
        // 但 Rust 默认对齐会导致 Q5_0Block 大于 22。
        assert_eq!(type_size(GGML_TYPE_Q4_0), 18);
        assert_eq!(type_size(GGML_TYPE_Q4_1), 20);
        assert_eq!(type_size(GGML_TYPE_Q5_0), 22);
        assert_eq!(type_size(GGML_TYPE_Q8_0), 34);
    }
}