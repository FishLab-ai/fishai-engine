//! GGUF 文件格式解析器
//!
//! 本模块实现了 GGUF（GPT-Generated Unified Format）文件格式的完整解析功能。
//! GGUF 是 llama.cpp / ggml 生态使用的模型文件格式，支持高效的内存映射加载。
//!
//! 主要功能：
//! - 解析 GGUF 文件头（魔数、版本、张量数量、元数据数量）
//! - 读取键值对元数据（支持全部 13 种值类型：UINT8 ~ FLOAT64、STRING、ARRAY）
//! - 解析张量信息（名称、维度、GGML 类型、偏移量）
//! - 通过内存映射（mmap）高效读取张量原始数据
//! - 支持反量化读取（调用 crate::quant::dequantize 转为 f32）
//! - 提取模型架构信息（上下文长度、嵌入维度、注意力头数、层数、词表大小等）
//!
//! # 使用示例
//!
//! ```no_run
//! use fishai_engine::gguf::GGUFFile;
//!
//! let file = GGUFFile::open("model.gguf").unwrap();
//! println!("架构: {:?}", file.model_architecture());
//! println!("上下文长度: {:?}", file.context_length());
//! println!("张量数量: {}", file.tensor_count());
//! ```

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------

/// GGUF 文件魔数：`"GGUF"` 的小端序 u32 表示（0x46465547）
pub const GGUF_MAGIC: u32 = 0x46465547;

/// GGUF 文件格式版本 V3（当前唯一广泛使用的稳定版本）
pub const GGUF_VERSION_V3: u32 = 3;

// ---------------------------------------------------------------------------
// GGUFValueType
// ---------------------------------------------------------------------------

/// GGUF 元数据值类型枚举，覆盖 GGUF 规范定义的全部 13 种类型。
///
/// | 值 | 类型   | 说明         |
/// |----|--------|-------------|
/// | 0  | UINT8  | 无符号 8 位  |
/// | 1  | INT8   | 有符号 8 位  |
/// | 2  | UINT16 | 无符号 16 位 |
/// | 3  | INT16  | 有符号 16 位 |
/// | 4  | UINT32 | 无符号 32 位 |
/// | 5  | INT32  | 有符号 32 位 |
/// | 6  | FLOAT32| 32 位浮点    |
/// | 7  | BOOL   | 布尔值       |
/// | 8  | STRING | GGUF 字符串  |
/// | 9  | ARRAY  | 类型化数组   |
/// | 10 | UINT64 | 无符号 64 位 |
/// | 11 | INT64  | 有符号 64 位 |
/// | 12 | FLOAT64| 64 位浮点    |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GGUFValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GGUFValueType {
    /// 将 u32 数值转换为 `GGUFValueType`。
    ///
    /// 对于 GGUF 规范中未定义的类型 ID 返回 `None`。
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

impl std::fmt::Display for GGUFValueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uint8 => write!(f, "UINT8"),
            Self::Int8 => write!(f, "INT8"),
            Self::Uint16 => write!(f, "UINT16"),
            Self::Int16 => write!(f, "INT16"),
            Self::Uint32 => write!(f, "UINT32"),
            Self::Int32 => write!(f, "INT32"),
            Self::Float32 => write!(f, "FLOAT32"),
            Self::Bool => write!(f, "BOOL"),
            Self::String => write!(f, "STRING"),
            Self::Array => write!(f, "ARRAY"),
            Self::Uint64 => write!(f, "UINT64"),
            Self::Int64 => write!(f, "INT64"),
            Self::Float64 => write!(f, "FLOAT64"),
        }
    }
}

// ---------------------------------------------------------------------------
// GGUFString
// ---------------------------------------------------------------------------

/// GGUF 字符串结构体，由 u64 长度前缀和 UTF-8 字节数据组成。
#[derive(Debug, Clone)]
pub struct GGUFString {
    /// 字符串字节长度
    pub len: u64,
    /// 原始字节数据（通常为 UTF-8 编码）
    pub data: Vec<u8>,
}

impl GGUFString {
    /// 尝试将字节数据作为 UTF-8 解析。
    ///
    /// 如果字节包含无效 UTF-8 序列，返回 `None`。
    pub fn as_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.data).ok()
    }

    /// 将字节数据转换为 `String`，无效 UTF-8 字节替换为 `U+FFFD`（）。
    pub fn to_string_lossy(&self) -> String {
        String::from_utf8_lossy(&self.data).into_owned()
    }
}

impl PartialEq for GGUFString {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for GGUFString {}

// ---------------------------------------------------------------------------
// GGUFMetadataValue / GGUFArray
// ---------------------------------------------------------------------------

/// GGUF 元数据值枚举，每个变体对应一种 GGUF 值类型。
#[derive(Debug, Clone)]
pub enum GGUFMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(GGUFArray),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl PartialEq for GGUFMetadataValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Uint8(a), Self::Uint8(b)) => a == b,
            (Self::Int8(a), Self::Int8(b)) => a == b,
            (Self::Uint16(a), Self::Uint16(b)) => a == b,
            (Self::Int16(a), Self::Int16(b)) => a == b,
            (Self::Uint32(a), Self::Uint32(b)) => a == b,
            (Self::Int32(a), Self::Int32(b)) => a == b,
            (Self::Float32(a), Self::Float32(b)) => a.to_bits() == b.to_bits(),
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::String(a), Self::String(b)) => a == b,
            (Self::Array(a), Self::Array(b)) => {
                a.item_type == b.item_type && a.values == b.values
            }
            (Self::Uint64(a), Self::Uint64(b)) => a == b,
            (Self::Int64(a), Self::Int64(b)) => a == b,
            (Self::Float64(a), Self::Float64(b)) => a.to_bits() == b.to_bits(),
            _ => false,
        }
    }
}

impl Eq for GGUFMetadataValue {}

/// GGUF 类型化数组，所有元素共享同一个值类型。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GGUFArray {
    /// 数组元素的值类型
    pub item_type: GGUFValueType,
    /// 数组元素列表
    pub values: Vec<GGUFMetadataValue>,
}

// ---------------------------------------------------------------------------
// TensorInfo
// ---------------------------------------------------------------------------

/// 张量描述信息，记录张量名称、形状、数据类型及在数据段中的偏移。
///
/// 注意：`offset` 是相对于数据段起始位置（对齐后的偏移）的值，而非文件绝对偏移。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// 张量名称（如 `"token_embd.weight"`）
    pub name: String,
    /// 维度数量（最大为 4）
    pub n_dims: u32,
    /// 各维度大小（长度等于 `n_dims`，通常按行优先存储）
    pub dims: Vec<u64>,
    /// GGML 数据类型 ID（如 0 = F32, 2 = Q4_0, 8 = Q8_0 等）
    pub ggml_type: u32,
    /// 相对于数据段起始位置的偏移量（字节）
    pub offset: u64,
}

impl TensorInfo {
    /// 计算张量元素总数，即各维度大小的乘积。
    ///
    /// 对于标量（0 维张量），返回 1。
    pub fn nelement(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    /// 返回张量形状，将各维度大小转为 `usize`。
    pub fn shape(&self) -> Vec<usize> {
        self.dims.iter().map(|&d| d as usize).collect()
    }
}

// ---------------------------------------------------------------------------
// 内部辅助函数
// ---------------------------------------------------------------------------

/// 将 `n` 向上对齐到 `alignment` 的整数倍。
///
/// 如果 `alignment` 为 0，直接返回 `n`。
#[inline]
fn align_to(n: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return n;
    }
    let mask = alignment - 1;
    (n + mask) & !mask
}

/// 从游标中读取一个 GGUF 字符串（u64 长度前缀 + 字节数据）。
fn read_string(cursor: &mut impl ReadBytesExt) -> Result<GGUFString, String> {
    let len = cursor
        .read_u64::<LittleEndian>()
        .map_err(|e| format!("读取字符串长度失败: {}", e))?;
    let len_usize = len as usize;

    // 防止恶意或损坏文件导致超大分配
    if len_usize > 256 * 1024 * 1024 {
        return Err(format!("字符串长度过大: {} 字节", len));
    }

    let mut data = vec![0u8; len_usize];
    cursor
        .read_exact(&mut data)
        .map_err(|e| format!("读取字符串数据失败（期望 {} 字节）: {}", len, e))?;
    Ok(GGUFString { len, data })
}

/// 根据值类型从游标中读取一个元数据值。
///
/// 对于 `Array` 类型会递归读取所有元素。
fn read_value_at(
    cursor: &mut impl ReadBytesExt,
    vtype: &GGUFValueType,
) -> Result<GGUFMetadataValue, String> {
    use GGUFValueType::*;

    match vtype {
        Uint8 => {
            let v = cursor
                .read_u8()
                .map_err(|e| format!("读取 UINT8 失败: {}", e))?;
            Ok(GGUFMetadataValue::Uint8(v))
        }
        Int8 => {
            let v = cursor
                .read_i8()
                .map_err(|e| format!("读取 INT8 失败: {}", e))?;
            Ok(GGUFMetadataValue::Int8(v))
        }
        Uint16 => {
            let v = cursor
                .read_u16::<LittleEndian>()
                .map_err(|e| format!("读取 UINT16 失败: {}", e))?;
            Ok(GGUFMetadataValue::Uint16(v))
        }
        Int16 => {
            let v = cursor
                .read_i16::<LittleEndian>()
                .map_err(|e| format!("读取 INT16 失败: {}", e))?;
            Ok(GGUFMetadataValue::Int16(v))
        }
        Uint32 => {
            let v = cursor
                .read_u32::<LittleEndian>()
                .map_err(|e| format!("读取 UINT32 失败: {}", e))?;
            Ok(GGUFMetadataValue::Uint32(v))
        }
        Int32 => {
            let v = cursor
                .read_i32::<LittleEndian>()
                .map_err(|e| format!("读取 INT32 失败: {}", e))?;
            Ok(GGUFMetadataValue::Int32(v))
        }
        Float32 => {
            let v = cursor
                .read_f32::<LittleEndian>()
                .map_err(|e| format!("读取 FLOAT32 失败: {}", e))?;
            Ok(GGUFMetadataValue::Float32(v))
        }
        Bool => {
            let v = cursor
                .read_u8()
                .map_err(|e| format!("读取 BOOL 失败: {}", e))?;
            Ok(GGUFMetadataValue::Bool(v != 0))
        }
        String => {
            let s = read_string(cursor)?;
            Ok(GGUFMetadataValue::String(s.to_string_lossy()))
        }
        Array => {
            let item_type_u32 = cursor
                .read_u32::<LittleEndian>()
                .map_err(|e| format!("读取数组元素类型失败: {}", e))?;
            let item_type = GGUFValueType::from_u32(item_type_u32)
                .ok_or(format!("未知的数组元素类型: {}", item_type_u32))?;
            let count = cursor
                .read_u64::<LittleEndian>()
                .map_err(|e| format!("读取数组长度失败: {}", e))?;

            // 防止超大数组
            if count as usize > 10_000_000 {
                return Err(format!("数组元素数量过大: {}", count));
            }

            let mut values = Vec::with_capacity(count as usize);
            for _ in 0..count {
                values.push(read_value_at(cursor, &item_type)?);
            }
            Ok(GGUFMetadataValue::Array(GGUFArray {
                item_type,
                values,
            }))
        }
        Uint64 => {
            let v = cursor
                .read_u64::<LittleEndian>()
                .map_err(|e| format!("读取 UINT64 失败: {}", e))?;
            Ok(GGUFMetadataValue::Uint64(v))
        }
        Int64 => {
            let v = cursor
                .read_i64::<LittleEndian>()
                .map_err(|e| format!("读取 INT64 失败: {}", e))?;
            Ok(GGUFMetadataValue::Int64(v))
        }
        Float64 => {
            let v = cursor
                .read_f64::<LittleEndian>()
                .map_err(|e| format!("读取 FLOAT64 失败: {}", e))?;
            Ok(GGUFMetadataValue::Float64(v))
        }
    }
}

// ---------------------------------------------------------------------------
// GGUFFile
// ---------------------------------------------------------------------------

/// GGUF 文件解析结果，包含完整的模型元数据和张量索引。
///
/// 通过 [`GGUFFile::open`] 以内存映射方式加载文件，所有张量数据通过 mmap
/// 直接访问，无需额外拷贝。
///
/// # 生命周期
///
/// `GGUFFile` 持有 `memmap2::Mmap` 句柄，确保映射在文件对象存活期间有效。
#[derive(Debug)]
pub struct GGUFFile {
    /// GGUF 文件版本号
    pub version: u32,
    /// 元数据键值对映射
    pub metadata: HashMap<String, GGUFMetadataValue>,
    /// 张量信息列表
    pub tensor_infos: Vec<TensorInfo>,
    /// 数据段对齐值（默认 32 字节）
    pub alignment: u64,
    /// 内存映射句柄（保持映射区域存活）
    _mmap: Option<Mmap>,
    /// 数据段在文件中的起始偏移（字节）
    data_offset: usize,
    /// 文件路径（用于诊断信息）
    #[allow(dead_code)]
    file_path: Option<String>,
}

impl GGUFFile {
    /// 打开并解析 GGUF 文件。
    ///
    /// 使用 `memmap2` 对文件进行内存映射，然后依次解析：
    /// 1. 文件头（魔数、版本、张量数量、元数据数量）
    /// 2. 元数据键值对
    /// 3. 张量描述信息
    /// 4. 计算对齐后的数据段偏移
    ///
    /// # 错误
    ///
    /// - 文件不存在或无法打开
    /// - 内存映射失败
    /// - 魔数不匹配（非 GGUF 文件）
    /// - 版本号不支持
    /// - 数据格式损坏
    pub fn open(path: &str) -> Result<Self, String> {
        let file = File::open(path)
            .map_err(|e| format!("无法打开文件 '{}': {}", path, e))?;

        // Safety: 文件以只读方式打开，仅从中读取数据，不会产生数据竞争
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("无法映射文件 '{}': {}", path, e))?;

        let mut cursor = Cursor::new(mmap.as_ref());

        // ====== 解析文件头 ======
        let magic = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("读取魔数失败: {}", e))?;
        if magic != GGUF_MAGIC {
            return Err(format!(
                "无效的 GGUF 魔数: 期望 0x{:08X}, 实际 0x{:08X}",
                GGUF_MAGIC, magic
            ));
        }

        let version = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("读取版本号失败: {}", e))?;
        if version > GGUF_VERSION_V3 {
            return Err(format!(
                "不支持的 GGUF 版本: {}（当前最大支持版本 {}）",
                version, GGUF_VERSION_V3
            ));
        }

        let tensor_count = cursor
            .read_u64::<LittleEndian>()
            .map_err(|e| format!("读取张量数量失败: {}", e))?;
        let metadata_kv_count = cursor
            .read_u64::<LittleEndian>()
            .map_err(|e| format!("读取元数据键值对数量失败: {}", e))?;

        // 防止恶意文件声称拥有极大的张量/元数据数量
        if tensor_count > 1_000_000 {
            return Err(format!("张量数量异常: {}", tensor_count));
        }
        if metadata_kv_count > 10_000_000 {
            return Err(format!("元数据键值对数量异常: {}", metadata_kv_count));
        }

        // ====== 解析元数据键值对 ======
        let mut metadata = HashMap::with_capacity(metadata_kv_count as usize);
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut cursor)?;
            let vtype_u32 = cursor
                .read_u32::<LittleEndian>()
                .map_err(|e| format!("读取值类型失败: {}", e))?;
            let vtype = GGUFValueType::from_u32(vtype_u32)
                .ok_or(format!("未知的值类型: {}", vtype_u32))?;
            let value = read_value_at(&mut cursor, &vtype)?;
            let key_str = key.to_string_lossy();
            metadata.insert(key_str, value);
        }

        // ====== 获取对齐值 ======
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| match v {
                GGUFMetadataValue::Uint64(v) => Some(*v),
                GGUFMetadataValue::Uint32(v) => Some(*v as u64),
                _ => None,
            })
            .unwrap_or(32);

        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(format!("无效的对齐值: {}（必须为 2 的幂）", alignment));
        }

        // ====== 解析张量信息 ======
        let mut tensor_infos = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name_gguf = read_string(&mut cursor)?;
            let n_dims = cursor
                .read_u32::<LittleEndian>()
                .map_err(|e| format!("读取张量维度数失败: {}", e))?;

            if n_dims > 4 {
                return Err(format!("张量维度数 {} 超过最大值 4", n_dims));
            }

            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let d = cursor
                    .read_u64::<LittleEndian>()
                    .map_err(|e| format!("读取张量维度大小失败: {}", e))?;
                dims.push(d);
            }

            let ggml_type = cursor
                .read_u32::<LittleEndian>()
                .map_err(|e| format!("读取张量 GGML 类型失败: {}", e))?;
            let offset = cursor
                .read_u64::<LittleEndian>()
                .map_err(|e| format!("读取张量偏移量失败: {}", e))?;

            tensor_infos.push(TensorInfo {
                name: name_gguf.to_string_lossy(),
                n_dims,
                dims,
                ggml_type,
                offset,
            });
        }

        // ====== 计算数据段对齐偏移 ======
        let pos = cursor.position() as usize;
        let data_offset = align_to(pos, alignment as usize);

        Ok(GGUFFile {
            version,
            metadata,
            tensor_infos,
            alignment,
            _mmap: Some(mmap),
            data_offset,
            file_path: Some(path.to_string()),
        })
    }

    // -----------------------------------------------------------------------
    // 元数据访问器
    // -----------------------------------------------------------------------

    /// 获取字符串类型的元数据值。
    ///
    /// 如果键不存在或值类型不是 `String`，返回 `None`。
    pub fn metadata_string(&self, key: &str) -> Option<&String> {
        self.metadata.get(key).and_then(|v| match v {
            GGUFMetadataValue::String(s) => Some(s),
            _ => None,
        })
    }

    /// 获取 u64 类型的元数据值。
    ///
    /// 同时支持 `Uint64` 和 `Uint32`（自动提升）。
    /// 如果键不存在或值类型不匹配，返回 `None`。
    pub fn metadata_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(|v| match v {
            GGUFMetadataValue::Uint64(v) => Some(*v),
            GGUFMetadataValue::Uint32(v) => Some(*v as u64),
            _ => None,
        })
    }

    /// 获取 i64 类型的元数据值。
    ///
    /// 同时支持 `Int64` 和 `Int32`（自动提升）。
    pub fn metadata_i64(&self, key: &str) -> Option<i64> {
        self.metadata.get(key).and_then(|v| match v {
            GGUFMetadataValue::Int64(v) => Some(*v),
            GGUFMetadataValue::Int32(v) => Some(*v as i64),
            _ => None,
        })
    }

    /// 获取 f32 类型的元数据值。
    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| match v {
            GGUFMetadataValue::Float32(v) => Some(*v),
            _ => None,
        })
    }

    /// 获取 bool 类型的元数据值。
    pub fn metadata_bool(&self, key: &str) -> Option<bool> {
        self.metadata.get(key).and_then(|v| match v {
            GGUFMetadataValue::Bool(v) => Some(*v),
            _ => None,
        })
    }

    // -----------------------------------------------------------------------
    // 张量查询
    // -----------------------------------------------------------------------

    /// 按名称查找张量信息。
    ///
    /// 返回第一个名称匹配的 `TensorInfo` 引用。如果未找到，返回 `None`。
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_infos.iter().find(|t| t.name == name)
    }

    /// 获取所有张量名称列表。
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_infos.iter().map(|t| t.name.as_str()).collect()
    }

    /// 获取张量总数。
    pub fn tensor_count(&self) -> usize {
        self.tensor_infos.len()
    }

    // -----------------------------------------------------------------------
    // 张量数据读取
    // -----------------------------------------------------------------------

    /// 计算指定张量在数据段中的字节大小。
    ///
    /// 通过比较当前张量与下一个张量的偏移量来确定大小；
    /// 最后一个张量的大小延伸到文件末尾。
    fn tensor_byte_size(&self, info: &TensorInfo) -> usize {
        let file_size = self._mmap.as_ref().map(|m| m.len()).unwrap_or(0);
        let data_section_end = file_size.saturating_sub(self.data_offset);

        // 查找当前张量之后偏移量最小的张量
        let mut next_offset = data_section_end as u64;
        for ti in &self.tensor_infos {
            if ti.offset > info.offset && ti.offset < next_offset {
                next_offset = ti.offset;
            }
        }

        (next_offset - info.offset) as usize
    }

    /// 读取张量的原始字节数据。
    ///
    /// 返回从 mmap 中拷贝的字节向量。如果映射不可用或数据越界，返回空向量。
    pub fn read_tensor_data(&self, info: &TensorInfo) -> Vec<u8> {
        let byte_size = self.tensor_byte_size(info);
        let start = self.data_offset + info.offset as usize;
        let end = start + byte_size;

        match self._mmap.as_ref() {
            Some(mmap) if end <= mmap.len() => mmap[start..end].to_vec(),
            _ => Vec::new(),
        }
    }

    /// 读取张量数据并反量化为 f32 向量。
    ///
    /// 调用 `crate::quant::dequantize` 将量化数据（如 Q4_0、Q8_0 等）
    /// 转换为全精度 f32 表示。
    ///
    /// # 错误
    ///
    /// - 张量数据为空
    /// - 反量化过程失败（不支持的类型或数据损坏）
    pub fn read_tensor_data_f32(&self, info: &TensorInfo) -> Result<Vec<f32>, String> {
        let raw = self.read_tensor_data(info);
        if raw.is_empty() {
            return Err(format!("张量 '{}' 的数据为空", info.name));
        }
        crate::quant::dequantize(&raw, info.ggml_type as i32, info.nelement())
    }

    // -----------------------------------------------------------------------
    // 模型架构信息提取
    // -----------------------------------------------------------------------

    /// 提取模型架构名称（如 `"llama"`、`"qwen2"`、`"mistral"` 等）。
    ///
    /// 对应元数据键 `general.architecture`。
    pub fn model_architecture(&self) -> Option<String> {
        self.metadata_string("general.architecture").cloned()
    }

    /// 提取上下文长度（context length）。
    ///
    /// 对应元数据键 `{architecture}.context_length`。
    pub fn context_length(&self) -> Option<u64> {
        let arch = self.model_architecture()?;
        self.metadata_u64(&format!("{}.context_length", arch))
    }

    /// 提取嵌入维度（embedding dimension）。
    ///
    /// 对应元数据键 `{architecture}.embedding_length`。
    pub fn embedding_length(&self) -> Option<u64> {
        let arch = self.model_architecture()?;
        self.metadata_u64(&format!("{}.embedding_length", arch))
    }

    /// 提取注意力头数。
    ///
    /// 对应元数据键 `{architecture}.attention.head_count`。
    pub fn head_count(&self) -> Option<u64> {
        let arch = self.model_architecture()?;
        self.metadata_u64(&format!("{}.attention.head_count", arch))
    }

    /// 提取 KV 注意力头数（用于分组查询注意力 GQA）。
    ///
    /// 对应元数据键 `{architecture}.attention.head_count_kv`。
    pub fn head_count_kv(&self) -> Option<u64> {
        let arch = self.model_architecture()?;
        self.metadata_u64(&format!("{}.attention.head_count_kv", arch))
    }

    /// 提取 Transformer 层数。
    ///
    /// 对应元数据键 `{architecture}.block_count`。
    pub fn layer_count(&self) -> Option<u64> {
        let arch = self.model_architecture()?;
        self.metadata_u64(&format!("{}.block_count", arch))
    }

    /// 提取词表大小。
    ///
    /// 优先读取 `general.tokenizer.ggml.vocab_size`；
    /// 若不存在，回退到 `general.tokenizer.ggml.tokens` 数组长度。
    pub fn vocab_size(&self) -> Option<u64> {
        // 优先使用显式字段
        if let Some(v) = self.metadata_u64("general.tokenizer.ggml.vocab_size") {
            return Some(v);
        }
        // 回退到 tokens 数组长度
        if let Some(GGUFMetadataValue::Array(arr)) =
            self.metadata.get("general.tokenizer.ggml.tokens")
        {
            return Some(arr.values.len() as u64);
        }
        None
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    // WriteBytesExt provides write_*, no need for std::io::Write explicitly

    /// 将 GGUF 字符串写入缓冲区（u64 长度 + 字节）
    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
        buf.extend_from_slice(s.as_bytes());
    }

    /// 写入一个 UINT32 元数据键值对
    fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
        write_gguf_string(buf, key);
        buf.write_u32::<LittleEndian>(GGUFValueType::Uint32 as u32)
            .unwrap();
        buf.write_u32::<LittleEndian>(value).unwrap();
    }

    /// 写入一个 STRING 元数据键值对
    fn write_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
        write_gguf_string(buf, key);
        buf.write_u32::<LittleEndian>(GGUFValueType::String as u32)
            .unwrap();
        write_gguf_string(buf, value);
    }

    /// 写入一个 UINT64 元数据键值对
    fn write_kv_u64(buf: &mut Vec<u8>, key: &str, value: u64) {
        write_gguf_string(buf, key);
        buf.write_u32::<LittleEndian>(GGUFValueType::Uint64 as u32)
            .unwrap();
        buf.write_u64::<LittleEndian>(value).unwrap();
    }

    /// 构建最小有效 GGUF 二进制数据并写入临时文件，返回文件路径。
    ///
    /// 生成的文件包含：
    /// - 文件头（魔数、版本 V3）
    /// - `general.alignment` 元数据（UINT32）
    /// - `general.architecture` 元数据（STRING）
    /// - 一个 2D 张量描述（4×8, F32）
    /// - 对齐后的张量数据（32 个 f32 值）
    fn build_minimal_gguf() -> (Vec<u8>, String) {
        let mut buf = Vec::new();

        // --- 文件头 ---
        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(GGUF_VERSION_V3).unwrap();
        buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count = 1
        buf.write_u64::<LittleEndian>(2).unwrap(); // metadata_kv_count = 2

        // --- 元数据 ---
        write_kv_u32(&mut buf, "general.alignment", 32);
        write_kv_string(&mut buf, "general.architecture", "testarch");

        // --- 张量信息 ---
        write_gguf_string(&mut buf, "test.weight");
        buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims = 2
        buf.write_u64::<LittleEndian>(4).unwrap(); // dim[0] = 4
        buf.write_u64::<LittleEndian>(8).unwrap(); // dim[1] = 8
        buf.write_u32::<LittleEndian>(0).unwrap(); // ggml_type = F32
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset = 0

        // --- 对齐填充 ---
        let header_end = buf.len();
        let aligned = align_to(header_end, 32);
        while buf.len() < aligned {
            buf.push(0);
        }

        // --- 张量数据（4×8 = 32 个 f32）---
        for i in 0..32 {
            buf.write_f32::<LittleEndian>((i as f32) * 0.5).unwrap();
        }

        let path = std::env::temp_dir().join("fishai_test_minimal.gguf");
        let path_str = path.to_string_lossy().into_owned();
        std::fs::write(&path, &buf).unwrap();

        (buf, path_str)
    }

    // -----------------------------------------------------------------------
    // 测试 1: GGUFValueType 转换
    // -----------------------------------------------------------------------

    #[test]
    fn test_value_type_conversion() {
        // 所有有效类型 ID 都应正确转换
        assert_eq!(GGUFValueType::from_u32(0), Some(GGUFValueType::Uint8));
        assert_eq!(GGUFValueType::from_u32(1), Some(GGUFValueType::Int8));
        assert_eq!(GGUFValueType::from_u32(2), Some(GGUFValueType::Uint16));
        assert_eq!(GGUFValueType::from_u32(3), Some(GGUFValueType::Int16));
        assert_eq!(GGUFValueType::from_u32(4), Some(GGUFValueType::Uint32));
        assert_eq!(GGUFValueType::from_u32(5), Some(GGUFValueType::Int32));
        assert_eq!(GGUFValueType::from_u32(6), Some(GGUFValueType::Float32));
        assert_eq!(GGUFValueType::from_u32(7), Some(GGUFValueType::Bool));
        assert_eq!(GGUFValueType::from_u32(8), Some(GGUFValueType::String));
        assert_eq!(GGUFValueType::from_u32(9), Some(GGUFValueType::Array));
        assert_eq!(GGUFValueType::from_u32(10), Some(GGUFValueType::Uint64));
        assert_eq!(GGUFValueType::from_u32(11), Some(GGUFValueType::Int64));
        assert_eq!(GGUFValueType::from_u32(12), Some(GGUFValueType::Float64));

        // 无效类型 ID 应返回 None
        assert_eq!(GGUFValueType::from_u32(13), None);
        assert_eq!(GGUFValueType::from_u32(100), None);
        assert_eq!(GGUFValueType::from_u32(255), None);
        assert_eq!(GGUFValueType::from_u32(u32::MAX), None);
    }

    // -----------------------------------------------------------------------
    // 测试 2: GGUFString
    // -----------------------------------------------------------------------

    #[test]
    fn test_gguf_string() {
        // 有效 UTF-8
        let s = GGUFString {
            len: 5,
            data: b"hello".to_vec(),
        };
        assert_eq!(s.as_str(), Some("hello"));
        assert_eq!(s.to_string_lossy(), "hello");

        // 包含中文的 UTF-8
        let s_cn = GGUFString {
            len: 9,
            data: "你好世界!".as_bytes().to_vec(),
        };
        assert_eq!(s_cn.as_str(), Some("你好世界!"));
        assert_eq!(s_cn.to_string_lossy(), "你好世界!");

        // 无效 UTF-8 字节
        let s_bad = GGUFString {
            len: 2,
            data: vec![0xFF, 0xFE],
        };
        assert!(s_bad.as_str().is_none());
        let lossy = s_bad.to_string_lossy();
        assert_eq!(lossy, "\u{FFFD}\u{FFFD}");

        // 空字符串
        let s_empty = GGUFString {
            len: 0,
            data: vec![],
        };
        assert_eq!(s_empty.as_str(), Some(""));
        assert_eq!(s_empty.to_string_lossy(), "");

        // PartialEq
        let s1 = GGUFString {
            len: 3,
            data: b"abc".to_vec(),
        };
        let s2 = GGUFString {
            len: 3,
            data: b"abc".to_vec(),
        };
        let s3 = GGUFString {
            len: 3,
            data: b"abd".to_vec(),
        };
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    // -----------------------------------------------------------------------
    // 测试 3: TensorInfo::shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensor_info_shape() {
        // 2D 张量
        let info_2d = TensorInfo {
            name: "layer.weight".to_string(),
            n_dims: 2,
            dims: vec![4096, 11008],
            ggml_type: 2, // Q4_0
            offset: 0,
        };
        assert_eq!(info_2d.shape(), vec![4096usize, 11008]);

        // 1D 张量
        let info_1d = TensorInfo {
            name: "bias".to_string(),
            n_dims: 1,
            dims: vec![4096],
            ggml_type: 0, // F32
            offset: 1024,
        };
        assert_eq!(info_1d.shape(), vec![4096usize]);

        // 4D 张量
        let info_4d = TensorInfo {
            name: "conv.weight".to_string(),
            n_dims: 4,
            dims: vec![3, 64, 7, 7],
            ggml_type: 0,
            offset: 2048,
        };
        assert_eq!(info_4d.shape(), vec![3usize, 64, 7, 7]);

        // 0D 标量
        let info_0d = TensorInfo {
            name: "scalar".to_string(),
            n_dims: 0,
            dims: vec![],
            ggml_type: 0,
            offset: 4096,
        };
        assert_eq!(info_0d.shape(), Vec::<usize>::new());
    }

    // -----------------------------------------------------------------------
    // 测试 4: TensorInfo::nelement
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensor_info_nelement() {
        // 2D: 4 × 8 = 32
        let info = TensorInfo {
            name: "test.weight".to_string(),
            n_dims: 2,
            dims: vec![4, 8],
            ggml_type: 0,
            offset: 0,
        };
        assert_eq!(info.nelement(), 32);

        // 1D: 128
        let info_1d = TensorInfo {
            name: "bias".to_string(),
            n_dims: 1,
            dims: vec![128],
            ggml_type: 0,
            offset: 0,
        };
        assert_eq!(info_1d.nelement(), 128);

        // 3D: 2 × 3 × 4 = 24
        let info_3d = TensorInfo {
            name: "conv".to_string(),
            n_dims: 3,
            dims: vec![2, 3, 4],
            ggml_type: 0,
            offset: 0,
        };
        assert_eq!(info_3d.nelement(), 24);

        // 标量（0 维）：乘积为 1
        let info_scalar = TensorInfo {
            name: "scalar".to_string(),
            n_dims: 0,
            dims: vec![],
            ggml_type: 0,
            offset: 0,
        };
        assert_eq!(info_scalar.nelement(), 1);

        // 大维度：4096 × 11008 = 45088768
        let info_large = TensorInfo {
            name: "large.weight".to_string(),
            n_dims: 2,
            dims: vec![4096, 11008],
            ggml_type: 0,
            offset: 0,
        };
        assert_eq!(info_large.nelement(), 45088768);
    }

    // -----------------------------------------------------------------------
    // 测试 5: 完整 GGUF 文件解析
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_minimal_gguf_file() {
        let (_raw, path) = build_minimal_gguf();

        let file = match GGUFFile::open(&path) {
            Ok(f) => f,
            Err(e) => {
                let _ = std::fs::remove_file(&path);
                panic!("解析 GGUF 文件失败: {}", e);
            }
        };

        // 验证基本属性
        assert_eq!(file.version, GGUF_VERSION_V3);
        assert_eq!(file.alignment, 32);
        assert_eq!(file.tensor_count(), 1);

        // 验证元数据
        assert_eq!(
            file.metadata_string("general.architecture"),
            Some(&"testarch".to_string())
        );
        assert_eq!(file.model_architecture(), Some("testarch".to_string()));
        assert_eq!(file.metadata_u64("general.alignment"), Some(32));

        // 缺失的元数据应返回 None
        assert!(file.metadata_string("nonexistent.key").is_none());
        assert!(file.metadata_f32("nonexistent.key").is_none());
        assert!(file.metadata_bool("nonexistent.key").is_none());

        // 验证张量信息
        assert_eq!(file.tensor_names(), vec!["test.weight"]);
        let ti = file.tensor_info("test.weight").expect("应找到张量 test.weight");
        assert_eq!(ti.n_dims, 2);
        assert_eq!(ti.dims, vec![4, 8]);
        assert_eq!(ti.ggml_type, 0); // F32
        assert_eq!(ti.shape(), vec![4usize, 8]);
        assert_eq!(ti.nelement(), 32);

        // 验证张量数据读取
        let raw_data = file.read_tensor_data(ti);
        assert_eq!(raw_data.len(), 32 * 4); // 32 个 f32 = 128 字节

        // 验证数据内容
        let mut cursor = Cursor::new(&raw_data);
        for i in 0..32u32 {
            let val = cursor.read_f32::<LittleEndian>().unwrap();
            assert!((val - (i as f32) * 0.5).abs() < f32::EPSILON);
        }

        // 清理
        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // 测试 6: 错误的魔数
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_magic() {
        let path = std::env::temp_dir().join("fishai_test_bad_magic.gguf");
        // 写入无效魔数
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(0xDEADBEEF).unwrap(); // 错误的魔数
        buf.write_u32::<LittleEndian>(GGUF_VERSION_V3).unwrap();
        buf.write_u64::<LittleEndian>(0).unwrap();
        buf.write_u64::<LittleEndian>(0).unwrap();
        std::fs::write(&path, &buf).unwrap();

        let result = GGUFFile::open(path.to_str().unwrap());
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("无效的 GGUF 魔数"), "错误信息应为魔数无效: {}", err_msg);

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // 测试 7: 不支持的版本号
    // -----------------------------------------------------------------------

    #[test]
    fn test_unsupported_version() {
        let path = std::env::temp_dir().join("fishai_test_bad_version.gguf");
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(99).unwrap(); // 不支持的版本
        buf.write_u64::<LittleEndian>(0).unwrap();
        buf.write_u64::<LittleEndian>(0).unwrap();
        std::fs::write(&path, &buf).unwrap();

        let result = GGUFFile::open(path.to_str().unwrap());
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("不支持的 GGUF 版本"),
            "错误信息应包含版本不支持: {}",
            err_msg
        );

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // 测试 8: 对齐计算
    // -----------------------------------------------------------------------

    #[test]
    fn test_align_to() {
        // 基本对齐
        assert_eq!(align_to(0, 32), 0);
        assert_eq!(align_to(1, 32), 32);
        assert_eq!(align_to(31, 32), 32);
        assert_eq!(align_to(32, 32), 32);
        assert_eq!(align_to(33, 32), 64);
        assert_eq!(align_to(63, 32), 64);
        assert_eq!(align_to(64, 32), 64);

        // 不同对齐值
        assert_eq!(align_to(0, 16), 0);
        assert_eq!(align_to(1, 16), 16);
        assert_eq!(align_to(15, 16), 16);
        assert_eq!(align_to(16, 16), 16);
        assert_eq!(align_to(17, 16), 32);

        assert_eq!(align_to(0, 64), 0);
        assert_eq!(align_to(33, 64), 64);
        assert_eq!(align_to(64, 64), 64);
        assert_eq!(align_to(65, 64), 128);

        // 对齐值为 0 时不应 panic
        assert_eq!(align_to(42, 0), 42);
        assert_eq!(align_to(0, 0), 0);
    }

    // -----------------------------------------------------------------------
    // 测试 9: read_string 和 read_value_at 辅助函数
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_string_from_bytes() {
        // 构造 GGUF 字符串: len=5, "hello"
        let mut buf = Vec::new();
        buf.write_u64::<LittleEndian>(5).unwrap();
        buf.extend_from_slice(b"hello");

        let mut cursor = Cursor::new(&buf[..]);
        let s = read_string(&mut cursor).unwrap();
        assert_eq!(s.len, 5);
        assert_eq!(s.as_str(), Some("hello"));
        assert_eq!(s.to_string_lossy(), "hello");
    }

    #[test]
    fn test_read_value_types() {
        // 测试所有标量类型的读取
        let mut buf = Vec::new();
        buf.write_u8(42).unwrap();                           // UINT8 = 42
        buf.write_i8(-10).unwrap();                          // INT8 = -10
        buf.write_u16::<LittleEndian>(1000).unwrap();        // UINT16
        buf.write_i16::<LittleEndian>(-2000).unwrap();       // INT16
        buf.write_u32::<LittleEndian>(300000).unwrap();      // UINT32
        buf.write_i32::<LittleEndian>(-400000).unwrap();     // INT32
        buf.write_f32::<LittleEndian>(3.14).unwrap();        // FLOAT32
        buf.write_u8(1).unwrap();                            // BOOL = true
        buf.write_u64::<LittleEndian>(12345678901234).unwrap(); // UINT64
        buf.write_i64::<LittleEndian>(-98765432109876).unwrap(); // INT64
        buf.write_f64::<LittleEndian>(2.718281828).unwrap(); // FLOAT64

        let mut cursor = Cursor::new(&buf[..]);

        let v = read_value_at(&mut cursor, &GGUFValueType::Uint8).unwrap();
        assert_eq!(v, GGUFMetadataValue::Uint8(42));

        let v = read_value_at(&mut cursor, &GGUFValueType::Int8).unwrap();
        assert_eq!(v, GGUFMetadataValue::Int8(-10));

        let v = read_value_at(&mut cursor, &GGUFValueType::Uint16).unwrap();
        assert_eq!(v, GGUFMetadataValue::Uint16(1000));

        let v = read_value_at(&mut cursor, &GGUFValueType::Int16).unwrap();
        assert_eq!(v, GGUFMetadataValue::Int16(-2000));

        let v = read_value_at(&mut cursor, &GGUFValueType::Uint32).unwrap();
        assert_eq!(v, GGUFMetadataValue::Uint32(300000));

        let v = read_value_at(&mut cursor, &GGUFValueType::Int32).unwrap();
        assert_eq!(v, GGUFMetadataValue::Int32(-400000));

        let v = read_value_at(&mut cursor, &GGUFValueType::Float32).unwrap();
        match v {
            GGUFMetadataValue::Float32(f) => assert!((f - 3.14).abs() < f32::EPSILON),
            _ => panic!("期望 Float32"),
        }

        let v = read_value_at(&mut cursor, &GGUFValueType::Bool).unwrap();
        assert_eq!(v, GGUFMetadataValue::Bool(true));

        let v = read_value_at(&mut cursor, &GGUFValueType::Uint64).unwrap();
        assert_eq!(v, GGUFMetadataValue::Uint64(12345678901234));

        let v = read_value_at(&mut cursor, &GGUFValueType::Int64).unwrap();
        assert_eq!(v, GGUFMetadataValue::Int64(-98765432109876));

        let v = read_value_at(&mut cursor, &GGUFValueType::Float64).unwrap();
        match v {
            GGUFMetadataValue::Float64(f) => assert!((f - 2.718281828).abs() < 1e-9),
            _ => panic!("期望 Float64"),
        }
    }

    // -----------------------------------------------------------------------
    // 测试 10: 文件不存在
    // -----------------------------------------------------------------------

    #[test]
    fn test_file_not_found() {
        let result = GGUFFile::open("/nonexistent/path/model.gguf");
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("无法打开文件"),
            "错误信息应包含打开失败: {}",
            err_msg
        );
    }

    // -----------------------------------------------------------------------
    // 测试 11: 含多个张量和多种元数据类型的 GGUF 文件
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_tensor_gguf() {
        let mut buf = Vec::new();

        // --- 文件头 ---
        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(GGUF_VERSION_V3).unwrap();
        buf.write_u64::<LittleEndian>(3).unwrap(); // tensor_count = 3
        buf.write_u64::<LittleEndian>(6).unwrap(); // metadata_kv_count = 6

        // --- 元数据 ---
        write_kv_u32(&mut buf, "general.alignment", 64);
        write_kv_string(&mut buf, "general.architecture", "llama");

        // context_length (u64)
        write_kv_u64(&mut buf, "llama.context_length", 4096);
        // embedding_length (u64)
        write_kv_u64(&mut buf, "llama.embedding_length", 4096);
        // block_count / layer_count (u64)
        write_kv_u64(&mut buf, "llama.block_count", 32);
        // head_count (u64)
        write_kv_u64(&mut buf, "llama.attention.head_count", 32);

        // --- 张量 1: token_embd.weight (1D, 4096) ---
        write_gguf_string(&mut buf, "token_embd.weight");
        buf.write_u32::<LittleEndian>(1).unwrap(); // n_dims = 1
        buf.write_u64::<LittleEndian>(4096).unwrap();
        buf.write_u32::<LittleEndian>(0).unwrap(); // F32
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset = 0

        // --- 张量 2: output_norm.weight (1D, 4096) ---
        write_gguf_string(&mut buf, "output_norm.weight");
        buf.write_u32::<LittleEndian>(1).unwrap();
        buf.write_u64::<LittleEndian>(4096).unwrap();
        buf.write_u32::<LittleEndian>(0).unwrap(); // F32
        buf.write_u64::<LittleEndian>(16384).unwrap(); // offset = 4096 * 4

        // --- 张量 3: output.weight (2D, 4096 × 32000) ---
        // 这里只写描述，不写真实数据（测试中只验证头部解析）
        write_gguf_string(&mut buf, "output.weight");
        buf.write_u32::<LittleEndian>(2).unwrap();
        buf.write_u64::<LittleEndian>(32000).unwrap();
        buf.write_u64::<LittleEndian>(4096).unwrap();
        buf.write_u32::<LittleEndian>(0).unwrap(); // F32
        buf.write_u64::<LittleEndian>(32768).unwrap(); // offset

        // --- 对齐填充 ---
        let header_end = buf.len();
        let aligned = align_to(header_end, 64);
        while buf.len() < aligned {
            buf.push(0);
        }

        // --- 写入部分张量数据（只写前两个张量的）---
        // token_embd.weight: 4096 * 4 = 16384 字节
        for i in 0..4096 {
            buf.write_f32::<LittleEndian>(i as f32).unwrap();
        }
        // output_norm.weight: 4096 * 4 = 16384 字节
        for i in 0..4096 {
            buf.write_f32::<LittleEndian>((i as f32) * 2.0).unwrap();
        }

        // 写入文件
        let path = std::env::temp_dir().join("fishai_test_multi.gguf");
        let path_str = path.to_string_lossy().into_owned();
        std::fs::write(&path, &buf).unwrap();

        // 解析
        let file = GGUFFile::open(&path_str).unwrap();

        assert_eq!(file.version, GGUF_VERSION_V3);
        assert_eq!(file.alignment, 64);
        assert_eq!(file.tensor_count(), 3);
        assert_eq!(file.model_architecture(), Some("llama".to_string()));

        // 架构信息提取
        assert_eq!(file.context_length(), Some(4096));
        assert_eq!(file.embedding_length(), Some(4096));
        assert_eq!(file.layer_count(), Some(32));
        assert_eq!(file.head_count(), Some(32));

        // 张量列表
        let names = file.tensor_names();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"token_embd.weight"));
        assert!(names.contains(&"output_norm.weight"));
        assert!(names.contains(&"output.weight"));

        // 读取第一个张量数据
        let ti_embd = file.tensor_info("token_embd.weight").unwrap();
        assert_eq!(ti_embd.nelement(), 4096);
        assert_eq!(ti_embd.shape(), vec![4096usize]);
        let data_embd = file.read_tensor_data(ti_embd);
        assert_eq!(data_embd.len(), 4096 * 4);

        // 读取第二个张量数据
        let ti_norm = file.tensor_info("output_norm.weight").unwrap();
        let data_norm = file.read_tensor_data(ti_norm);
        assert_eq!(data_norm.len(), 4096 * 4);

        // 第三个张量数据应该为空（未写入）
        let ti_out = file.tensor_info("output.weight").unwrap();
        assert_eq!(ti_out.nelement(), 32000 * 4096);
        let data_out = file.read_tensor_data(ti_out);
        assert!(data_out.is_empty());

        // 清理
        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // 测试 12: metadata_u64 兼容 Uint32
    // -----------------------------------------------------------------------

    #[test]
    fn test_metadata_u64_from_u32() {
        let path = std::env::temp_dir().join("fishai_test_u32_u64.gguf");
        let mut buf = Vec::new();

        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(GGUF_VERSION_V3).unwrap();
        buf.write_u64::<LittleEndian>(0).unwrap(); // 0 张量
        buf.write_u64::<LittleEndian>(2).unwrap(); // 2 个元数据

        // general.alignment = 32 (UINT32)
        write_kv_u32(&mut buf, "general.alignment", 32);

        // my_value = 100 (UINT32, 但通过 metadata_u64 读取)
        write_kv_u32(&mut buf, "my_value", 100);

        // 对齐
        let header_end = buf.len();
        let aligned = align_to(header_end, 32);
        while buf.len() < aligned {
            buf.push(0);
        }

        std::fs::write(&path, &buf).unwrap();

        let file = GGUFFile::open(path.to_str().unwrap()).unwrap();

        // metadata_u64 应能读取 Uint32 值
        assert_eq!(file.metadata_u64("my_value"), Some(100));
        assert_eq!(file.metadata_u64("general.alignment"), Some(32));

        // metadata_i64 不应匹配 Uint32
        assert_eq!(file.metadata_i64("my_value"), None);

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // 测试 13: GGUFValueType Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_value_type_display() {
        assert_eq!(format!("{}", GGUFValueType::Uint8), "UINT8");
        assert_eq!(format!("{}", GGUFValueType::String), "STRING");
        assert_eq!(format!("{}", GGUFValueType::Float64), "FLOAT64");
        assert_eq!(format!("{}", GGUFValueType::Array), "ARRAY");
        assert_eq!(format!("{}", GGUFValueType::Bool), "BOOL");
    }
}