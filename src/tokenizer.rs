//! BPE (Byte Pair Encoding) 分词器模块
//!
//! 本模块实现了与 GGUF / SentencePiece 兼容的 BPE 分词器，核心功能包括：
//! - 基于正则表达式的预分词（GPT-2 风格）
//! - 字节对编码合并算法（按合并优先级迭代合并相邻 token 对）
//! - 字节回退机制（未知字节映射为 `<0xNN>` 形式的单字节 token）
//! - 特殊 token 处理（BOS / EOS / 用户自定义）
//! - GGUF 元数据解析（自动提取 tokenizer 配置）
//!
//! # 典型用法
//!
//! ```ignore
//! use fishai_engine::tokenizer::BpeTokenizer;
//!
//! let tokenizer = BpeTokenizer::from_gguf(tokens, scores, token_types, Some(merges), config)?;
//! let ids = tokenizer.encode("Hello, world!", true);
//! let text = tokenizer.decode(&ids);
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// TokenType 枚举
// ---------------------------------------------------------------------------

/// Token 类型，对应 SentencePiece / GGUF 中的 token_type 字段
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenType {
    /// 普通文本 token
    Normal = 0,
    /// 未知字节回退 token
    Unknown = 1,
    /// 单字节 token（byte-fallback BPE）
    Byte = 2,
    /// 特殊控制 token（BOS、EOS 等）
    Control = 3,
    /// 用户自定义特殊 token
    UserDefined = 4,
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenType::Normal => write!(f, "Normal"),
            TokenType::Unknown => write!(f, "Unknown"),
            TokenType::Byte => write!(f, "Byte"),
            TokenType::Control => write!(f, "Control"),
            TokenType::UserDefined => write!(f, "UserDefined"),
        }
    }
}

impl From<i32> for TokenType {
    fn from(v: i32) -> Self {
        match v {
            0 => TokenType::Normal,
            1 => TokenType::Unknown,
            2 => TokenType::Byte,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            _ => TokenType::Unknown,
        }
    }
}

// ---------------------------------------------------------------------------
// Token 结构体
// ---------------------------------------------------------------------------

/// 单个 token 的完整信息
#[derive(Debug, Clone)]
pub struct Token {
    /// token 在词表中的唯一编号
    pub id: u32,
    /// token 的文本表示
    pub text: String,
    /// token 的分数（通常来自 SentencePiece 训练，或 BPE 合并优先级）
    pub score: f32,
    /// token 类型
    pub type_: TokenType,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Token(id={}, text={:?}, score={:.6}, type={})",
            self.id, self.text, self.score, self.type_
        )
    }
}

impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Token {}

// ---------------------------------------------------------------------------
// TokenizerConfig
// ---------------------------------------------------------------------------

/// 从 GGUF 元数据中提取的分词器配置
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// BOS token ID
    pub bos_token_id: Option<u32>,
    /// EOS token ID
    pub eos_token_id: Option<u32>,
    /// EOT (end-of-turn) token ID
    pub eot_token_id: Option<u32>,
    /// EOM (end-of-message) token ID
    pub eom_token_id: Option<u32>,
    /// UNK token ID
    pub unk_token_id: Option<u32>,
    /// 编码时是否自动添加 BOS
    pub add_bos: bool,
    /// 编码时是否自动添加 EOS
    pub add_eos: bool,
    /// 是否在文本前添加空格前缀（SentencePiece 风格）
    pub add_space_prefix: bool,
    /// 是否移除特殊空白字符
    pub remove_special_whitespaces: bool,
    /// 是否在最前面插入 BOS（与 add_bos 类似但语义不同）
    pub prepend_bos: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        TokenizerConfig {
            bos_token_id: None,
            eos_token_id: None,
            eot_token_id: None,
            eom_token_id: None,
            unk_token_id: None,
            add_bos: false,
            add_eos: false,
            add_space_prefix: false,
            remove_special_whitespaces: false,
            prepend_bos: false,
        }
    }
}

// ---------------------------------------------------------------------------
// BpeTokenizer
// ---------------------------------------------------------------------------

/// BPE 分词器，兼容 GGUF / SentencePiece 词表
pub struct BpeTokenizer {
    /// 词表中所有 token（按 id 索引）
    vocab: Vec<Token>,
    /// 文本 → token id 的反向查找表
    token_to_id: HashMap<String, u32>,
    /// byte_tokens[byte_value] = token_id，用于字节回退
    byte_tokens: Vec<u32>,
    /// 分词器配置
    config: TokenizerConfig,
    /// BPE 合并优先级表：(left_id, right_id) → rank（数值越小优先级越高）
    bpe_ranks: HashMap<(u32, u32), u32>,
    /// 特殊 token 文本 → id 查找表
    special_tokens: HashMap<String, u32>,
    /// 预分词正则（GPT-2 风格）
    regex_pattern: regex::Regex,
}

impl fmt::Debug for BpeTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BpeTokenizer")
            .field("vocab_size", &self.vocab.len())
            .field("byte_tokens_len", &self.byte_tokens.len())
            .field("config", &self.config)
            .field("bpe_ranks_len", &self.bpe_ranks.len())
            .field("special_tokens", &self.special_tokens)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// 公共方法
// ---------------------------------------------------------------------------

impl BpeTokenizer {
    /// 创建一个空的分词器（词表为空）
    pub fn new() -> Self {
        // GPT-2 风格预分词正则：匹配词或空白
        // 匹配单个词（可选前导空格 + 非空白字符）或连续空白
        let regex_pattern = regex::Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\s]+|\s+")
            .expect("预分词正则编译失败");

        BpeTokenizer {
            vocab: Vec::new(),
            token_to_id: HashMap::new(),
            byte_tokens: vec![0u32; 256],
            config: TokenizerConfig::default(),
            bpe_ranks: HashMap::new(),
            special_tokens: HashMap::new(),
            regex_pattern,
        }
    }

    /// 从 GGUF 嵌入数据构建分词器
    ///
    /// # 参数
    /// - `tokens`: 词表中的所有 token（按 id 排列）
    /// - `scores`: 每个 token 的分数（与 tokens 一一对应）
    /// - `token_types`: 每个 token 的类型（与 tokens 一一对应）
    /// - `merges`: BPE 合并文件内容（每行格式为 "a b"），可选
    /// - `config`: 分词器配置
    pub fn from_gguf(
        tokens: Vec<Token>,
        scores: Vec<f32>,
        token_types: Vec<i32>,
        merges: Option<String>,
        config: TokenizerConfig,
    ) -> Result<Self, String> {
        let mut vocab: Vec<Token> = Vec::with_capacity(tokens.len());
        let mut token_to_id: HashMap<String, u32> = HashMap::new();
        let mut byte_tokens: Vec<u32> = vec![0u32; 256];
        let mut special_tokens: HashMap<String, u32> = HashMap::new();
        let mut byte_token_set: bool = false;

        for (i, mut token) in tokens.into_iter().enumerate() {
            let id = i as u32;
            token.id = id;

            // 如果 scores 和 token_types 长度匹配，则覆盖
            if i < scores.len() {
                token.score = scores[i];
            }
            if i < token_types.len() {
                token.type_ = TokenType::from(token_types[i]);
            }

            // 处理字节 token
            if token.type_ == TokenType::Byte {
                // 尝试从 <0xNN> 格式解析字节值
                let byte_val = parse_byte_token_text(&token.text);
                if let Some(b) = byte_val {
                    byte_tokens[b as usize] = id;
                    byte_token_set = true;
                }
            }

            // 记录特殊 token
            if token.type_ == TokenType::Control || token.type_ == TokenType::UserDefined {
                special_tokens.insert(token.text.clone(), id);
            }

            token_to_id.insert(token.text.clone(), id);
            vocab.push(token);
        }

        // 如果没有通过 Byte 类型 token 建立字节映射，尝试通过 text 查找
        if !byte_token_set {
            build_byte_tokens_fallback(&vocab, &mut byte_tokens);
        }

        // 解析 BPE 合并规则
        let mut bpe_ranks: HashMap<(u32, u32), u32> = HashMap::new();
        if let Some(merges_str) = merges {
            let lines: Vec<&str> = merges_str.lines().collect();
            for (rank, line) in lines.iter().enumerate() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let parts: Vec<&str> = line.split(' ').collect();
                if parts.len() >= 2 {
                    if let (Some(&left_id), Some(&right_id)) =
                        (token_to_id.get(parts[0]), token_to_id.get(parts[1]))
                    {
                        bpe_ranks.insert((left_id, right_id), rank as u32);
                    }
                }
            }
        }

        // GPT-2 风格预分词正则
        let regex_pattern = regex::Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\s]+|\s+")
            .expect("预分词正则编译失败");

        Ok(BpeTokenizer {
            vocab,
            token_to_id,
            byte_tokens,
            config,
            bpe_ranks,
            special_tokens,
            regex_pattern,
        })
    }

    /// 将文本编码为 token id 序列，可选添加 BOS/EOS
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut ids = Vec::new();

        if add_special && self.config.prepend_bos {
            if let Some(bos) = self.config.bos_token_id {
                ids.push(bos);
            }
        }

        let processed_text = if self.config.add_space_prefix {
            // 在文本前添加空格前缀（SentencePiece 风格）
            // 检查是否已有前导空格
            if !text.starts_with(' ') && !text.is_empty() {
                format!(" {}", text)
            } else {
                text.to_string()
            }
        } else {
            text.to_string()
        };

        let text_to_tokenize = if self.config.remove_special_whitespaces {
            remove_special_whitespaces(&processed_text)
        } else {
            processed_text
        };

        // 预分词：使用正则拆分
        let pieces: Vec<&str> = self
            .regex_pattern
            .find_iter(&text_to_tokenize)
            .map(|m| m.as_str())
            .collect();

        for piece in pieces {
            // 对每个片段应用 BPE
            let subwords = self.bpe(piece);
            for sw in &subwords {
                if let Some(&id) = self.token_to_id.get(sw) {
                    ids.push(id);
                } else {
                    // 回退：将未知子词按字节编码
                    let byte_ids = self.bytes_to_tokens(sw.as_bytes());
                    ids.extend(byte_ids);
                }
            }
        }

        if add_special && self.config.add_eos {
            if let Some(eos) = self.config.eos_token_id {
                ids.push(eos);
            }
        }

        ids
    }

    /// 编码文本，同时保留文本中的特殊 token（如 `<|im_start|>`）
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let remaining = text;
        let mut last_end = 0;

        // 查找所有特殊 token 的位置
        let mut special_positions: Vec<(usize, &str, u32)> = Vec::new();
        for (special_text, &special_id) in &self.special_tokens {
            let mut search_start = 0;
            while let Some(pos) = remaining[search_start..].find(special_text.as_str()) {
                let abs_pos = search_start + pos;
                special_positions.push((abs_pos, special_text.as_str(), special_id));
                search_start = abs_pos + special_text.len();
            }
        }

        // 按位置排序
        special_positions.sort_by_key(|(pos, _, _)| *pos);

        for (start, special_text, special_id) in special_positions {
            // 编码特殊 token 之前的普通文本
            if start > last_end {
                let normal_text = &text[last_end..start];
                let normal_ids = self.encode(normal_text, false);
                ids.extend(normal_ids);
            }
            // 插入特殊 token
            ids.push(special_id);
            last_end = start + special_text.len();
        }

        // 编码剩余的普通文本
        if last_end < text.len() {
            let normal_text = &text[last_end..];
            let normal_ids = self.encode(normal_text, false);
            ids.extend(normal_ids);
        }

        ids
    }

    /// 将 token id 序列解码回文本
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut result = String::new();
        for &id in ids {
            if let Some(piece) = self.token_to_piece(id) {
                // 如果是 Byte 类型的 token，将 <0xNN> 转回字节
                if let Some(byte_val) = parse_byte_token_text(&piece) {
                    result.push(byte_val as char);
                } else {
                    result.push_str(&piece);
                }
            }
        }
        result
    }

    /// 获取 token id 对应的文本片段
    pub fn token_to_piece(&self, id: u32) -> Option<String> {
        if (id as usize) < self.vocab.len() {
            Some(self.vocab[id as usize].text.clone())
        } else {
            None
        }
    }

    /// 返回词表大小
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// 返回 BOS token id
    pub fn bos_token(&self) -> Option<u32> {
        self.config.bos_token_id
    }

    /// 返回 EOS token id
    pub fn eos_token(&self) -> Option<u32> {
        self.config.eos_token_id
    }
}

// ---------------------------------------------------------------------------
// 私有辅助方法
// ---------------------------------------------------------------------------

impl BpeTokenizer {
    /// 将原始字节序列转换为 token id（使用字节回退）
    fn bytes_to_tokens(&self, bytes: &[u8]) -> Vec<u32> {
        let mut ids = Vec::with_capacity(bytes.len());
        for &b in bytes {
            ids.push(self.byte_tokens[b as usize]);
        }
        ids
    }

    /// 对单个预分词片段应用 BPE 合并算法
    ///
    /// 算法步骤：
    /// 1. 将片段拆分为单字符序列
    /// 2. 找到相邻 pair 中具有最低合并优先级（rank）的一对
    /// 3. 将该对合并为一个新的 token
    /// 4. 重复步骤 2-3 直到没有更多可合并的 pair
    fn bpe(&self, piece: &str) -> Vec<String> {
        if piece.is_empty() {
            return Vec::new();
        }

        // 将每个字符作为初始 token
        let mut word: Vec<String> = piece.chars().map(|c| c.to_string()).collect();

        if word.len() == 1 {
            return word;
        }

        loop {
            // 获取所有相邻 pair
            let pairs = get_pairs(&word);
            if pairs.is_empty() {
                break;
            }

            // 找到具有最低 rank 的 pair
            let mut best_pair: Option<(String, String)> = None;
            let mut best_rank: u32 = u32::MAX;

            for pair in &pairs {
                if let (Some(&left_id), Some(&right_id)) =
                    (self.token_to_id.get(&pair.0), self.token_to_id.get(&pair.1))
                {
                    if let Some(&rank) = self.bpe_ranks.get(&(left_id, right_id)) {
                        if rank < best_rank {
                            best_rank = rank;
                            best_pair = Some(pair.clone());
                        }
                    }
                }
            }

            // 没有可合并的 pair 了
            let best = match best_pair {
                Some(bp) => bp,
                None => break,
            };

            // 合并 best pair
            let merged = format!("{}{}", best.0, best.1);
            let mut new_word: Vec<String> = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i < word.len() - 1 && word[i] == best.0 && word[i + 1] == best.1 {
                    new_word.push(merged.clone());
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            word = new_word;

            if word.len() == 1 {
                break;
            }
        }

        word
    }
}

// ---------------------------------------------------------------------------
// 自由函数
// ---------------------------------------------------------------------------

/// 从 word（子词字符串列表）中获取所有相邻 pair
pub fn get_pairs(word: &[String]) -> Vec<(String, String)> {
    let mut pairs = Vec::with_capacity(word.len().saturating_sub(1));
    for i in 0..word.len().saturating_sub(1) {
        pairs.push((word[i].clone(), word[i + 1].clone()));
    }
    pairs
}

/// 从 GGUF 元数据中解析分词器配置
///
/// 支持的元数据键：
/// - `tokenizer.ggml.bos_token_id`
/// - `tokenizer.ggml.eos_token_id`
/// - `tokenizer.ggml.eot_token_id`
/// - `tokenizer.ggml.eom_token_id`
/// - `tokenizer.ggml.unknown_token_id`
/// - `tokenizer.ggml.add_bos`
/// - `tokenizer.ggml.add_eos`
/// - `tokenizer.ggml.add_space_prefix`
/// - `tokenizer.ggml.remove_special_whitespaces`
/// - `tokenizer.ggml.prepend_bos`
pub fn parse_tokenizer_config_from_gguf(metadata: &HashMap<String, String>) -> TokenizerConfig {
    let get_u32 = |key: &str| -> Option<u32> {
        metadata.get(key).and_then(|v| v.parse::<u32>().ok())
    };
    let get_bool = |key: &str| -> bool {
        metadata
            .get(key)
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false)
    };

    TokenizerConfig {
        bos_token_id: get_u32("tokenizer.ggml.bos_token_id"),
        eos_token_id: get_u32("tokenizer.ggml.eos_token_id"),
        eot_token_id: get_u32("tokenizer.ggml.eot_token_id"),
        eom_token_id: get_u32("tokenizer.ggml.eom_token_id"),
        unk_token_id: get_u32("tokenizer.ggml.unknown_token_id"),
        add_bos: get_bool("tokenizer.ggml.add_bos"),
        add_eos: get_bool("tokenizer.ggml.add_eos"),
        add_space_prefix: get_bool("tokenizer.ggml.add_space_prefix"),
        remove_special_whitespaces: get_bool("tokenizer.ggml.remove_special_whitespaces"),
        prepend_bos: get_bool("tokenizer.ggml.prepend_bos"),
    }
}

/// 从 `<0xNN>` 格式的文本中解析字节值
fn parse_byte_token_text(text: &str) -> Option<u8> {
    if text.starts_with("<0x") && text.ends_with('>') && text.len() == 6 {
        let hex = &text[3..5];
        u8::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

/// 当没有显式 Byte 类型 token 时，通过文本模式查找字节 token
fn build_byte_tokens_fallback(vocab: &[Token], byte_tokens: &mut [u32]) {
    for token in vocab {
        if let Some(byte_val) = parse_byte_token_text(&token.text) {
            byte_tokens[byte_val as usize] = token.id;
        }
    }
}

/// 移除特殊空白字符（零宽空格、不间断空格等）
fn remove_special_whitespaces(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{00A0}' | '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' => ' ',
            _ => c,
        })
        .collect()
}

// ===========================================================================
// 测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // 辅助：构建用于测试的简易分词器
    // -----------------------------------------------------------------------

    /// 创建一个包含基本 ASCII 字母 + 常见标点的最小词表和合并规则
    ///
    /// 词表布局:
    ///   0: "<unk>"  (Control)
    ///   1: "<s>"    (Control, BOS)
    ///   2: "</s>"   (Control, EOS)
    ///   3..258: "<0x00>" ~ "<0xFF>" (Byte)
    ///   259+: 字母 a-z, A-Z, 空格, 常见标点 (Normal)
    fn make_test_tokenizer() -> BpeTokenizer {
        let mut tokens: Vec<Token> = Vec::new();
        let mut id: u32 = 0;

        // 特殊 token
        tokens.push(Token {
            id,
            text: "<unk>".into(),
            score: 0.0,
            type_: TokenType::Control,
        });
        id += 1;

        let bos_id = id;
        tokens.push(Token {
            id,
            text: "<s>".into(),
            score: 0.0,
            type_: TokenType::Control,
        });
        id += 1;

        let eos_id = id;
        tokens.push(Token {
            id,
            text: "</s>".into(),
            score: 0.0,
            type_: TokenType::Control,
        });
        id += 1;

        // 字节 token <0x00> ~ <0xFF>
        for b in 0u8..=255 {
            let hex = format!("<0x{:02X}>", b);
            tokens.push(Token {
                id,
                text: hex,
                score: 0.0,
                type_: TokenType::Byte,
            });
            id += 1;
        }

        // 添加常用字符作为 Normal token: 空格 + 字母 + 标点
        let normal_chars: &str = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?";
        let mut char_to_id: HashMap<char, u32> = HashMap::new();
        for c in normal_chars.chars() {
            char_to_id.insert(c, id);
            tokens.push(Token {
                id,
                text: c.to_string(),
                score: -(id as f32), // 分数任意
                type_: TokenType::Normal,
            });
            id += 1;
        }

        // 合并规则: "a b" -> "ab", "b c" -> "bc", "a b c" 不直接存在, 需要
        // 先 merge "a b" 然后在含 "ab c" 时 merge "ab c" -> "abc"
        // 简单起见只添加一些:
        //   rank 0: "a" + "b" -> "ab"
        //   rank 1: "a" + "b" + "c": 需要 "ab"(合并后) + "c"
        //     我们需要先把 "ab" 加入词表
        let ab_id = id;
        tokens.push(Token {
            id,
            text: "ab".into(),
            score: -999.0,
            type_: TokenType::Normal,
        });
        id += 1;

        let _abc_id = id;
        tokens.push(Token {
            id,
            text: "abc".into(),
            score: -1000.0,
            type_: TokenType::Normal,
        });
        id += 1;

        let _he_id = id;
        tokens.push(Token {
            id,
            text: "he".into(),
            score: -1001.0,
            type_: TokenType::Normal,
        });
        id += 1;

        let _llo_id = id;
        tokens.push(Token {
            id,
            text: "llo".into(),
            score: -1002.0,
            type_: TokenType::Normal,
        });
        id += 1;

        // 构建 token_to_id 查找表
        let mut token_to_id: HashMap<String, u32> = HashMap::new();
        for t in &tokens {
            token_to_id.insert(t.text.clone(), t.id);
        }

        // 字节 token 映射
        let mut byte_tokens: Vec<u32> = vec![0u32; 256];
        for b in 0u8..=255 {
            byte_tokens[b as usize] = tokens[(3 + b as usize) as usize].id;
        }

        // 合并规则字符串
        // 需要确保合并的左右 token text 都在 token_to_id 中
        let a_id = *char_to_id.get(&'a').unwrap();
        let b_id = *char_to_id.get(&'b').unwrap();
        let c_id = *char_to_id.get(&'c').unwrap();
        let h_id = *char_to_id.get(&'h').unwrap();
        let e_id = *char_to_id.get(&'e').unwrap();
        let l_id = *char_to_id.get(&'l').unwrap();
        let o_id = *char_to_id.get(&'o').unwrap();

        let mut merges = String::new();
        // 构建合并规则: rank, left_text, right_text
        let merge_entries: Vec<(&str, &str)> = vec![
            ("a", "b"),   // -> ab (rank 0)
            ("ab", "c"),  // -> abc (rank 1)
            ("h", "e"),   // -> he (rank 2)
            ("l", "l"),   // -> ll (rank 3)
            ("l", "o"),   // -> lo (rank 4)
            ("ll", "o"),  // -> llo (rank 5)
        ];

        // 我们需要 "ll" 也作为词表中的 token
        let ll_id = id;
        tokens.push(Token {
            id,
            text: "ll".into(),
            score: -1003.0,
            type_: TokenType::Normal,
        });
        id += 1;
        token_to_id.insert("ll".into(), ll_id);

        let lo_id = id;
        tokens.push(Token {
            id,
            text: "lo".into(),
            score: -1004.0,
            type_: TokenType::Normal,
        });
        token_to_id.insert("lo".into(), lo_id);

        for (left, right) in &merge_entries {
            merges.push_str(&format!("{} {}\n", left, right));
        }

        // 构建 bpe_ranks
        let mut bpe_ranks: HashMap<(u32, u32), u32> = HashMap::new();
        // rank 0: a + b
        bpe_ranks.insert((a_id, b_id), 0);
        // rank 1: ab + c
        bpe_ranks.insert((ab_id, c_id), 1);
        // rank 2: h + e
        bpe_ranks.insert((h_id, e_id), 2);
        // rank 3: l + l
        bpe_ranks.insert((l_id, l_id), 3);
        // rank 4: l + o
        bpe_ranks.insert((l_id, o_id), 4);
        // rank 5: ll + o
        bpe_ranks.insert((ll_id, o_id), 5);

        let mut special_tokens: HashMap<String, u32> = HashMap::new();
        special_tokens.insert("<s>".into(), bos_id);
        special_tokens.insert("</s>".into(), eos_id);
        special_tokens.insert("<unk>".into(), 0);

        let regex_pattern = regex::Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\s]+|\s+").unwrap();

        BpeTokenizer {
            vocab: tokens,
            token_to_id,
            byte_tokens,
            config: TokenizerConfig {
                bos_token_id: Some(bos_id),
                eos_token_id: Some(eos_id),
                eot_token_id: None,
                eom_token_id: None,
                unk_token_id: Some(0),
                add_bos: false,
                add_eos: false,
                add_space_prefix: false,
                remove_special_whitespaces: false,
                prepend_bos: false,
            },
            bpe_ranks,
            special_tokens,
            regex_pattern,
        }
    }

    // -----------------------------------------------------------------------
    // 测试用例
    // -----------------------------------------------------------------------

    #[test]
    fn test_token_type() {
        let normal = TokenType::Normal;
        let unknown = TokenType::Unknown;
        let byte = TokenType::Byte;
        let control = TokenType::Control;
        let user = TokenType::UserDefined;

        assert_eq!(normal as i32, 0);
        assert_eq!(unknown as i32, 1);
        assert_eq!(byte as i32, 2);
        assert_eq!(control as i32, 3);
        assert_eq!(user as i32, 4);

        // 测试 From<i32> 转换
        assert_eq!(TokenType::from(0), TokenType::Normal);
        assert_eq!(TokenType::from(1), TokenType::Unknown);
        assert_eq!(TokenType::from(2), TokenType::Byte);
        assert_eq!(TokenType::from(3), TokenType::Control);
        assert_eq!(TokenType::from(4), TokenType::UserDefined);
        assert_eq!(TokenType::from(99), TokenType::Unknown); // 未知值默认 Unknown

        // 测试 Display
        assert_eq!(format!("{}", normal), "Normal");
        assert_eq!(format!("{}", control), "Control");
    }

    #[test]
    fn test_tokenizer_creation() {
        let tok = BpeTokenizer::new();
        assert_eq!(tok.vocab_size(), 0);
        assert_eq!(tok.bos_token(), None);
        assert_eq!(tok.eos_token(), None);
    }

    #[test]
    fn test_encode_basic() {
        let tok = make_test_tokenizer();
        // "hi" 应该编码为 h 和 i 各一个 token
        let ids = tok.encode("hi", false);
        // 每个 ASCII 字符都应该有自己的 token
        for &id in &ids {
            assert!(id < tok.vocab_size() as u32);
        }
        // 至少有 2 个 token: 'h' 和 'i'
        assert!(ids.len() >= 2);
        // 验证解码回来是 "hi"
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hi");
    }

    #[test]
    fn test_encode_with_bos_eos() {
        let mut tok = make_test_tokenizer();
        tok.config.prepend_bos = true;
        tok.config.add_eos = true;

        let bos_id = tok.config.bos_token_id.unwrap();
        let eos_id = tok.config.eos_token_id.unwrap();

        let ids = tok.encode("hi", true);
        assert!(!ids.is_empty());
        assert_eq!(ids[0], bos_id, "第一个 token 应该是 BOS");
        assert_eq!(*ids.last().unwrap(), eos_id, "最后一个 token 应该是 EOS");

        // 中间应该包含 "hi" 的 token
        assert!(ids.len() > 2);
    }

    #[test]
    fn test_decode_basic() {
        let tok = make_test_tokenizer();

        // 编码 "abc"
        let ids = tok.encode("abc", false);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "abc");

        // 编码 "hello"
        let ids = tok.encode("hello", false);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_roundtrip() {
        let tok = make_test_tokenizer();
        let test_texts = vec![
            "hello",
            "abc",
            "hi there",
            "Hello, World!",
            "a",
            "test 123",
        ];

        for text in &test_texts {
            let ids = tok.encode(text, false);
            let decoded = tok.decode(&ids);
            assert_eq!(
                &decoded, text,
                "Roundtrip 失败: 原文 {:?} -> {:?} -> {:?}",
                text, ids, decoded
            );
        }
    }

    #[test]
    fn test_byte_fallback() {
        let tok = make_test_tokenizer();

        // byte_tokens 应该已经设置好: byte_tokens[byte_val] = 对应 <0xNN> token id
        // 每个字节 0x00-0xFF 都应该映射到正确 id
        for b in 0u8..=255 {
            let id = tok.byte_tokens[b as usize];
            assert_ne!(id, 0, "字节 {} 应该有非零的 token id", b);
            assert!(
                (id as usize) < tok.vocab_size(),
                "字节 {} 的 token id {} 超出词表范围",
                b,
                id
            );
            let piece = tok.token_to_piece(id).unwrap();
            assert_eq!(piece, format!("<0x{:02X}>", b));
        }

        // bytes_to_tokens 应该为每个字节返回正确的 id
        let input_bytes = b"hello";
        let byte_ids = tok.bytes_to_tokens(input_bytes);
        assert_eq!(byte_ids.len(), input_bytes.len());
        for (i, &id) in byte_ids.iter().enumerate() {
            let piece = tok.token_to_piece(id).unwrap();
            assert_eq!(piece, format!("<0x{:02X}>", input_bytes[i]));
        }
    }

    #[test]
    fn test_bpe_merge() {
        let tok = make_test_tokenizer();

        // "abc" 应该被 BPE 合并为: a+b -> ab, ab+c -> abc
        let subwords = tok.bpe("abc");
        assert!(
            subwords.contains(&"abc".to_string()),
            "BPE 合并 'abc' 应该产生 ['abc'], 实际: {:?}",
            subwords
        );
        assert_eq!(subwords.len(), 1, "abc 应该被合并为单个 token");

        // "hello" 应该被 BPE 合并: h+e -> he, l+l -> ll, ll+o -> llo, 最终 [he, llo]
        let subwords = tok.bpe("hello");
        assert!(
            subwords.contains(&"he".to_string()),
            "BPE 'hello' 应包含 'he'"
        );
        assert!(
            subwords.contains(&"llo".to_string()),
            "BPE 'hello' 应包含 'llo'"
        );
        assert_eq!(
            subwords.len(),
            2,
            "hello 应该被合并为 [he, llo], 实际: {:?}",
            subwords
        );

        // "ab" 应该被合并为单个 token
        let subwords = tok.bpe("ab");
        assert_eq!(subwords, vec!["ab"]);
    }

    #[test]
    fn test_vocab_size() {
        let tok = make_test_tokenizer();
        // 3 个特殊 token + 256 字节 token + 常规字符 + 合并 token
        assert!(tok.vocab_size() > 256);
        assert_eq!(tok.vocab_size(), tok.vocab.len());
    }

    #[test]
    fn test_get_pairs() {
        let word = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let pairs = get_pairs(&word);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], ("a".to_string(), "b".to_string()));
        assert_eq!(pairs[1], ("b".to_string(), "c".to_string()));

        // 单个元素，没有 pair
        let word = vec!["x".to_string()];
        let pairs = get_pairs(&word);
        assert!(pairs.is_empty());

        // 空列表
        let word: Vec<String> = Vec::new();
        let pairs = get_pairs(&word);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_from_gguf_basic() {
        // 构建最小词表
        let tokens = vec![
            Token {
                id: 0,
                text: "<s>".into(),
                score: 0.0,
                type_: TokenType::Control,
            },
            Token {
                id: 1,
                text: "</s>".into(),
                score: 0.0,
                type_: TokenType::Control,
            },
        ];
        let scores = vec![0.0, 0.0];
        let token_types = vec![3, 3]; // Control
        let config = TokenizerConfig {
            bos_token_id: Some(0),
            eos_token_id: Some(1),
            ..Default::default()
        };

        let tok = BpeTokenizer::from_gguf(tokens, scores, token_types, None, config).unwrap();
        assert_eq!(tok.vocab_size(), 2);
        assert_eq!(tok.bos_token(), Some(0));
        assert_eq!(tok.eos_token(), Some(1));
    }

    #[test]
    fn test_parse_tokenizer_config_from_gguf() {
        let mut metadata: HashMap<String, String> = HashMap::new();
        metadata.insert("tokenizer.ggml.bos_token_id".into(), "1".into());
        metadata.insert("tokenizer.ggml.eos_token_id".into(), "2".into());
        metadata.insert("tokenizer.ggml.add_bos".into(), "true".into());
        metadata.insert("tokenizer.ggml.add_eos".into(), "false".into());
        metadata.insert("tokenizer.ggml.add_space_prefix".into(), "1".into());

        let config = parse_tokenizer_config_from_gguf(&metadata);
        assert_eq!(config.bos_token_id, Some(1));
        assert_eq!(config.eos_token_id, Some(2));
        assert_eq!(config.add_bos, true);
        assert_eq!(config.add_eos, false);
        assert_eq!(config.add_space_prefix, true);
        assert_eq!(config.prepend_bos, false);
    }

    #[test]
    fn test_token_to_piece() {
        let tok = make_test_tokenizer();

        // BOS token
        let piece = tok.token_to_piece(1);
        assert_eq!(piece, Some("<s>".to_string()));

        // EOS token
        let piece = tok.token_to_piece(2);
        assert_eq!(piece, Some("</s>".to_string()));

        // 超出范围的 id
        let piece = tok.token_to_piece(999_999);
        assert_eq!(piece, None);
    }

    #[test]
    fn test_encode_with_special_tokens() {
        let tok = make_test_tokenizer();

        // 测试包含特殊 token 的文本
        let text = "<s>hello</s>";
        let ids = tok.encode_with_special(text);

        // 第一个 token 应该是 BOS (id=1)
        assert_eq!(ids[0], tok.config.bos_token_id.unwrap());

        // 最后一个 token 应该是 EOS (id=2)
        assert_eq!(*ids.last().unwrap(), tok.config.eos_token_id.unwrap());

        // 中间应该包含 "hello" 的 token
        assert!(ids.len() > 2);
    }
}
