//! FishAI Engine — FishLab-ai 自研 AI 引擎核心逻辑
//!
//! Rust 编写的 AI 逻辑层，覆盖：
//! - Token 采样策略（temperature / top-k / top-p / repetition penalty）
//! - 系统提示词组装（身份 + 能力 + 记忆注入 + 深度思考）
//! - 记忆管理（指令提取 / 内容清理）
//! - 深度思考解析（流式标签解析）
//! - 聊天引擎（消息构建 / 后处理）
//! - HTTP API 服务
//! - GGUF 模型文件解析（内存映射 / 元数据 / 张量加载）
//! - KV 缓存（自回归推理中 Key/Value 张量的高效缓存）
//! - Transformer 模型推理（前向传播 / KV 缓存 / 自回归生成）

pub mod chat;
pub mod gguf;
pub mod http;
pub mod kv_cache;
pub mod memory;
pub mod model;
pub mod prompt;
pub mod quant;
pub mod sampling;
pub mod tensor;
pub mod thinking;
pub mod tokenizer;

pub use chat::{ChatEngine, ChatMessage, ResponseResult};
pub use kv_cache::KVCache;
pub use memory::{MemoryAction, MemoryManager, MemoryOp};
pub use model::{GenerationConfig, GenerationState, ModelConfig, ModelWeights};
pub use prompt::{MemoryEntry, MemoryMode, Memories, PromptOptions, SystemPrompt};
pub use sampling::Sampler;
pub use thinking::{ParseEvent, ThinkingParser};
pub use tokenizer::{BpeTokenizer, Token, TokenType, TokenizerConfig};