//! FishAI Engine — FishLab-ai 自研 AI 引擎核心逻辑
//!
//! Rust 编写的 AI 逻辑层，覆盖：
//! - Token 采样策略（temperature / top-k / top-p / repetition penalty）
//! - 系统提示词组装（身份 + 能力 + 记忆注入 + 深度思考）
//! - 记忆管理（指令提取 / 内容清理）
//! - 深度思考解析（流式标签解析）
//! - 聊天引擎（消息构建 / 后处理）
//! - HTTP API 服务

pub mod chat;
pub mod http;
pub mod memory;
pub mod prompt;
pub mod sampling;
pub mod thinking;

pub use chat::{ChatEngine, ChatMessage, ResponseResult};
pub use memory::{MemoryAction, MemoryManager, MemoryOp};
pub use prompt::{MemoryEntry, MemoryMode, Memories, PromptOptions, SystemPrompt};
pub use sampling::Sampler;
pub use thinking::{ParseEvent, ThinkingParser};
