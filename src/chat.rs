//! FishAI 聊天引擎
//!
//! 组装系统提示、构建消息列表、后处理 AI 输出。

use crate::memory::{MemoryManager, MemoryOp};
use crate::prompt::{Memories, MemoryMode, PromptOptions, SystemPrompt};
use crate::thinking::ThinkingParser;

/// 聊天消息
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

/// 提示词选项（完整版，包含记忆数据）
#[derive(Debug, Clone)]
pub struct FullPromptOptions {
    pub deep_thinking: bool,
    pub memory_mode: MemoryMode,
    pub memories: Option<Memories>,
}

impl Default for FullPromptOptions {
    fn default() -> Self {
        Self {
            deep_thinking: false,
            memory_mode: MemoryMode::Balanced,
            memories: None,
        }
    }
}

/// 后处理结果
#[derive(Debug, Clone)]
pub struct ResponseResult {
    /// 清理后的回答（移除记忆指令）
    pub clean_content: String,
    /// 思考过程
    pub thinking: String,
    /// 提取的记忆操作
    pub memory_ops: Vec<MemoryOp>,
}

/// 聊天引擎：编排提示词、记忆、思考解析
pub struct ChatEngine;

impl ChatEngine {
    /// 构建完整系统提示词
    pub fn build_system_prompt(options: Option<&FullPromptOptions>) -> String {
        let default_opts = FullPromptOptions::default();
        let opts = options.unwrap_or(&default_opts);

        let prompt_opts = PromptOptions {
            deep_thinking: opts.deep_thinking,
            memory_mode: opts.memory_mode,
        };

        let base = SystemPrompt::build(Some(&prompt_opts));

        if let Some(memories) = &opts.memories {
            SystemPrompt::with_memories(base, memories)
        } else {
            base
        }
    }

    /// 构建发送给推理引擎的完整消息列表
    pub fn build_messages(options: BuildMessagesOptions) -> Vec<ChatMessage> {
        let mut messages = vec![ChatMessage::system(&options.system_prompt)];

        // 添加历史
        for msg in &options.history {
            messages.push(ChatMessage {
                role: msg.role.clone(),
                content: msg.content.clone(),
            });
        }

        // 添加当前用户消息
        if let Some(search) = &options.search_results {
            if !search.is_empty() {
                messages.push(ChatMessage::user(format!(
                    "[联网搜索结果]\n{}\n\n[用户问题]\n{}",
                    search, options.user_message
                )));
            } else {
                messages.push(ChatMessage::user(&options.user_message));
            }
        } else {
            messages.push(ChatMessage::user(&options.user_message));
        }

        messages
    }

    /// 后处理 AI 原始输出
    pub fn post_process(raw_content: &str, deep_thinking: bool) -> ResponseResult {
        let (thinking, content) = ThinkingParser::finalize(raw_content, deep_thinking);
        let clean_content = MemoryManager::clean_content(&content);
        let memory_ops = MemoryManager::extract_ops(raw_content);
        ResponseResult {
            clean_content,
            thinking,
            memory_ops,
        }
    }
}

/// build_messages 参数
pub struct BuildMessagesOptions {
    pub system_prompt: String,
    pub history: Vec<ChatMessage>,
    pub user_message: String,
    pub search_results: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::MemoryEntry;

    #[test]
    fn test_build_system_prompt_basic() {
        let prompt = ChatEngine::build_system_prompt(None);
        assert!(prompt.contains("FishAI"));
        assert!(prompt.contains("FishLab-ai"));
    }

    #[test]
    fn test_build_system_prompt_with_deep_thinking() {
        let opts = FullPromptOptions {
            deep_thinking: true,
            ..Default::default()
        };
        let prompt = ChatEngine::build_system_prompt(Some(&opts));
        assert!(prompt.contains("深度思考"));
    }

    #[test]
    fn test_build_system_prompt_with_memories() {
        let opts = FullPromptOptions {
            memories: Some(Memories {
                active: vec![MemoryEntry {
                    category: "personal".into(),
                    content: "喜欢猫".into(),
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let prompt = ChatEngine::build_system_prompt(Some(&opts));
        assert!(prompt.contains("[personal] 喜欢猫"));
    }

    #[test]
    fn test_build_messages_basic() {
        let opts = BuildMessagesOptions {
            system_prompt: "你是助手".into(),
            history: vec![
                ChatMessage::user("你好"),
                ChatMessage::assistant("你好！有什么可以帮你的？"),
            ],
            user_message: "今天天气如何？".into(),
            search_results: None,
        };
        let msgs = ChatEngine::build_messages(opts);
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[2].role, "assistant");
        assert_eq!(msgs[3].role, "user");
        assert!(msgs[3].content.contains("今天天气"));
    }

    #[test]
    fn test_build_messages_with_search() {
        let opts = BuildMessagesOptions {
            system_prompt: "系统提示".into(),
            history: vec![],
            user_message: "什么是 Rust？".into(),
            search_results: Some("Rust 是一种系统编程语言".into()),
        };
        let msgs = ChatEngine::build_messages(opts);
        assert_eq!(msgs.len(), 2); // system + user with search
        assert!(msgs[1].content.contains("联网搜索结果"));
        assert!(msgs[1].content.contains("什么是 Rust？"));
    }

    #[test]
    fn test_build_messages_empty_search() {
        let opts = BuildMessagesOptions {
            system_prompt: "系统提示".into(),
            history: vec![],
            user_message: "测试".into(),
            search_results: Some(String::new()),
        };
        let msgs = ChatEngine::build_messages(opts);
        assert_eq!(msgs.len(), 2);
        // 空搜索结果应该直接用 user_message
        assert_eq!(msgs[1].content, "测试");
    }

    #[test]
    fn test_post_process_no_thinking() {
        let result = ChatEngine::post_process("这是回答", false);
        assert_eq!(result.clean_content, "这是回答");
        assert_eq!(result.thinking, "");
        assert!(result.memory_ops.is_empty());
    }

    #[test]
    fn test_post_process_with_thinking() {
        let result = ChatEngine::post_process(
            "<thinkthink>先分析一下</thinkthink>最终答案",
            true,
        );
        assert_eq!(result.thinking, "先分析一下");
        assert_eq!(result.clean_content, "最终答案");
    }

    #[test]
    fn test_post_process_with_memory() {
        let result = ChatEngine::post_process(
            "我记住了。[MEM:active|personal|用户叫小明]很高兴认识你！",
            false,
        );
        assert!(!result.clean_content.contains("[MEM:"));
        assert_eq!(result.memory_ops.len(), 1);
        assert_eq!(result.memory_ops[0].content, "用户叫小明");
    }

    #[test]
    fn test_post_process_with_thinking_and_memory() {
        let result = ChatEngine::post_process(
            "<thinkthink>需要记住</thinkthink>好的。[MEM:persistent|preference|暗色主题]完成。",
            true,
        );
        assert_eq!(result.thinking, "需要记住");
        assert!(!result.clean_content.contains("[MEM:"));
        assert!(result.clean_content.contains("好的。完成。"));
        assert_eq!(result.memory_ops.len(), 1);
    }

    #[test]
    fn test_chat_message_constructors() {
        let s = ChatMessage::system("sys");
        assert_eq!(s.role, "system");
        let u = ChatMessage::user("usr");
        assert_eq!(u.role, "user");
        let a = ChatMessage::assistant("ast");
        assert_eq!(a.role, "assistant");
    }
}
