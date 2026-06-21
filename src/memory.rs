//! FishAI 记忆管理器
//!
//! 从模型回答中提取/清理记忆指令。
//! 不操作数据库——由调用方（fishai-server）负责持久化。

use std::fmt;

/// 记忆操作动作
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MemoryAction {
    /// 用户记事本（每次对话可见）
    Active,
    /// 自动记忆（后台积累）
    Persistent,
    /// 更新已有记忆
    Update,
    /// 删除记忆
    Delete,
}

impl fmt::Display for MemoryAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Persistent => write!(f, "persistent"),
            Self::Update => write!(f, "update"),
            Self::Delete => write!(f, "delete"),
        }
    }
}

/// 单条记忆操作
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryOp {
    pub action: MemoryAction,
    pub category: Option<String>,
    pub content: String,
    pub old_key: Option<String>,
}

/// 记忆指令前缀
const MEM_PREFIX: &str = "[MEM:";
const MEM_CLOSE: &str = "]";

/// 合法分类
const VALID_CATEGORIES: &[&str] = &[
    "personal",
    "preference",
    "knowledge",
    "schedule",
    "general",
];

pub struct MemoryManager;

impl MemoryManager {
    /// 从模型回答中提取记忆指令
    ///
    /// 匹配格式: `[MEM:action|category|content]` 或 `[MEM:action|content]`
    pub fn extract_ops(content: &str) -> Vec<MemoryOp> {
        let mut ops = Vec::new();
        let mut search_from = 0;

        while let Some(start) = content[search_from..].find(MEM_PREFIX) {
            let abs_start = search_from + start;
            let inner_start = abs_start + MEM_PREFIX.len();

            // 找到最近的 ]
            if let Some(close_pos) = content[inner_start..].find(MEM_CLOSE) {
                let abs_end = inner_start + close_pos;
                let inner = &content[inner_start..abs_end];

                if let Some(op) = Self::parse_instruction(inner) {
                    ops.push(op);
                }

                search_from = abs_end + MEM_CLOSE.len();
            } else {
                // 没有闭括号，跳过
                search_from = inner_start;
            }
        }

        ops
    }

    /// 从回答中移除所有记忆指令（不展示给用户）
    pub fn clean_content(content: &str) -> String {
        let mut result = String::with_capacity(content.len());
        let mut search_from = 0;

        while let Some(start) = content[search_from..].find(MEM_PREFIX) {
            let abs_start = search_from + start;
            let inner_start = abs_start + MEM_PREFIX.len();

            // 复制 MEM 之前的文本
            result.push_str(&content[search_from..abs_start]);

            if let Some(close_pos) = content[inner_start..].find(MEM_CLOSE) {
                let abs_end = inner_start + close_pos + MEM_CLOSE.len();
                search_from = abs_end;
            } else {
                // 没有闭括号，保留原文
                result.push_str(&content[abs_start..]);
                search_from = content.len();
            }
        }

        // 复制剩余文本
        if search_from < content.len() {
            result.push_str(&content[search_from..]);
        }

        result.trim().to_string()
    }

    /// 格式化记忆列表用于注入系统提示
    pub fn format_for_prompt(memories: &[(String, String)]) -> String {
        memories
            .iter()
            .map(|(cat, content)| format!("[{}] {}", cat, content))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn parse_instruction(inner: &str) -> Option<MemoryOp> {
        let parts: Vec<&str> = inner.splitn(2, '|').collect();
        if parts.is_empty() {
            return None;
        }
        let action_str = parts[0];
        let payload = parts.get(1).copied().unwrap_or("");

        let action = match action_str {
            "active" => MemoryAction::Active,
            "persistent" => MemoryAction::Persistent,
            "update" => MemoryAction::Update,
            "delete" => MemoryAction::Delete,
            _ => return None,
        };

        match action {
            MemoryAction::Update => {
                let sub_parts: Vec<&str> = payload.splitn(2, '|').collect();
                Some(MemoryOp {
                    action,
                    category: None,
                    old_key: sub_parts.first().map(|s| s.to_string()),
                    content: sub_parts.get(1).map(|s| s.to_string()).unwrap_or_default(),
                })
            }
            MemoryAction::Delete => Some(MemoryOp {
                action,
                category: None,
                content: payload.to_string(),
                old_key: None,
            }),
            _ => {
                let sub_parts: Vec<&str> = payload.splitn(2, '|').collect();
                let first = sub_parts.first().copied().unwrap_or("");
                let has_category =
                    !first.is_empty() && VALID_CATEGORIES.contains(&first);

                Some(MemoryOp {
                    action,
                    category: if has_category {
                        Some(first.to_string())
                    } else {
                        None
                    },
                    content: if has_category {
                        sub_parts.get(1).copied().unwrap_or(payload).to_string()
                    } else {
                        payload.to_string()
                    },
                    old_key: None,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_active_memory() {
        let content = "记得了。[MEM:active|personal|用户喜欢猫]";
        let ops = MemoryManager::extract_ops(content);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].action, MemoryAction::Active);
        assert_eq!(ops[0].category.as_deref(), Some("personal"));
        assert_eq!(ops[0].content, "用户喜欢猫");
    }

    #[test]
    fn test_extract_persistent_memory() {
        let content = "好的。[MEM:persistent|preference|喜欢暗色主题]";
        let ops = MemoryManager::extract_ops(content);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].action, MemoryAction::Persistent);
        assert_eq!(ops[0].content, "喜欢暗色主题");
    }

    #[test]
    fn test_extract_update_memory() {
        let content = "[MEM:update|旧地址|新地址是上海]";
        let ops = MemoryManager::extract_ops(content);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].action, MemoryAction::Update);
        assert_eq!(ops[0].old_key.as_deref(), Some("旧地址"));
        assert_eq!(ops[0].content, "新地址是上海");
    }

    #[test]
    fn test_extract_delete_memory() {
        let content = "已删除。[MEM:delete|过期信息]";
        let ops = MemoryManager::extract_ops(content);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].action, MemoryAction::Delete);
        assert_eq!(ops[0].content, "过期信息");
    }

    #[test]
    fn test_extract_multiple_ops() {
        let content = "[MEM:active|personal|张三]和[MEM:persistent|preference|喜欢Python]";
        let ops = MemoryManager::extract_ops(content);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].action, MemoryAction::Active);
        assert_eq!(ops[1].action, MemoryAction::Persistent);
    }

    #[test]
    fn test_extract_no_category_falls_to_general() {
        let content = "[MEM:active|用户的名字是小明]";
        let ops = MemoryManager::extract_ops(content);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].category, None);
        assert_eq!(ops[0].content, "用户的名字是小明");
    }

    #[test]
    fn test_extract_invalid_category() {
        let content = "[MEM:active|invalid_cat|一些内容]";
        let ops = MemoryManager::extract_ops(content);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].category, None);
        assert_eq!(ops[0].content, "invalid_cat|一些内容");
    }

    #[test]
    fn test_extract_empty_content() {
        let content = "[MEM:active|]";
        let ops = MemoryManager::extract_ops(content);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].content, "");
    }

    #[test]
    fn test_extract_no_memories() {
        let content = "这是普通的回答，没有记忆指令。";
        let ops = MemoryManager::extract_ops(content);
        assert!(ops.is_empty());
    }

    #[test]
    fn test_clean_content_removes_memories() {
        let content = "你好！[MEM:active|personal|记住这个]很高兴认识你。";
        let cleaned = MemoryManager::clean_content(content);
        assert!(!cleaned.contains("[MEM:"));
        assert!(cleaned.contains("你好！"));
        assert!(cleaned.contains("很高兴认识你。"));
    }

    #[test]
    fn test_clean_content_no_memories() {
        let content = "这是干净的回答。";
        let cleaned = MemoryManager::clean_content(content);
        assert_eq!(cleaned, "这是干净的回答。");
    }

    #[test]
    fn test_clean_content_removes_all_types() {
        let content = "[MEM:active|x|y][MEM:persistent|a|b][MEM:update|c|d][MEM:delete|e]";
        let cleaned = MemoryManager::clean_content(content);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_content_trims() {
        let content = "  [MEM:active|general|测试]  ";
        let cleaned = MemoryManager::clean_content(content);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_format_for_prompt() {
        let memories = vec![
            ("personal".to_string(), "喜欢猫".to_string()),
            ("preference".to_string(), "暗色主题".to_string()),
        ];
        let formatted = MemoryManager::format_for_prompt(&memories);
        assert_eq!(formatted, "[personal] 喜欢猫\n[preference] 暗色主题");
    }

    #[test]
    fn test_format_for_prompt_empty() {
        let memories: Vec<(String, String)> = vec![];
        let formatted = MemoryManager::format_for_prompt(&memories);
        assert_eq!(formatted, "");
    }

    #[test]
    fn test_memory_action_display() {
        assert_eq!(format!("{}", MemoryAction::Active), "active");
        assert_eq!(format!("{}", MemoryAction::Delete), "delete");
    }
}
