//! FishAI 系统提示词
//!
//! 定义 AI 的身份、能力、行为规范。
//! 构建：基础提示 → + 深度思考模式 → + 记忆积极度 → + 记忆注入

/// 记忆模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryMode {
    /// 几乎任何用户信息都记
    Aggressive,
    /// 适度记录重要信息
    #[default]
    Balanced,
    /// 只在用户明确要求时记录
    Passive,
}

impl MemoryMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Aggressive => "aggressive",
            Self::Balanced => "balanced",
            Self::Passive => "passive",
        }
    }
}

/// 提示词构建选项
#[derive(Debug, Clone)]
pub struct PromptOptions {
    /// 是否开启深度思考
    pub deep_thinking: bool,
    /// 记忆模式
    pub memory_mode: MemoryMode,
}

impl Default for PromptOptions {
    fn default() -> Self {
        Self {
            deep_thinking: false,
            memory_mode: MemoryMode::Balanced,
        }
    }
}

/// 记忆条目
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryEntry {
    /// 分类
    pub category: String,
    /// 内容
    pub content: String,
}

/// 记忆分组
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Memories {
    pub active: Vec<MemoryEntry>,
    pub notebook: Vec<MemoryEntry>,
    pub core: Vec<MemoryEntry>,
    pub recent: Vec<MemoryEntry>,
}

pub struct SystemPrompt;

impl SystemPrompt {
    const BASE: &str = "你叫 FishAI，是 FishLab-ai 团队完全自研的 AI 助手。

## 核心架构

基于 LLaMA-style 架构，Rust 从零编写，融合 DeepSeek MLA（多头潜在注意力）、YaRN（长度外推）、FlashAttention-2、FishLab-ai 自研的 Fish-Ring 环形注意力、Fish-Scroll 滚动压缩，支持超长上下文，混合精度量化后约 12MB。

## 记忆系统

你有记忆能力。**记事本**（active）是用户和你都能写的便签，每次对话可见；**自动记忆**（persistent）在后台积累用户偏好和信息。上下文最多 8M Token，换对话重置。

记忆指令（不显示给用户）：
- 写记事本：[MEM:active|分类|内容]
- 自动记忆：[MEM:persistent|分类|内容]
- 更新：[MEM:update|旧关键词|新内容]
- 删除：[MEM:delete|内容关键词]
- 分类：personal/preference/knowledge/schedule/general

写记事本时机：用户明确要求记忆、重要配置/账号、待办事项。
自动记忆时机：个人信息、偏好习惯、反复出现的模式。

## 思维框架

1. **理解**：识别真实意图和隐含前提
2. **拆解**：复杂问题分解为子问题，确定解决顺序
3. **推理**：链式推理，每步写清依据，不确定时标注置信度
4. **验证**：sanity check，主动寻找反例
5. **呈现**：先结论后过程，关键处加粗，善用结构化格式

## 写代码

- 完整可运行，不省略关键代码
- 生产级质量：错误处理、类型安全、性能考量
- 多方案时对比优劣，推荐最佳
- 用 ```语言 代码块，关键行加注释，函数/类前加文档注释

## 回答问题

1. **直接答案**（1-2句）
2. **原理解释**
3. **具体例子**
4. **常见误区**（如有）
5. **延伸视角**

用类比解释抽象概念，常见误解主动指出，适当引用数据增强说服力。

## 风格

- 自信但谦逊，确定的果断说，不确定的标注
- 深度优先，技术话题专业，日常话题亲切
- 默认中文，用户用英文则英文回答
- 技术术语保留英文原文，首次附中文解释

## 记忆使用规则

自然地使用记忆信息，不要说\"根据我的记忆\"。记忆与当前对话冲突时以对话为准并更新。用户说\"记住这个\"时写进记事本。";

    const DEEP_THINKING_SUFFIX: &str =
        "\n\n## 深度思考模式\n用户开启了深度思考。先在 <thinkthink></thinkthink> \
         标签内详尽思考（分析→列举思路→评估→选最优→验证），再在标签外给出最终回答。";

    const MEMORY_AGGRESSIVE: &str =
        "\n\n## 记忆积极度：积极\n你非常主动地管理记忆。几乎任何关于用户的信息都要记\
         ——个人信息、偏好、习惯、提到的人名地名、项目信息等。宁可多记也不要漏记。\
         只要用户提到关于自己的任何信息，就记下来。";

    const MEMORY_BALANCED: &str =
        "\n\n## 记忆积极度：平衡\n你适度地管理记忆。重要的个人信息、明确的偏好、反复提到的习惯要记。\
         琐碎的细节可以忽略。用户明确让你记的东西一定要记。";

    const MEMORY_PASSIVE: &str =
        "\n\n## 记忆积极度：被动\n你只在被明确要求时才记录记忆。如果用户说\"记住这个\"、\"帮我记一下\"，\
         你就记。其他时候不主动记录。";

    /// 构建完整系统提示（基础 + 深度思考 + 记忆积极度）
    pub fn build(options: Option<&PromptOptions>) -> String {
        let default_opts = PromptOptions::default();
        let opts = options.unwrap_or(&default_opts);
        let mut prompt = String::from(Self::BASE);

        if opts.deep_thinking {
            prompt.push_str(Self::DEEP_THINKING_SUFFIX);
        }

        match opts.memory_mode {
            MemoryMode::Aggressive => prompt.push_str(Self::MEMORY_AGGRESSIVE),
            MemoryMode::Balanced => prompt.push_str(Self::MEMORY_BALANCED),
            MemoryMode::Passive => prompt.push_str(Self::MEMORY_PASSIVE),
        }

        prompt
    }

    /// 在基础提示上注入记忆
    pub fn with_memories(base_prompt: String, memories: &Memories) -> String {
        let mut prompt = base_prompt;

        if !memories.active.is_empty() {
            let text = Self::format_entries(&memories.active);
            prompt.push_str("\n\n## 记事本（用户标记的重要笔记，所有对话直接可见）\n");
            prompt.push_str(&text);
        }

        if !memories.notebook.is_empty() {
            let text = Self::format_entries(&memories.notebook);
            prompt.push_str("\n\n## 记事本（用户存的普通笔记，需要时参考）\n");
            prompt.push_str(&text);
        }

        if !memories.core.is_empty() {
            let text = Self::format_entries(&memories.core);
            prompt.push_str(
                "\n\n## 你牢牢记住的关于用户的信息（每次对话都会带着）\n",
            );
            prompt.push_str(&text);
        }

        if !memories.recent.is_empty() {
            let text = Self::format_entries(&memories.recent);
            prompt.push_str(
                "\n\n## 你模糊记得的关于用户的信息（可能有用的参考）\n",
            );
            prompt.push_str(&text);
        }

        prompt
    }

    fn format_entries(entries: &[MemoryEntry]) -> String {
        entries
            .iter()
            .map(|e| format!("[{}] {}", e.category, e.content))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_basic_prompt() {
        let prompt = SystemPrompt::build(None);
        assert!(prompt.contains("FishAI"));
        assert!(prompt.contains("FishLab-ai"));
        assert!(prompt.contains("Rust"));
        assert!(prompt.contains("记忆系统"));
        assert!(prompt.contains("思维框架"));
    }

    #[test]
    fn test_build_with_deep_thinking() {
        let opts = PromptOptions {
            deep_thinking: true,
            ..Default::default()
        };
        let prompt = SystemPrompt::build(Some(&opts));
        assert!(prompt.contains("深度思考模式"));
        assert!(prompt.contains("thinkthink"));
    }

    #[test]
    fn test_build_without_deep_thinking() {
        let opts = PromptOptions::default();
        let prompt = SystemPrompt::build(Some(&opts));
        assert!(!prompt.contains("深度思考模式"));
    }

    #[test]
    fn test_build_aggressive_memory() {
        let opts = PromptOptions {
            deep_thinking: false,
            memory_mode: MemoryMode::Aggressive,
        };
        let prompt = SystemPrompt::build(Some(&opts));
        assert!(prompt.contains("积极"));
        assert!(!prompt.contains("平衡"));
        assert!(!prompt.contains("被动"));
    }

    #[test]
    fn test_build_passive_memory() {
        let opts = PromptOptions {
            deep_thinking: false,
            memory_mode: MemoryMode::Passive,
        };
        let prompt = SystemPrompt::build(Some(&opts));
        assert!(prompt.contains("被动"));
    }

    #[test]
    fn test_with_memories_active() {
        let base = SystemPrompt::build(None);
        let memories = Memories {
            active: vec![MemoryEntry {
                category: "personal".into(),
                content: "喜欢猫".into(),
            }],
            ..Default::default()
        };
        let prompt = SystemPrompt::with_memories(base, &memories);
        assert!(prompt.contains("[personal] 喜欢猫"));
        assert!(prompt.contains("重要笔记"));
    }

    #[test]
    fn test_with_memories_core_and_recent() {
        let base = SystemPrompt::build(None);
        let memories = Memories {
            core: vec![MemoryEntry {
                category: "preference".into(),
                content: "Vim 用户".into(),
            }],
            recent: vec![MemoryEntry {
                category: "knowledge".into(),
                content: "在学习 Rust".into(),
            }],
            ..Default::default()
        };
        let prompt = SystemPrompt::with_memories(base, &memories);
        assert!(prompt.contains("[preference] Vim 用户"));
        assert!(prompt.contains("[knowledge] 在学习 Rust"));
        assert!(prompt.contains("牢牢记住"));
        assert!(prompt.contains("模糊记得"));
    }

    #[test]
    fn test_with_empty_memories_no_extra_sections() {
        let base = SystemPrompt::build(None);
        let prompt_before = base.len();
        let memories = Memories::default();
        let prompt = SystemPrompt::with_memories(base, &memories);
        // 空记忆不应该增加任何内容
        assert_eq!(prompt.len(), prompt_before);
    }

    #[test]
    fn test_prompt_size_reasonable() {
        let prompt = SystemPrompt::build(None);
        assert!(prompt.len() < 3000, "基础提示应该 < 3000 字符");
    }

    #[test]
    fn test_memory_mode_default() {
        assert_eq!(MemoryMode::default(), MemoryMode::Balanced);
    }
}
