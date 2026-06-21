//! FishAI 深度思考解析器
//!
//! 解析 AI 回答中的 `<thinkthink>...</thinkthink>` 或 `<LMTHINK>...</LMTHINK>` 标签。
//! 支持流式解析——边生成边分离思考过程和最终回答。

use std::fmt;

/// 解析事件
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseEvent {
    /// 思考过程
    Thinking(String),
    /// 最终回答
    Content(String),
}

impl fmt::Display for ParseEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Thinking(s) => write!(f, "[thinking] {}", s),
            Self::Content(s) => write!(f, "[content] {}", s),
        }
    }
}

/// 标签对
#[derive(Debug, Clone)]
struct TagPair {
    open_idx: Option<usize>,
    open_tag_len: usize,
    close_idx: Option<usize>,
    close_tag_len: usize,
}

impl TagPair {
}

/// 深度思考解析器（有状态，支持流式）
pub struct ThinkingParser {
    thinking_done: bool,
    last_sent_thinking_len: usize,
    buffered_content: String,
}

impl Default for ThinkingParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ThinkingParser {
    pub fn new() -> Self {
        Self {
            thinking_done: false,
            last_sent_thinking_len: 0,
            buffered_content: String::new(),
        }
    }

    /// 流式解析：传入完整累积内容 + 新 chunk，返回分离后的事件
    pub fn parse(&mut self, full_content: &str, chunk: &str, deep_thinking: bool) -> Vec<ParseEvent> {
        // 非深度思考模式：直接透传
        if !deep_thinking {
            if chunk.is_empty() {
                return vec![];
            }
            return vec![ParseEvent::Content(chunk.to_string())];
        }

        // 思考已结束：直接透传
        if self.thinking_done {
            if chunk.is_empty() {
                return vec![];
            }
            return vec![ParseEvent::Content(chunk.to_string())];
        }

        let tags = Self::find_tag_pair(full_content);

        // 还没出现开标签
        if tags.open_idx.is_none() {
            return self.handle_no_open_tag(full_content, chunk);
        }

        // 清除之前的 buffer
        self.buffered_content.clear();

        // 开标签出现但闭标签还没——流式输出思考
        if tags.close_idx.is_none() {
            return self.handle_streaming_thinking(
                full_content,
                tags.open_idx.unwrap(),
                tags.open_tag_len,
            );
        }

        // 闭标签出现——最终状态
        self.thinking_done = true;
        let mut events = Vec::new();

        let think_content =
            &full_content[tags.open_idx.unwrap() + tags.open_tag_len..tags.close_idx.unwrap()];
        if think_content.len() > self.last_sent_thinking_len {
            let new_part = &think_content[self.last_sent_thinking_len..];
            self.last_sent_thinking_len = think_content.len();
            events.push(ParseEvent::Thinking(new_part.to_string()));
        }

        let after_think = &full_content[tags.close_idx.unwrap() + tags.close_tag_len..];
        if !after_think.is_empty() {
            events.push(ParseEvent::Content(after_think.to_string()));
        }

        events
    }

    /// 最终处理：一次性分离思考和回答（流结束后调用）
    pub fn finalize(full_content: &str, deep_thinking: bool) -> (String, String) {
        if !deep_thinking {
            return (String::new(), full_content.to_string());
        }

        let (open_idx, open_tag_len, close_idx, close_tag_len) =
            Self::find_first_tag_pair(full_content);

        if open_idx.is_none() {
            return (String::new(), full_content.to_string());
        }

        let oi = open_idx.unwrap();
        let otl = open_tag_len;

        match close_idx {
            None => (
                full_content[oi + otl..].to_string(),
                String::new(),
            ),
            Some(ci) => (
                full_content[oi + otl..ci].to_string(),
                full_content[ci + close_tag_len..].trim().to_string(),
            ),
        }
    }

    /// 重置解析器（新对话时调用）
    pub fn reset(&mut self) {
        self.thinking_done = false;
        self.last_sent_thinking_len = 0;
        self.buffered_content.clear();
    }

    /// 查找标签对（支持两种标签风格）
    fn find_tag_pair(content: &str) -> TagPair {
        let open_tt = content.find("<thinkthink>");
        let open_lm = content.find("<LMTHINK>");
        let close_tt = content.find("</thinkthink>");
        let close_lm = content.find("</LMTHINK>");

        let (open_idx, open_tag_len) = match (open_tt, open_lm) {
            (Some(i), Some(j)) => if i <= j { (Some(i), "<thinkthink>".len()) } else { (Some(j), "<LMTHINK>".len()) },
            (Some(i), None) => (Some(i), "<thinkthink>".len()),
            (None, Some(j)) => (Some(j), "<LMTHINK>".len()),
            (None, None) => (None, 0),
        };

        let (close_idx, close_tag_len) = match (close_tt, close_lm) {
            (Some(i), Some(j)) => if i <= j { (Some(i), "</thinkthink>".len()) } else { (Some(j), "</LMTHINK>".len()) },
            (Some(i), None) => (Some(i), "</thinkthink>".len()),
            (None, Some(j)) => (Some(j), "</LMTHINK>".len()),
            (None, None) => (None, 0),
        };

        TagPair {
            open_idx,
            open_tag_len,
            close_idx,
            close_tag_len,
        }
    }

    fn find_first_tag_pair(content: &str) -> (Option<usize>, usize, Option<usize>, usize) {
        let tags = Self::find_tag_pair(content);
        (
            tags.open_idx,
            tags.open_tag_len,
            tags.close_idx,
            tags.close_tag_len,
        )
    }

    fn handle_no_open_tag(&mut self, full_content: &str, chunk: &str) -> Vec<ParseEvent> {
        if Self::could_be_partial_tag(full_content) {
            self.buffered_content = full_content.to_string();
            return vec![];
        }
        self.buffered_content.clear();
        if chunk.is_empty() {
            return vec![];
        }
        vec![ParseEvent::Content(chunk.to_string())]
    }

    fn handle_streaming_thinking(
        &mut self,
        full_content: &str,
        open_idx: usize,
        open_tag_len: usize,
    ) -> Vec<ParseEvent> {
        let think_content = &full_content[open_idx + open_tag_len..];
        if think_content.len() > self.last_sent_thinking_len {
            let new_part = think_content[self.last_sent_thinking_len..].to_string();
            self.last_sent_thinking_len = think_content.len();
            vec![ParseEvent::Thinking(new_part)]
        } else {
            vec![]
        }
    }

    fn could_be_partial_tag(s: &str) -> bool {
        const TAGS: &[&str] = &[
            "<thinkthink>",
            "</thinkthink>",
            "<LMTHINK>",
            "</LMTHINK>",
        ];
        for tag in TAGS {
            for len in 1..=tag.len() {
                if s.ends_with(&tag[..len]) {
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finalize_no_tags() {
        let (thinking, content) = ThinkingParser::finalize("普通回答", false);
        assert_eq!(thinking, "");
        assert_eq!(content, "普通回答");
    }

    #[test]
    fn test_finalize_no_tags_deep_thinking() {
        let (thinking, content) = ThinkingParser::finalize("普通回答", true);
        assert_eq!(thinking, "");
        assert_eq!(content, "普通回答");
    }

    #[test]
    fn test_finalize_with_thinkthink_tags() {
        let input = "<thinkthink>思考内容</thinkthink>最终回答";
        let (thinking, content) = ThinkingParser::finalize(input, true);
        assert_eq!(thinking, "思考内容");
        assert_eq!(content, "最终回答");
    }

    #[test]
    fn test_finalize_with_lmthink_tags() {
        let input = "<LMTHINK>LM思考</LMTHINK>LM回答";
        let (thinking, content) = ThinkingParser::finalize(input, true);
        assert_eq!(thinking, "LM思考");
        assert_eq!(content, "LM回答");
    }

    #[test]
    fn test_finalize_open_tag_no_close() {
        let input = "<thinkthink>还没结束的思考";
        let (thinking, content) = ThinkingParser::finalize(input, true);
        assert_eq!(thinking, "还没结束的思考");
        assert_eq!(content, "");
    }

    #[test]
    fn test_finalize_empty_content() {
        let (thinking, content) = ThinkingParser::finalize("", true);
        assert_eq!(thinking, "");
        assert_eq!(content, "");
    }

    #[test]
    fn test_finalize_multiline() {
        let input = "<thinkthink>第一步\n第二步\n第三步</thinkthink>最终答案";
        let (thinking, content) = ThinkingParser::finalize(input, true);
        assert_eq!(thinking, "第一步\n第二步\n第三步");
        assert_eq!(content, "最终答案");
    }

    #[test]
    fn test_streaming_parse_no_deep_thinking() {
        let mut parser = ThinkingParser::new();
        let events = parser.parse("hello world", "hello world", false);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], ParseEvent::Content("hello world".to_string()));
    }

    #[test]
    fn test_streaming_parse_deep_thinking_returns_events() {
        let mut parser = ThinkingParser::new();
        // 模拟流式输出
        let events = parser.parse("<thinkthink>思", "思", true);
        // "思" 是 thinkthink 内部的内容
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], ParseEvent::Thinking("思".to_string()));
    }

    #[test]
    fn test_streaming_parse_deep_thinking_done() {
        let mut parser = ThinkingParser::new();
        let full = "<thinkthink>思考完了</thinkthink>回答内容";
        let events = parser.parse(full, "回答内容", true);
        // 闭标签已出现
        assert!(!events.is_empty());
        assert!(events.iter().any(|e| matches!(e, ParseEvent::Content(_))));
    }

    #[test]
    fn test_streaming_empty_chunk() {
        let mut parser = ThinkingParser::new();
        let events = parser.parse("部分内容", "", true);
        assert!(events.is_empty());
    }

    #[test]
    fn test_reset() {
        let mut parser = ThinkingParser::new();
        parser.thinking_done = true;
        parser.last_sent_thinking_len = 100;
        parser.buffered_content = "buffer".to_string();
        parser.reset();
        assert!(!parser.thinking_done);
        assert_eq!(parser.last_sent_thinking_len, 0);
        assert!(parser.buffered_content.is_empty());
    }

    #[test]
    fn test_could_be_partial_tag() {
        assert!(ThinkingParser::could_be_partial_tag("<think"));
        assert!(ThinkingParser::could_be_partial_tag("<thi"));
        assert!(ThinkingParser::could_be_partial_tag("<LMTHINK>"));
        assert!(ThinkingParser::could_be_partial_tag("</LM"));
        assert!(!ThinkingParser::could_be_partial_tag("普通文本"));
        assert!(!ThinkingParser::could_be_partial_tag(""));
    }

    #[test]
    fn test_finalize_preserves_whitespace() {
        let input = "<thinkthink>思考</thinkthink>  回答  ";
        let (thinking, content) = ThinkingParser::finalize(input, true);
        assert_eq!(thinking, "思考");
        // content 被 trim
        assert_eq!(content, "回答");
    }
}
