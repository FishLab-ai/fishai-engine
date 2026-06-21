//! FishAI Engine HTTP 服务
//!
//! 提供独立 HTTP API，供 fishai-server (Node.js) 通过网络调用。

use axum::{
    extract::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

use crate::chat::{self, ChatMessage};
use crate::sampling::{Sampler, SamplerConfig};
use crate::thinking::ThinkingParser;

/// 服务器配置
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8900,
        }
    }
}

// ==================== API 请求/响应类型 ====================

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    engine: String,
}

#[derive(Debug, Deserialize)]
struct SampleRequest {
    logits: Vec<f32>,
    config: Option<SamplerConfigOverride>,
    token_freq: Option<Vec<usize>>,
    token_present: Option<Vec<bool>>,
}

#[derive(Debug, Deserialize)]
struct SamplerConfigOverride {
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
}

#[derive(Debug, Serialize)]
struct SampleResponse {
    token_id: usize,
}

#[derive(Debug, Deserialize)]
struct PromptBuildRequest {
    deep_thinking: Option<bool>,
    memory_mode: Option<String>,
    memories: Option<PromptMemoriesInput>,
}

#[derive(Debug, Deserialize)]
struct PromptMemoriesInput {
    active: Option<Vec<MemoryEntryInput>>,
    notebook: Option<Vec<MemoryEntryInput>>,
    core: Option<Vec<MemoryEntryInput>>,
    recent: Option<Vec<MemoryEntryInput>>,
}

#[derive(Debug, Deserialize)]
struct MemoryEntryInput {
    category: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct PromptBuildResponse {
    system_prompt: String,
    length: usize,
}

#[derive(Debug, Deserialize)]
struct PostProcessRequest {
    content: String,
    deep_thinking: bool,
}

#[derive(Debug, Serialize)]
struct PostProcessResponse {
    clean_content: String,
    thinking: String,
    memory_ops: Vec<crate::memory::MemoryOp>,
}

#[derive(Debug, Deserialize)]
struct ThinkingParseRequest {
    content: String,
    deep_thinking: bool,
}

#[derive(Debug, Serialize)]
struct ThinkingParseResponse {
    thinking: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct MessagesBuildRequest {
    system_prompt: String,
    history: Vec<ChatMessage>,
    user_message: String,
    search_results: Option<String>,
}

#[derive(Debug, Serialize)]
struct MessagesBuildResponse {
    messages: Vec<ChatMessage>,
    count: usize,
}

// ==================== 路由处理器 ====================

async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        engine: "fishai-engine (Rust)".into(),
    })
}

async fn sample_token(Json(req): Json<SampleRequest>) -> Json<SampleResponse> {
    let mut config = SamplerConfig::default();
    if let Some(overrides) = &req.config {
        if let Some(t) = overrides.temperature {
            config.temperature = t;
        }
        if let Some(k) = overrides.top_k {
            config.top_k = k;
        }
        if let Some(p) = overrides.top_p {
            config.top_p = p;
        }
        if let Some(r) = overrides.repetition_penalty {
            config.repetition_penalty = r;
        }
        if let Some(f) = overrides.frequency_penalty {
            config.frequency_penalty = f;
        }
        if let Some(p) = overrides.presence_penalty {
            config.presence_penalty = p;
        }
    }

    let mut sampler = Sampler::new(config);
    if let Some(freq) = req.token_freq {
        for (id, count) in freq.iter().enumerate() {
            for _ in 0..*count {
                sampler.observe_token(id);
            }
        }
    }
    if let Some(present) = req.token_present {
        for (id, &is_present) in present.iter().enumerate() {
            if is_present {
                sampler.observe_token(id);
            }
        }
    }

    let token_id = sampler.sample(&req.logits);
    Json(SampleResponse { token_id })
}

async fn build_prompt(Json(req): Json<PromptBuildRequest>) -> Json<PromptBuildResponse> {
    use crate::prompt::{MemoryEntry, Memories, MemoryMode};

    let opts = crate::chat::FullPromptOptions {
        deep_thinking: req.deep_thinking.unwrap_or(false),
        memory_mode: match req.memory_mode.as_deref() {
            Some("aggressive") => MemoryMode::Aggressive,
            Some("passive") => MemoryMode::Passive,
            _ => MemoryMode::Balanced,
        },
        memories: req.memories.map(|m| Memories {
            active: m
                .active
                .unwrap_or_default()
                .into_iter()
                .map(|e| MemoryEntry {
                    category: e.category,
                    content: e.content,
                })
                .collect(),
            notebook: m
                .notebook
                .unwrap_or_default()
                .into_iter()
                .map(|e| MemoryEntry {
                    category: e.category,
                    content: e.content,
                })
                .collect(),
            core: m
                .core
                .unwrap_or_default()
                .into_iter()
                .map(|e| MemoryEntry {
                    category: e.category,
                    content: e.content,
                })
                .collect(),
            recent: m
                .recent
                .unwrap_or_default()
                .into_iter()
                .map(|e| MemoryEntry {
                    category: e.category,
                    content: e.content,
                })
                .collect(),
        }),
    };

    let prompt = chat::ChatEngine::build_system_prompt(Some(&opts));
    let length = prompt.len();
    Json(PromptBuildResponse {
        system_prompt: prompt,
        length,
    })
}

async fn post_process(Json(req): Json<PostProcessRequest>) -> Json<PostProcessResponse> {
    let result = chat::ChatEngine::post_process(&req.content, req.deep_thinking);
    Json(PostProcessResponse {
        clean_content: result.clean_content,
        thinking: result.thinking,
        memory_ops: result.memory_ops,
    })
}

async fn parse_thinking(Json(req): Json<ThinkingParseRequest>) -> Json<ThinkingParseResponse> {
    let (thinking, content) = ThinkingParser::finalize(&req.content, req.deep_thinking);
    Json(ThinkingParseResponse { thinking, content })
}

async fn build_messages(Json(req): Json<MessagesBuildRequest>) -> Json<MessagesBuildResponse> {
    let opts = chat::BuildMessagesOptions {
        system_prompt: req.system_prompt,
        history: req.history,
        user_message: req.user_message,
        search_results: req.search_results,
    };
    let messages = chat::ChatEngine::build_messages(opts);
    let count = messages.len();
    Json(MessagesBuildResponse { messages, count })
}

// ==================== 应用构建 ====================

pub fn create_app() -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/api/sample", post(sample_token))
        .route("/api/prompt", post(build_prompt))
        .route("/api/post-process", post(post_process))
        .route("/api/thinking", post(parse_thinking))
        .route("/api/messages", post(build_messages))
        .layer(CorsLayer::permissive())
}

/// 启动 HTTP 服务
pub async fn run_server(config: ServerConfig) {
    let app = create_app();
    let addr = format!("{}:{}", config.host, config.port);
    println!("🐟 FishAI Engine v{} — Rust", env!("CARGO_PKG_VERSION"));
    println!("   Listening on http://{}", addr);
    println!("   Endpoints:");
    println!("     GET  /health         — 健康检查");
    println!("     POST /api/sample     — Token 采样");
    println!("     POST /api/prompt     — 构建系统提示");
    println!("     POST /api/post-process — 后处理输出");
    println!("     POST /api/thinking   — 思考解析");
    println!("     POST /api/messages   — 构建消息列表");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
