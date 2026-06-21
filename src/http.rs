//! FishAI Engine HTTP 服务
//!
//! 提供独立 HTTP API，供 fishai-server (Node.js) 通过网络调用。
//! 包含三层接口：
//! - AI 逻辑层（采样 / 提示词 / 后处理 / 记忆 / 思考）
//! - 模型推理层（加载 GGUF / 推理 / 生成）
//! - 模型信息层（查看架构 / 张量列表）

use std::sync::{Arc, Mutex};
use axum::{
    extract::{Json, State},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

use crate::chat::{self, ChatMessage};
use crate::model::{GenerationConfig, ModelConfig, ModelWeights, GenerationState};
use crate::sampling::{Sampler, SamplerConfig};
use crate::thinking::ThinkingParser;

/// 服务器共享状态
#[derive(Debug, Clone)]
pub struct AppState {
    /// 已加载的模型（可选）
    pub model: Arc<Mutex<Option<Arc<ModelWeights>>>>,
}

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
    model_loaded: bool,
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

// ==================== 模型加载/推理 API ====================

#[derive(Debug, Deserialize)]
struct LoadModelRequest {
    /// GGUF 模型文件路径
    path: String,
}

#[derive(Debug, Serialize)]
struct LoadModelResponse {
    success: bool,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<ModelInfo>,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_embd: usize,
    vocab_size: usize,
    context_len: usize,
    n_ff: usize,
}

#[derive(Debug, Deserialize)]
struct GenerateRequest {
    /// 提示词文本
    prompt: String,
    /// 生成配置（可选）
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
}

#[derive(Debug, Serialize)]
struct GenerateResponse {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_prompt_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_generated_tokens: Option<usize>,
    message: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelStatusResponse {
    loaded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<ModelInfo>,
}

// ==================== 路由处理器 ====================

async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let model_loaded = state.model.lock().unwrap().is_some();
    Json(HealthResponse {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        engine: "fishai-engine (Rust)".into(),
        model_loaded,
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

// ==================== 模型路由处理器 ====================

async fn load_model(State(state): State<AppState>, Json(req): Json<LoadModelRequest>) -> Json<LoadModelResponse> {
    use crate::gguf::GGUFFile;

    // 1. 打开并解析 GGUF 文件
    let gguf = match GGUFFile::open(&req.path) {
        Ok(f) => f,
        Err(e) => {
            return Json(LoadModelResponse {
                success: false,
                message: format!("无法打开 GGUF 文件: {}", e),
                config: None,
            });
        }
    };

    // 2. 提取模型配置
    let config = match ModelConfig::from_gguf(&gguf) {
        Ok(c) => c,
        Err(e) => {
            return Json(LoadModelResponse {
                success: false,
                message: format!("无法解析模型配置: {}", e),
                config: None,
            });
        }
    };

    let info = ModelInfo {
        n_layers: config.n_layers,
        n_heads: config.n_heads,
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        n_embd: config.n_embd,
        vocab_size: config.vocab_size,
        context_len: config.context_len,
        n_ff: config.n_ff(),
    };

    // 3. 加载模型权重
    let model = match ModelWeights::load_from_gguf(&gguf, &config) {
        Ok(m) => m,
        Err(e) => {
            return Json(LoadModelResponse {
                success: false,
                message: format!("无法加载模型权重: {}", e),
                config: Some(info),
            });
        }
    };

    // 4. 存入共享状态
    {
        let mut model_guard = state.model.lock().unwrap();
        *model_guard = Some(Arc::new(model));
    }

    Json(LoadModelResponse {
        success: true,
        message: format!(
            "模型加载成功: {}层, {}头, {}维, {}词表",
            config.n_layers, config.n_heads, config.n_embd, config.vocab_size
        ),
        config: Some(info),
    })
}

async fn generate(State(state): State<AppState>, Json(req): Json<GenerateRequest>) -> Json<GenerateResponse> {
    // 获取已加载的模型
    let model = {
        let guard = state.model.lock().unwrap();
        match guard.as_ref() {
            Some(m) => Arc::clone(m),
            None => {
                return Json(GenerateResponse {
                    success: false,
                    text: None,
                    token_ids: None,
                    n_prompt_tokens: None,
                    n_generated_tokens: None,
                    message: Some("模型未加载，请先调用 POST /api/model/load".into()),
                });
            }
        }
    };

    // 构建生成配置
    let gen_config = GenerationConfig {
        max_tokens: req.max_tokens.unwrap_or(256),
        temperature: req.temperature.unwrap_or(0.7),
        top_k: req.top_k.unwrap_or(40),
        top_p: req.top_p.unwrap_or(0.95),
        repetition_penalty: req.repetition_penalty.unwrap_or(1.15),
        stop_token_ids: vec![],
    };

    // 创建生成状态
    let mut state = GenerationState::new((*model).clone(), gen_config);

    // 使用简易 tokenizer 将文本转为 token IDs（逐字符 fallback）
    // 实际使用中应该用 BpeTokenizer，这里做简单编码
    let prompt_tokens: Vec<u32> = req.prompt.chars().map(|c| c as u32).collect();
    let n_prompt = prompt_tokens.len();

    // 生成
    let generated = state.generate(&prompt_tokens);
    let n_gen = generated.len();

    // 将 token IDs 转回文本
    let text: String = generated.iter().map(|&id| {
        char::from_u32(id).unwrap_or('\u{FFFD}')
    }).collect();

    Json(GenerateResponse {
        success: true,
        text: Some(text),
        token_ids: Some(generated),
        n_prompt_tokens: Some(n_prompt),
        n_generated_tokens: Some(n_gen),
        message: None,
    })
}

async fn model_status(State(state): State<AppState>) -> Json<ModelStatusResponse> {
    let guard = state.model.lock().unwrap();
    match guard.as_ref() {
        Some(m) => {
            let c = &m.config;
            Json(ModelStatusResponse {
                loaded: true,
                config: Some(ModelInfo {
                    n_layers: c.n_layers,
                    n_heads: c.n_heads,
                    n_kv_heads: c.n_kv_heads,
                    head_dim: c.head_dim,
                    n_embd: c.n_embd,
                    vocab_size: c.vocab_size,
                    context_len: c.context_len,
                    n_ff: c.n_ff(),
                }),
            })
        }
        None => Json(ModelStatusResponse {
            loaded: false,
            config: None,
        }),
    }
}

// ==================== 应用构建 ====================

pub fn create_app() -> Router {
    let state = AppState {
        model: Arc::new(Mutex::new(None)),
    };
    Router::new()
        // 健康检查
        .route("/health", get(health_check))
        // AI 逻辑层
        .route("/api/sample", post(sample_token))
        .route("/api/prompt", post(build_prompt))
        .route("/api/post-process", post(post_process))
        .route("/api/thinking", post(parse_thinking))
        .route("/api/messages", post(build_messages))
        // 模型推理层
        .route("/api/model/load", post(load_model))
        .route("/api/model/generate", post(generate))
        .route("/api/model/status", get(model_status))
        .with_state(state)
        .layer(CorsLayer::permissive())
}

/// 启动 HTTP 服务
pub async fn run_server(config: ServerConfig) {
    let app = create_app();
    let addr = format!("{}:{}", config.host, config.port);
    println!("🐟 FishAI Engine v{} — Rust", env!("CARGO_PKG_VERSION"));
    println!("   Listening on http://{}", addr);
    println!("   Endpoints:");
    println!("     GET  /health              — 健康检查");
    println!("     POST /api/sample          — Token 采样");
    println!("     POST /api/prompt          — 构建系统提示");
    println!("     POST /api/post-process    — 后处理输出");
    println!("     POST /api/thinking        — 思考解析");
    println!("     POST /api/messages        — 构建消息列表");
    println!("     POST /api/model/load      — 加载 GGUF 模型");
    println!("     POST /api/model/generate   — 模型推理生成");
    println!("     GET  /api/model/status     — 模型状态查询");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
