//! FishAI Engine HTTP 服务
//!
//! 提供独立 HTTP API，供 fishai-server (Node.js) 通过网络调用。
//! 包含三层接口：
//! - AI 逻辑层（采样 / 提示词 / 后处理 / 记忆 / 思考）
//! - 模型推理层（加载 GGUF / 推理 / 生成）
//! - 模型信息层（查看架构 / 张量列表）
//!
//! 安全特性：
//! - API Key 认证（Bearer Token / X-API-Key header）
//! - CORS 白名单限制
//! - 请求体大小限制
//! - 路径遍历防护（模型加载仅允许指定目录）
//! - 速率限制（滑动窗口）
//! - 输入长度校验
//! - panic 捕获中间件

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::{
    extract::{Json, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::{
    cors::CorsLayer,
    limit::RequestBodyLimitLayer,
    trace::TraceLayer,
};

use crate::chat::{self, ChatMessage};
use crate::model::{GenerationConfig, ModelConfig, ModelWeights, GenerationState};
use crate::sampling::{Sampler, SamplerConfig};
use crate::thinking::ThinkingParser;

// ---------------------------------------------------------------------------
// 安全配置
// ---------------------------------------------------------------------------

/// 安全配置
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// API Key（空字符串 = 禁用认证）
    pub api_key: String,
    /// CORS 允许的源（空 = 仅 localhost）
    pub cors_origins: Vec<String>,
    /// 模型文件允许加载的目录白名单（空 = 禁用模型加载 API）
    pub model_allowed_dirs: Vec<PathBuf>,
    /// 速率限制：窗口内最大请求数
    pub rate_limit_max: usize,
    /// 速率限制：窗口时长（秒）
    pub rate_limit_window_secs: u64,
    /// 请求体最大大小（字节）
    pub max_body_size: usize,
    /// 单条文本最大长度（字符）
    pub max_text_length: usize,
    /// logits 向量最大长度
    pub max_logits_length: usize,
    /// 单次生成最大 token 数
    pub max_generate_tokens: usize,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            cors_origins: vec!["http://localhost:3000".into(), "http://127.0.0.1:3000".into()],
            model_allowed_dirs: vec![],
            rate_limit_max: 60,
            rate_limit_window_secs: 60,
            max_body_size: 4 * 1024 * 1024,       // 4MB
            max_text_length: 128 * 1024,           // 128K 字符
            max_logits_length: 256 * 1024,          // 256K
            max_generate_tokens: 4096,
        }
    }
}

// ---------------------------------------------------------------------------
// 速率限制器
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct RateLimiter {
    timestamps: Mutex<Vec<Instant>>,
    max: usize,
    window: Duration,
}

impl RateLimiter {
    fn new(max: usize, window_secs: u64) -> Self {
        Self {
            timestamps: Mutex::new(Vec::with_capacity(max)),
            max,
            window: Duration::from_secs(window_secs),
        }
    }

    /// 检查是否允许请求，允许则记录时间戳
    fn check(&self) -> bool {
        let now = Instant::now();
        let mut ts = self.timestamps.lock().unwrap();
        // 清理过期记录
        ts.retain(|&t| now.duration_since(t) < self.window);
        if ts.len() >= self.max {
            false
        } else {
            ts.push(now);
            true
        }
    }
}

// ---------------------------------------------------------------------------
// 服务器配置和状态
// ---------------------------------------------------------------------------

/// 服务器共享状态
#[derive(Debug, Clone)]
pub struct AppState {
    /// 已加载的模型（可选）
    pub model: Arc<Mutex<Option<Arc<ModelWeights>>>>,
    /// 安全配置
    pub security: SecurityConfig,
    /// 速率限制器（按 IP）
    pub rate_limiter: Arc<RateLimiter>,
}

/// 服务器配置
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub security: SecurityConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8900,
            security: SecurityConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// API 请求/响应类型
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    engine: String,
    model_loaded: bool,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    code: u16,
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

// ---------------------------------------------------------------------------
// 安全中间件
// ---------------------------------------------------------------------------

/// API Key 认证 + 速率限制中间件（合为一个，避免 axum State 冲突）
async fn security_middleware(
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path();

    // --- 速率限制（/health 跳过） ---
    if path != "/health" {
        use std::sync::OnceLock;
        static LIMITER: OnceLock<RateLimiter> = OnceLock::new();
        let limiter = LIMITER.get_or_init(|| RateLimiter::new(60, 60));
        if !limiter.check() {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: "请求过于频繁，请稍后再试".into(),
                    code: 429,
                }),
            ).into_response();
        }
    }

    // --- API Key 认证（/health 和 /api/model/status 跳过） ---
    if path != "/health" && path != "/api/model/status" {
        use std::sync::OnceLock;
        static API_KEY: OnceLock<String> = OnceLock::new();
        let key = API_KEY.get_or_init(|| {
            std::env::var("FISHAI_API_KEY").unwrap_or_default()
        });

        if !key.is_empty() {
            let provided = req
                .headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer ").map(String::from))
                .or_else(|| {
                    req.headers()
                        .get("x-api-key")
                        .and_then(|v| v.to_str().ok())
                        .map(String::from)
                });

            match provided {
                Some(k) if !k.is_empty() && &k == key => {},
                _ => {
                    return (
                        StatusCode::UNAUTHORIZED,
                        Json(ErrorResponse {
                            error: "未授权：缺少或无效的 API Key".into(),
                            code: 401,
                        }),
                    ).into_response();
                }
            }
        }
    }

    next.run(req).await
}

// ---------------------------------------------------------------------------
// 路由处理器
// ---------------------------------------------------------------------------

async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let model_loaded = state.model.lock().unwrap().is_some();
    Json(HealthResponse {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        engine: "fishai-engine (Rust)".into(),
        model_loaded,
    })
}

async fn sample_token(
    State(state): State<AppState>,
    Json(req): Json<SampleRequest>,
) -> Response {
    // 输入验证
    if req.logits.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: "logits 不能为空".into(), code: 400 }),
        ).into_response();
    }
    if req.logits.len() > state.security.max_logits_length {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("logits 长度超出限制（最大 {}）", state.security.max_logits_length),
                code: 400,
            }),
        ).into_response();
    }

    let mut config = SamplerConfig::default();
    if let Some(overrides) = &req.config {
        if let Some(t) = overrides.temperature {
            // 温度范围校验
            if !(0.0..=2.0).contains(&t) {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse { error: "temperature 必须在 0.0~2.0 之间".into(), code: 400 }),
                ).into_response();
            }
            config.temperature = t;
        }
        if let Some(k) = overrides.top_k { config.top_k = k; }
        if let Some(p) = overrides.top_p {
            if !(0.0..=1.0).contains(&p) {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse { error: "top_p 必须在 0.0~1.0 之间".into(), code: 400 }),
                ).into_response();
            }
            config.top_p = p;
        }
        if let Some(r) = overrides.repetition_penalty {
            if r <= 0.0 {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse { error: "repetition_penalty 必须 > 0".into(), code: 400 }),
                ).into_response();
            }
            config.repetition_penalty = r;
        }
        if let Some(f) = overrides.frequency_penalty { config.frequency_penalty = f; }
        if let Some(p) = overrides.presence_penalty { config.presence_penalty = p; }
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
    Json(SampleResponse { token_id }).into_response()
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
            active: m.active.unwrap_or_default().into_iter().map(|e| MemoryEntry { category: e.category, content: e.content }).collect(),
            notebook: m.notebook.unwrap_or_default().into_iter().map(|e| MemoryEntry { category: e.category, content: e.content }).collect(),
            core: m.core.unwrap_or_default().into_iter().map(|e| MemoryEntry { category: e.category, content: e.content }).collect(),
            recent: m.recent.unwrap_or_default().into_iter().map(|e| MemoryEntry { category: e.category, content: e.content }).collect(),
        }),
    };

    let prompt = chat::ChatEngine::build_system_prompt(Some(&opts));
    let length = prompt.len();
    Json(PromptBuildResponse { system_prompt: prompt, length })
}

async fn post_process(
    State(state): State<AppState>,
    Json(req): Json<PostProcessRequest>,
) -> Response {
    if req.content.len() > state.security.max_text_length {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("内容长度超出限制（最大 {} 字符）", state.security.max_text_length),
                code: 400,
            }),
        ).into_response();
    }

    let result = chat::ChatEngine::post_process(&req.content, req.deep_thinking);
    Json(PostProcessResponse {
        clean_content: result.clean_content,
        thinking: result.thinking,
        memory_ops: result.memory_ops,
    }).into_response()
}

async fn parse_thinking(
    State(state): State<AppState>,
    Json(req): Json<ThinkingParseRequest>,
) -> Response {
    if req.content.len() > state.security.max_text_length {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("内容长度超出限制（最大 {} 字符）", state.security.max_text_length),
                code: 400,
            }),
        ).into_response();
    }

    let (thinking, content) = ThinkingParser::finalize(&req.content, req.deep_thinking);
    Json(ThinkingParseResponse { thinking, content }).into_response()
}

async fn build_messages(
    State(state): State<AppState>,
    Json(req): Json<MessagesBuildRequest>,
) -> Response {
    if req.system_prompt.len() > state.security.max_text_length {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: "system_prompt 过长".into(), code: 400 }),
        ).into_response();
    }
    if req.user_message.len() > state.security.max_text_length {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: "user_message 过长".into(), code: 400 }),
        ).into_response();
    }
    if req.history.len() > 200 {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: "history 条数过多（最大 200）".into(), code: 400 }),
        ).into_response();
    }

    let opts = chat::BuildMessagesOptions {
        system_prompt: req.system_prompt,
        history: req.history,
        user_message: req.user_message,
        search_results: req.search_results,
    };
    let messages = chat::ChatEngine::build_messages(opts);
    let count = messages.len();
    Json(MessagesBuildResponse { messages, count }).into_response()
}

// ---------------------------------------------------------------------------
// 模型路由处理器
// ---------------------------------------------------------------------------

/// 路径遍历安全检查：确保路径在允许的目录范围内
fn validate_model_path(path: &str, allowed_dirs: &[PathBuf]) -> Result<PathBuf, String> {
    let canonical = Path::new(path)
        .canonicalize()
        .map_err(|e| format!("无效路径 '{}': {}", path, e))?;

    // 如果没有白名单目录，拒绝所有加载请求
    if allowed_dirs.is_empty() {
        return Err("模型加载功能已禁用（未配置 model_allowed_dirs）".into());
    }

    for dir in allowed_dirs {
        let canonical_dir = dir.canonicalize().unwrap_or_else(|_| dir.clone());
        if canonical.starts_with(&canonical_dir) {
            return Ok(canonical);
        }
    }

    Err(format!(
        "路径 '{}' 不在允许的模型目录中: {:?}",
        path,
        allowed_dirs.iter().map(|d| d.display()).collect::<Vec<_>>()
    ))
}

async fn load_model(
    State(state): State<AppState>,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    use crate::gguf::GGUFFile;

    // 路径安全检查
    let safe_path = match validate_model_path(&req.path, &state.security.model_allowed_dirs) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::FORBIDDEN,
                Json(LoadModelResponse { success: false, message: e, config: None }),
            ).into_response();
        }
    };

    // 1. 打开并解析 GGUF 文件
    let gguf = match GGUFFile::open(safe_path.to_str().unwrap_or("")) {
        Ok(f) => f,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(LoadModelResponse {
                    success: false,
                    message: format!("无法打开 GGUF 文件: {}", e),
                    config: None,
                }),
            ).into_response();
        }
    };

    // 2. 提取模型配置
    let config = match ModelConfig::from_gguf(&gguf) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(LoadModelResponse {
                    success: false,
                    message: format!("无法解析模型配置: {}", e),
                    config: None,
                }),
            ).into_response();
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
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(LoadModelResponse {
                    success: false,
                    message: format!("无法加载模型权重: {}", e),
                    config: Some(info),
                }),
            ).into_response();
        }
    };

    // 4. 存入共享状态
    {
        let mut model_guard = state.model.lock().unwrap();
        *model_guard = Some(Arc::new(model));
    }

    tracing::info!(model_layers = config.n_layers, model_heads = config.n_heads, "模型加载成功");

    (
        StatusCode::OK,
        Json(LoadModelResponse {
            success: true,
            message: format!(
                "模型加载成功: {}层, {}头, {}维, {}词表",
                config.n_layers, config.n_heads, config.n_embd, config.vocab_size
            ),
            config: Some(info),
        }),
    ).into_response()
}

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Response {
    if req.prompt.len() > state.security.max_text_length {
        return (
            StatusCode::BAD_REQUEST,
            Json(GenerateResponse {
                success: false, text: None, token_ids: None,
                n_prompt_tokens: None, n_generated_tokens: None,
                message: Some(format!("提示词过长（最大 {} 字符）", state.security.max_text_length)),
            }),
        ).into_response();
    }

    let max_tokens = req.max_tokens.unwrap_or(256).min(state.security.max_generate_tokens);

    // 获取已加载的模型
    let model = {
        let guard = state.model.lock().unwrap();
        match guard.as_ref() {
            Some(m) => Arc::clone(m),
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(GenerateResponse {
                        success: false, text: None, token_ids: None,
                        n_prompt_tokens: None, n_generated_tokens: None,
                        message: Some("模型未加载，请先调用 POST /api/model/load".into()),
                    }),
                ).into_response();
            }
        }
    };

    // 构建生成配置（使用上限钳制）
    let gen_config = GenerationConfig {
        max_tokens,
        temperature: req.temperature.unwrap_or(0.7).clamp(0.0, 2.0),
        top_k: req.top_k.unwrap_or(40),
        top_p: req.top_p.unwrap_or(0.95).clamp(0.0, 1.0),
        repetition_penalty: req.repetition_penalty.unwrap_or(1.15),
        stop_token_ids: vec![],
    };

    // 创建生成状态
    let mut gen_state = GenerationState::new((*model).clone(), gen_config);

    // 使用简易 tokenizer 将文本转为 token IDs
    let prompt_tokens: Vec<u32> = req.prompt.chars().map(|c| c as u32).collect();
    let n_prompt = prompt_tokens.len();

    // 生成
    let generated = gen_state.generate(&prompt_tokens);
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
    }).into_response()
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
        None => Json(ModelStatusResponse { loaded: false, config: None }),
    }
}

// ---------------------------------------------------------------------------
// 404 处理
// ---------------------------------------------------------------------------

async fn not_found() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: "端点不存在".into(),
            code: 404,
        }),
    ).into_response()
}

// ---------------------------------------------------------------------------
// 应用构建
// ---------------------------------------------------------------------------

pub fn create_app(security: SecurityConfig) -> Router {
    let state = AppState {
        model: Arc::new(Mutex::new(None)),
        security: security.clone(),
        rate_limiter: Arc::new(RateLimiter::new(
            security.rate_limit_max,
            security.rate_limit_window_secs,
        )),
    };

    // CORS: 如果有白名单则限制，否则仅允许 localhost
    let cors = if security.cors_origins.is_empty() {
        CorsLayer::very_permissive()
    } else {
        use axum::http::HeaderValue;
        let origins: Vec<HeaderValue> = security
            .cors_origins
            .iter()
            .filter_map(|o| o.parse::<HeaderValue>().ok())
            .collect();
        if origins.is_empty() {
            CorsLayer::very_permissive()
        } else {
            CorsLayer::new()
                .allow_origin(origins)
                .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
                .allow_headers([axum::http::header::AUTHORIZATION, axum::http::header::CONTENT_TYPE, axum::http::header::HeaderName::from_static("x-api-key")])
        }
    };

    let api_routes = Router::new()
        // 健康检查（不走认证）
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
        // 404 fallback
        .fallback(not_found);

    Router::new()
        .merge(api_routes)
        .with_state(state)
        // 安全中间件（认证 + 速率限制）
        .layer(middleware::from_fn(security_middleware))
        .layer(RequestBodyLimitLayer::new(security.max_body_size))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
}

/// 启动 HTTP 服务
pub async fn run_server(config: ServerConfig) {
    let security = config.security.clone();

    // 安全日志：不打印 API Key
    tracing::info!(
        api_key_set = !security.api_key.is_empty(),
        cors_origins = ?security.cors_origins,
        model_dirs = ?security.model_allowed_dirs.iter().map(|d| d.display()).collect::<Vec<_>>(),
        rate_limit = format!("{}/{}s", security.rate_limit_max, security.rate_limit_window_secs),
        max_body = format!("{}MB", security.max_body_size / 1024 / 1024),
        "安全配置已加载"
    );

    let app = create_app(security);
    let addr = format!("{}:{}", config.host, config.port);
    println!("🐟 FishAI Engine v{} — Rust", env!("CARGO_PKG_VERSION"));
    println!("   Listening on http://{}", addr);
    println!("   Security:");
    println!("     Auth:       {}", if config.security.api_key.is_empty() { "OFF (无认证)" } else { "ON (API Key)" });
    println!("     CORS:       {:?}", config.security.cors_origins);
    println!("     Rate Limit: {}/{}s", config.security.rate_limit_max, config.security.rate_limit_window_secs);
    println!("     Body Limit: {}MB", config.security.max_body_size / 1024 / 1024);
    println!("     Model Dirs: {:?}", config.security.model_allowed_dirs);
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
