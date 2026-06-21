//! FishAI Engine — HTTP 服务入口
//!
//! 用法:
//!   cargo run                                    # 默认 127.0.0.1:8900
//!   FISHAI_PORT=9000 cargo run                   # 自定义端口
//!   FISHAI_API_KEY=secret cargo run               # 启用 API Key 认证
//!   FISHAI_MODEL_DIR=/path/to/models cargo run    # 允许加载该目录下的模型
//!   FISHAI_CORS_ORIGINS=http://a.com,http://b.com cargo run  # CORS 白名单
//!   FISHAI_RATE_LIMIT=100:60 cargo run            # 100请求/60秒
//!   RUST_LOG=debug cargo run                      # 调试日志

use std::path::PathBuf;

use fishai_engine::http::SecurityConfig;
use fishai_engine::http::ServerConfig;

#[tokio::main]
async fn main() {
    // 初始化日志
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".parse().unwrap()),
        )
        .init();

    let security = SecurityConfig {
        api_key: std::env::var("FISHAI_API_KEY").unwrap_or_default(),
        cors_origins: parse_list_env("FISHAI_CORS_ORIGINS"),
        model_allowed_dirs: parse_path_list_env("FISHAI_MODEL_DIR"),
        rate_limit_max: parse_rate_env("FISHAI_RATE_LIMIT").0,
        rate_limit_window_secs: parse_rate_env("FISHAI_RATE_LIMIT").1,
        max_body_size: std::env::var("FISHAI_MAX_BODY_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(4 * 1024 * 1024),
        ..Default::default()
    };

    let config = ServerConfig {
        port: std::env::var("FISHAI_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(8900),
        host: std::env::var("FISHAI_HOST")
            .ok()
            .unwrap_or_else(|| "127.0.0.1".to_string()),
        security,
    };

    fishai_engine::http::run_server(config).await;
}

/// 解析 "val1,val2,val3" 格式的环境变量为 Vec<String>
fn parse_list_env(key: &str) -> Vec<String> {
    std::env::var(key)
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// 解析路径列表环境变量
fn parse_path_list_env(key: &str) -> Vec<PathBuf> {
    std::env::var(key)
        .unwrap_or_default()
        .split(':')
        .map(|s| PathBuf::from(s.trim()))
        .filter(|p| !p.as_os_str().is_empty())
        .collect()
}

/// 解析速率限制 "100:60" → (100, 60)
fn parse_rate_env(key: &str) -> (usize, u64) {
    std::env::var(key)
        .ok()
        .and_then(|v| {
            let parts: Vec<&str> = v.split(':').collect();
            if parts.len() == 2 {
                Some((
                    parts[0].parse().ok()?,
                    parts[1].parse().ok()?,
                ))
            } else {
                None
            }
        })
        .unwrap_or((60, 60))
}
