//! FishAI Engine — HTTP 服务入口
//!
//! 用法:
//!   cargo run              # 默认 127.0.0.1:8900
//!   FISHAI_PORT=9000 cargo run  # 自定义端口

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

    let config = ServerConfig {
        port: std::env::var("FISHAI_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(8900),
        host: std::env::var("FISHAI_HOST")
            .ok()
            .unwrap_or_else(|| "127.0.0.1".to_string()),
    };

    fishai_engine::http::run_server(config).await;
}
