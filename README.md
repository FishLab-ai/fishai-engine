# 🐟 FishAI Engine v3

> FishLab-ai 自研 GPT 推理引擎 — Rust 编写，LLaMA-style 架构，KV Cache，BPE 分词器，多尺寸模型

## v3 重大升级

| 特性 | v2 | v3 |
|------|----|----|
| KV Cache | ❌ 每步重算全部序列 | ✅ O(n) 增量推理，100-500× 加速 |
| 分词器 | ❌ 假 BPE (字节级, 260 词) | ✅ 真正 BPE (HuggingFace, 32K 词) |
| 流式输出 | ❌ 假流式 (一次返回全部) | ✅ 真正 SSE token-by-token |
| 采样 | ❌ 仅 temperature | ✅ temperature + top-k + top-p |
| 模型尺寸 | ❌ 仅 34M | ✅ 34M / 400M / 1.5B 三档 |
| RoPE 缩放 | ❌ 固定 512 tokens | ✅ Linear + YaRN 上下文扩展 |
| 量化 | ❌ 简单 INT4/INT8 | ✅ GPTQ 风格分组量化 (128 元素/组) |
| 权重格式 | ❌ JSON (10-100× 冗余) | ✅ 二进制格式 (.fq3) |
| 基准测试 | ❌ 无 | ✅ TTFT / tokens/s / 内存 |
| INT8 bug | ❌ 反量化索引错误 | ✅ 已修复 |

## 模型配置

### FishAI-S (~34M 参数, ~12MB 量化)
- d_model: 512, n_heads: 8, n_kv_heads: 4, n_layers: 6, d_ff: 1408
- 定位: 嵌入式/边缘设备推理

### FishAI-M (~400M 参数, ~150MB 量化)
- d_model: 896, n_heads: 14, n_kv_heads: 2, n_layers: 24, d_ff: 4864
- 定位: 对标 Qwen2.5-0.5B，目标 MMLU > 40

### FishAI-L (~1.5B 参数, ~500MB 量化)
- d_model: 1536, n_heads: 12, n_kv_heads: 4, n_layers: 28, d_ff: 8960
- 定位: 对标 Qwen2.5-1.5B，目标 MMLU > 55

## 项目结构

```
src/
├── main.rs       # 入口：加载权重+分词器，启动 HTTP 服务器
├── model.rs      # 模型架构 (GQA+RoPE+SwiGLU+KV Cache+多尺寸配置)
├── quantize.rs   # 分组量化 (GroupQuant4/INT8/FP16) + 二进制序列化
├── tokenizer.rs  # BPE 分词器 (HuggingFace tokenizers) + 字节级回退
├── api.rs        # HTTP API (真 SSE 流式, top-k/top-p 参数)
├── bench.rs      # 基准测试 (TTFT, tokens/s, 内存)
└── lib.rs        # 模块声明
```

## 量化策略

| 层类型 | 量化方式 | 精度 | 原因 |
|--------|----------|------|------|
| Token Embedding | FP16 | 2 bytes/param | 查表操作，精度敏感 |
| RMSNorm γ | FP16 | 2 bytes/param | 归一化参数，精度敏感 |
| Q/K 投影 | INT8 (per-channel) | 1 byte/param | 注意力精度敏感 |
| V/O 投影 | GroupQuant4 (128/组) | 0.5 byte/param | 中等精度需求 |
| FFN | GroupQuant4 (128/组) | 0.5 byte/param | 参数量大，精度容错高 |

## 路线图

- [ ] 训练 FishAI-M (400M)，目标 MMLU > 40
- [ ] 训练 FishAI-L (1.5B)，目标 MMLU > 55
- [ ] 知识蒸馏 + DPO 对齐
- [ ] 多 GPU 推理

## 许可证

MIT License - FishLab-ai
