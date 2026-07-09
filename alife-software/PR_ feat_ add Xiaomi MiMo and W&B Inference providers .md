# PR: feat: add Xiaomi MiMo and W&B Inference providers (#25, #26)

---

## Summary

This PR adds two new provider integrations to Free Claude Code, bringing the total supported provider count from 24 to **26**.

| Provider | ID | Transport | Auth env var |
|---|---|---|---|
| **Xiaomi MiMo** | `xiaomimimo` | `anthropic_messages` (native) | `XIAOMIMIMO_API_KEY` |
| **W&B Inference** | `wandb_inference` | `openai_chat` | `WANDB_API_KEY` |

---

## Motivation

### Xiaomi MiMo

Xiaomi released [MiMo](https://github.com/XiaomiMiMo/MiMo) as an open-weight reasoning model family trained with a process-reward signal. The hosted API at `api.xiaomimimo.com` speaks the **native Anthropic Messages format**, making it a natural fit for FCC's `AnthropicMessagesTransport`. The model's strong chain-of-thought reasoning capability (comparable to DeepSeek-R1 on several benchmarks) makes it a compelling free/cheap alternative for Opus-tier routing.

A **Token Plan** endpoint (`token-plan-cn.xiaomimimo.com`) is also supported via the `XIAOMIMIMO_BASE_URL` override.

### W&B Inference

[Weights & Biases Inference](https://wandb.ai/site/inference) provides access to a curated set of open-source models (DeepSeek-V3.1, Qwen3-Coder-480B, Llama-3.3-70B, and others) hosted on CoreWeave infrastructure. Pricing is typically lower than going to the model providers directly. The endpoint speaks the **OpenAI Chat Completions format**, mapping cleanly onto FCC's `OpenAIChatTransport`. W&B's `reasoning_content` field in streaming responses is translated to FCC's `thinking_delta` SSE events when `enable_thinking=True`.

---

## Changes

### New files

| File | Purpose |
|---|---|
| `providers/xiaomimimo/__init__.py` | Package export |
| `providers/xiaomimimo/client.py` | `XiaomiMiMoProvider` — Anthropic Messages transport, Bearer auth, `/v1/models` list override |
| `providers/wandb_inference/__init__.py` | Package export |
| `providers/wandb_inference/client.py` | `WandbInferenceProvider` — OpenAI Chat transport, Bearer auth |
| `tests/providers/test_xiaomimimo.py` | 8 unit tests (stream, non-stream, model list, thinking blocks, error handling) |
| `tests/providers/test_wandb_inference.py` | 9 unit tests (stream, non-stream, reasoning_content→thinking_delta, error handling) |

### Modified files

| File | What changed |
|---|---|
| `config/provider_catalog.py` | Added `XIAOMIMIMO` and `WANDB_INFERENCE` `ProviderDescriptor` entries + exported constants |
| `config/settings.py` | Added `xiaomimimo_api_key`, `xiaomimimo_base_url`, `xiaomimimo_proxy`, `wandb_api_key`, `wandb_inference_proxy` fields |
| `providers/defaults.py` | Exported `XIAOMIMIMO_DEFAULT_BASE`, `WANDB_INFERENCE_DEFAULT_BASE` |
| `providers/runtime/factory.py` | Added factory functions and `PROVIDER_FACTORIES` entries for both providers |
| `api/admin_config/provider_manifest.py` | Added display name, description, and credential field overrides for both providers |
| `api/admin_config/manifest.py` | Added smoke model fields `FCC_SMOKE_MODEL_XIAOMIMIMO` and `FCC_SMOKE_MODEL_WANDB_INFERENCE` |
| `smoke/lib/config.py` | Added both providers to `PROVIDER_SMOKE_DEFAULT_MODELS` and `has_provider_configuration` |
| `ARCHITECTURE.md` | Added both providers to the provider reference table |
| `README.md` | Bumped count 24→26, added §25 Xiaomi MiMo and §26 W&B Inference sections, renumbered Mix Providers to §27 |
| `.env.example` | Added `XIAOMIMIMO_API_KEY`, `XIAOMIMIMO_BASE_URL`, `XIAOMIMIMO_PROXY`, `WANDB_API_KEY`, `WANDB_INFERENCE_PROXY`, and smoke model env vars |
| `tests/api/test_dependencies.py` | Added mock settings fields for both providers |
| `tests/api/test_model_router.py` | Added routing tests for `xiaomimimo/MiMo-72B-RL` and `wandb_inference/deepseek-ai/DeepSeek-V3.1` |
| `tests/config/test_config.py` | Added `test_xiaomimimo_settings_from_env` and `test_wandb_inference_settings_from_env` |
| `tests/contracts/test_feature_manifest.py` | Added both provider classes to the registry dict |
| `tests/contracts/test_provider_catalog_order.py` | Added both providers to the expected catalog order |
| `tests/contracts/test_smoke_config.py` | Added both providers to the `_settings` fixture and added smoke config tests |
| `tests/providers/test_provider_runtime.py` | Added descriptor tests, config tests, and instantiation cases for both providers |

---

## Implementation notes

### Xiaomi MiMo — model list URL override

The MiMo hosted API exposes the model list at `/v1/models` (standard OpenAI path) rather than `/anthropic/v1/models` (Anthropic-prefixed path). `XiaomiMiMoProvider` overrides `_send_model_list_request` to hit the correct endpoint:

```python
async def _send_model_list_request(self) -> httpx.Response:
    url = f"{self._base_url.rstrip('/')}/v1/models"
    return await self._http_client.get(url, headers=self._auth_headers())
```

This is the same pattern used by `KimiProvider`.

### W&B Inference — reasoning_content forwarding

When `enable_thinking=True`, the W&B streaming response may include `reasoning_content` deltas alongside `content` deltas. `WandbInferenceProvider` translates these to FCC's `thinking_delta` SSE events, preserving the reasoning trace for clients that consume it:

```python
if chunk.choices[0].delta.reasoning_content:
    yield ThinkingDeltaEvent(thinking=chunk.choices[0].delta.reasoning_content)
```

### Token Plan support (MiMo)

Users on the MiMo Token Plan can override the base URL:

```bash
XIAOMIMIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/anthropic/v1
```

The `base_url_attr = "xiaomimimo_base_url"` field in the `ProviderDescriptor` wires this automatically through `build_provider_config`.

---

## Configuration

### Xiaomi MiMo

```bash
# .env
XIAOMIMIMO_API_KEY=your-key-here
# Optional: use Token Plan endpoint
# XIAOMIMIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/anthropic/v1

MODEL=xiaomimimo/MiMo-72B-RL
# or for tier routing:
MODEL_OPUS=xiaomimimo/MiMo-72B-RL
```

Get your API key at: https://api.xiaomimimo.com

### W&B Inference

```bash
# .env
WANDB_API_KEY=your-wandb-api-key

MODEL=wandb_inference/deepseek-ai/DeepSeek-V3.1
# Other available models:
# wandb_inference/Qwen/Qwen3-Coder-480B-A35B-Instruct
# wandb_inference/meta-llama/Llama-3.3-70B-Instruct
```

Get your API key at: https://wandb.ai/settings (API Keys section)

---

## Test results

```
1953 passed, 9 skipped
ruff format --check: all clean
ruff check: all clean
```

New tests added: **17** (8 for MiMo, 9 for W&B)

---

## Checklist

- [x] Provider client implemented and exported
- [x] `ProviderDescriptor` added to `provider_catalog.py`
- [x] `Settings` fields added for all env vars
- [x] `factory.py` wired
- [x] `provider_manifest.py` display names and descriptions
- [x] `manifest.py` smoke model field
- [x] `smoke/lib/config.py` default model and `has_provider_configuration`
- [x] `ARCHITECTURE.md` updated
- [x] `README.md` updated (count, new sections, renumbered Mix Providers)
- [x] `.env.example` updated
- [x] Unit tests: provider client (stream, non-stream, model list, error handling)
- [x] Contract tests: catalog order, feature manifest, smoke config
- [x] Integration tests: settings env vars, model router routing, provider runtime instantiation
- [x] All 1953 tests pass
- [x] ruff format + check: clean
