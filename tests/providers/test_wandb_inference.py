"""Tests for W&B Inference (OpenAI-compatible Chat Completions) provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.wandb_inference import (
    WANDB_INFERENCE_DEFAULT_BASE,
    WandbInferenceProvider,
)


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "deepseek-ai/DeepSeek-V3.1"
        self.messages = [MagicMock(role="user", content="Hello")]
        self.max_tokens = 100
        self.temperature = 0.7
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.thinking = MagicMock()
        self.thinking.enabled = False
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def wandb_config():
    return ProviderConfig(
        api_key="test-wandb-api-key",
        base_url=WANDB_INFERENCE_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=False,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.transports.openai_chat.transport.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def wandb_provider(wandb_config):
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI"):
        return WandbInferenceProvider(wandb_config)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────


def test_default_base_url_constant():
    assert WANDB_INFERENCE_DEFAULT_BASE == "https://api.inference.wandb.ai/v1"


# ──────────────────────────────────────────────────────────────────────────────
# Initialisation
# ──────────────────────────────────────────────────────────────────────────────


def test_init_uses_default_base_url_and_api_key(wandb_config):
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI") as mock_openai:
        provider = WandbInferenceProvider(wandb_config)
    assert provider._api_key == "test-wandb-api-key"
    assert provider._base_url == WANDB_INFERENCE_DEFAULT_BASE
    mock_openai.assert_called_once()


def test_init_strips_trailing_slash(wandb_config):
    config = wandb_config.model_copy(
        update={"base_url": f"{WANDB_INFERENCE_DEFAULT_BASE}/"}
    )
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI"):
        provider = WandbInferenceProvider(config)
    assert provider._base_url == WANDB_INFERENCE_DEFAULT_BASE


# ──────────────────────────────────────────────────────────────────────────────
# Request body
# ──────────────────────────────────────────────────────────────────────────────


def test_build_request_body_basic(wandb_provider):
    body = wandb_provider._build_request_body(MockRequest())
    assert body["model"] == "deepseek-ai/DeepSeek-V3.1"
    assert body["messages"][0]["role"] == "system"
    assert body["max_tokens"] == 100
    assert "max_completion_tokens" not in body


def test_build_request_body_preserves_extra_body(wandb_provider):
    req = MockRequest(extra_body={"metadata": {"user": "u1"}})
    body = wandb_provider._build_request_body(req)
    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    assert eb.get("metadata") == {"user": "u1"}


# ──────────────────────────────────────────────────────────────────────────────
# Streaming
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_response_text(wandb_provider):
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from W&B",
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        wandb_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()
        events = [
            event async for event in wandb_provider.stream_response(MockRequest())
        ]

    assert any(
        '"text_delta"' in event and "Hello from W&B" in event for event in events
    )


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(wandb_config):
    """Reasoning content is forwarded as thinking_delta when thinking is enabled."""
    thinking_config = wandb_config.model_copy(update={"enable_thinking": True})
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI"):
        provider = WandbInferenceProvider(thinking_config)

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content="Thinking via W&B",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    req = MockRequest()
    req.thinking = MagicMock()
    req.thinking.enabled = True

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()
        events = [event async for event in provider.stream_response(req)]

    assert any(
        '"thinking_delta"' in event and "Thinking via W&B" in event for event in events
    )


@pytest.mark.asyncio
async def test_cleanup(wandb_provider):
    wandb_provider._client = AsyncMock()
    await wandb_provider.cleanup()
    wandb_provider._client.close.assert_called_once()
