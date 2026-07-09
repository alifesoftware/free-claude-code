"""Tests for Xiaomi MiMo (OpenAI-compatible Chat Completions) provider."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models.anthropic import Message, MessagesRequest
from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.stream_contracts import (
    parse_sse_text,
    text_content,
    thinking_content,
)
from providers.base import ProviderConfig
from providers.transports.openai_chat import OpenAIChatTransport
from providers.xiaomimimo import XIAOMIMIMO_DEFAULT_BASE, XiaomiMiMoProvider


class AsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


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
def mimo_provider():
    return XiaomiMiMoProvider(
        ProviderConfig(
            api_key="test-mimo-key",
            base_url=XIAOMIMIMO_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
            enable_thinking=True,
        )
    )


def _chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    finish_reason: str | None = None,
):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        usage=None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────


def test_default_base_url():
    assert XIAOMIMIMO_DEFAULT_BASE == "https://api.xiaomimimo.com/v1"


# ──────────────────────────────────────────────────────────────────────────────
# Initialisation
# ──────────────────────────────────────────────────────────────────────────────


def test_init_uses_openai_chat_transport(mimo_provider):
    assert isinstance(mimo_provider, OpenAIChatTransport)
    assert mimo_provider._api_key == "test-mimo-key"
    assert mimo_provider._base_url == XIAOMIMIMO_DEFAULT_BASE
    assert mimo_provider._provider_name == "XIAOMIMIMO"


def test_init_accepts_token_plan_base_url():
    token_plan_url = "https://token-plan-cn.xiaomimimo.com/v1"
    provider = XiaomiMiMoProvider(
        ProviderConfig(
            api_key="test-mimo-key",
            base_url=token_plan_url,
            rate_limit=10,
            rate_window=60,
        )
    )
    assert provider._base_url == token_plan_url


# ──────────────────────────────────────────────────────────────────────────────
# Request body
# ──────────────────────────────────────────────────────────────────────────────


def test_build_request_body_basic(mimo_provider):
    request = MessagesRequest(
        model="MiMo-7B-RL",
        messages=[Message(role="user", content="Hello")],
    )
    body = mimo_provider._build_request_body(request)
    assert body["model"] == "MiMo-7B-RL"
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


# ──────────────────────────────────────────────────────────────────────────────
# Model list — uses OpenAI-compatible /models endpoint
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lists_models_from_openai_models_endpoint(mimo_provider):
    mimo_provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(id="MiMo-7B-RL"), SimpleNamespace(id="MiMo-72B-RL")]
        )
    )
    assert await mimo_provider.list_model_ids() == frozenset(
        {"MiMo-7B-RL", "MiMo-72B-RL"}
    )


# ──────────────────────────────────────────────────────────────────────────────
# Streaming
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_response_text(mimo_provider):
    request = MessagesRequest(
        model="MiMo-7B-RL",
        messages=[Message(role="user", content="hi")],
    )
    stream = AsyncStream(
        [
            _chunk(content="hello", finish_reason="stop"),
        ]
    )
    with patch.object(
        mimo_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream,
    ):
        events = [event async for event in mimo_provider.stream_response(request)]
    parsed = parse_sse_text("".join(events))
    assert text_content(parsed) == "hello"
    assert stream.closed


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(mimo_provider):
    request = MessagesRequest(
        model="MiMo-7B-RL",
        messages=[Message(role="user", content="hi")],
    )
    stream = AsyncStream(
        [
            _chunk(reasoning_content="plan"),
            _chunk(content="done", finish_reason="stop"),
        ]
    )
    with patch.object(
        mimo_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream,
    ):
        events = [event async for event in mimo_provider.stream_response(request)]
    parsed = parse_sse_text("".join(events))
    assert thinking_content(parsed) == "plan"
    assert text_content(parsed) == "done"
    assert stream.closed


# ──────────────────────────────────────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cleanup_closes_openai_client(mimo_provider):
    mimo_provider._client = MagicMock()
    mimo_provider._client.close = AsyncMock()
    await mimo_provider.cleanup()
    mimo_provider._client.close.assert_awaited_once()
