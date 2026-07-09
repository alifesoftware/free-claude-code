"""Tests for Xiaomi MiMo (Anthropic-compatible Messages) provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from api.models.anthropic import Message, MessagesRequest
from core.anthropic.stream_contracts import parse_sse_text, thinking_content
from providers.base import ProviderConfig
from providers.xiaomimimo import XIAOMIMIMO_DEFAULT_BASE, XiaomiMiMoProvider

_OPENAI_MODELS_URL = "https://api.xiaomimimo.com/v1/models"


class FakeResponse:
    def __init__(self, *, lines=None):
        self.status_code = 200
        self._lines = lines or []
        self.is_closed = False
        self.headers = httpx.Headers()
        self.request = httpx.Request(
            "POST", "https://api.xiaomimimo.com/anthropic/v1/messages"
        )

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self):
        self.is_closed = True


@pytest.fixture
def mimo_config():
    return ProviderConfig(
        api_key="test-mimo-key",
        base_url=XIAOMIMIMO_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    with patch(
        "providers.transports.anthropic_messages.transport.GlobalRateLimiter"
    ) as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def mimo_provider(mimo_config):
    return XiaomiMiMoProvider(mimo_config)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────


def test_default_base_url():
    assert XIAOMIMIMO_DEFAULT_BASE == "https://api.xiaomimimo.com/anthropic/v1"


# ──────────────────────────────────────────────────────────────────────────────
# Initialisation
# ──────────────────────────────────────────────────────────────────────────────


def test_init_uses_default_base_url_and_strips_trailing_slash(mimo_config):
    config = mimo_config.model_copy(update={"base_url": f"{XIAOMIMIMO_DEFAULT_BASE}/"})
    with patch("httpx.AsyncClient"):
        provider = XiaomiMiMoProvider(config)
    assert provider._api_key == "test-mimo-key"
    assert provider._base_url == XIAOMIMIMO_DEFAULT_BASE
    assert provider._provider_name == "XIAOMIMIMO"


def test_init_accepts_token_plan_base_url(mimo_config):
    token_plan_url = "https://token-plan-cn.xiaomimimo.com/anthropic/v1"
    config = mimo_config.model_copy(update={"base_url": token_plan_url})
    with patch("httpx.AsyncClient"):
        provider = XiaomiMiMoProvider(config)
    assert provider._base_url == token_plan_url


# ──────────────────────────────────────────────────────────────────────────────
# Request headers
# ──────────────────────────────────────────────────────────────────────────────


def test_request_headers_use_bearer_token(mimo_provider):
    headers = mimo_provider._request_headers()
    assert headers["Authorization"] == "Bearer test-mimo-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["Content-Type"] == "application/json"


def test_model_list_headers_use_bearer_token(mimo_provider):
    headers = mimo_provider._model_list_headers()
    assert headers == {"Authorization": "Bearer test-mimo-key"}


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
    assert body["stream"] is True


# ──────────────────────────────────────────────────────────────────────────────
# Model list — uses /v1/models, not /anthropic/v1/models
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_model_list_requests_openai_compat_url(mimo_provider):
    """MiMo exposes models at /v1/models, not under the /anthropic/v1 prefix."""
    called: dict[str, str] = {}

    async def fake_get(url: str, **_k):
        called["url"] = url
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = lambda: {"data": [{"id": "MiMo-7B-RL"}, {"id": "MiMo-72B-RL"}]}
        mock_resp.aclose = AsyncMock()
        return mock_resp

    mimo_provider._client.get = fake_get
    ids = await mimo_provider.list_model_ids()
    assert called["url"] == _OPENAI_MODELS_URL
    assert ids == frozenset({"MiMo-7B-RL", "MiMo-72B-RL"})


# ──────────────────────────────────────────────────────────────────────────────
# Streaming
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_response_text_and_thinking(mimo_provider):
    request = MessagesRequest(
        model="MiMo-7B-RL",
        messages=[Message(role="user", content="hi")],
    )
    response = FakeResponse(
        lines=[
            "event: message_start",
            'data: {"type":"message_start"}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reasoning"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"done"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":1}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
    )
    with (
        patch.object(
            mimo_provider._client, "build_request", return_value=MagicMock()
        ) as mock_build,
        patch.object(
            mimo_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [event async for event in mimo_provider.stream_response(request)]

    parsed = parse_sse_text("".join(events))
    assert thinking_content(parsed) == "reasoning"
    assert response.is_closed
    assert mock_build.call_args.args[:2] == ("POST", "/messages")
    assert (
        mock_build.call_args.kwargs["headers"]["Authorization"]
        == "Bearer test-mimo-key"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cleanup_aclose(mimo_provider):
    mimo_provider._client = AsyncMock()
    await mimo_provider.cleanup()
    mimo_provider._client.aclose.assert_awaited_once()
