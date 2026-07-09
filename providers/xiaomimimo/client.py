"""Xiaomi MiMo provider implementation (Anthropic-compatible Messages API)."""

from typing import Any

import httpx

from providers.base import ProviderConfig
from providers.defaults import XIAOMIMIMO_DEFAULT_BASE
from providers.transports.anthropic_messages import (
    AnthropicMessagesTransport,
    NativeMessagesRequestPolicy,
    build_native_messages_request_body,
)

# MiMo exposes models at the standard /v1/models path, not under /anthropic/v1.
_XIAOMIMIMO_OPENAI_MODELS_URL = "https://api.xiaomimimo.com/v1/models"
_ANTHROPIC_VERSION = "2023-06-01"

_REQUEST_POLICY = NativeMessagesRequestPolicy(provider_name="XIAOMIMIMO")


class XiaomiMiMoProvider(AnthropicMessagesTransport):
    """Xiaomi MiMo provider using Anthropic-compatible Messages at api.xiaomimimo.com/anthropic/v1.

    Pay-As-You-Go endpoint: https://api.xiaomimimo.com/anthropic/v1
    Token Plan endpoint:    https://token-plan-cn.xiaomimimo.com/anthropic/v1
                            (set XIAOMIMIMO_BASE_URL to override)
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="XIAOMIMIMO",
            default_base_url=XIAOMIMIMO_DEFAULT_BASE,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_native_messages_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )

    def _request_headers(self) -> dict[str, str]:
        return {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    async def _send_model_list_request(self) -> httpx.Response:
        """Models are listed from /v1/models, not the /anthropic/v1 prefix."""
        return await self._client.get(
            _XIAOMIMIMO_OPENAI_MODELS_URL,
            headers=self._model_list_headers(),
        )

    def _model_list_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}
