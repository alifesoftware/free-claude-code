"""Xiaomi MiMo provider implementation (OpenAI-compatible Chat Completions)."""

from typing import Any

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from providers.base import ProviderConfig
from providers.defaults import XIAOMIMIMO_DEFAULT_BASE
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

# MiMo exposes an OpenAI-compatible Chat Completions endpoint.
# Pay-As-You-Go:  https://api.xiaomimimo.com/v1
# Token Plan:     https://token-plan-cn.xiaomimimo.com/v1  (set XIAOMIMIMO_BASE_URL)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="XIAOMIMIMO",
    default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
    include_extra_body=True,
)


class XiaomiMiMoProvider(OpenAIChatTransport):
    """Xiaomi MiMo provider using OpenAI-compatible Chat Completions.

    Pay-As-You-Go endpoint: https://api.xiaomimimo.com/v1
    Token Plan endpoint:    https://token-plan-cn.xiaomimimo.com/v1
                            (set XIAOMIMIMO_BASE_URL to override)
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="XIAOMIMIMO",
            base_url=config.base_url or XIAOMIMIMO_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        effective_thinking_enabled = self._is_thinking_enabled(
            request, thinking_enabled
        )
        return build_openai_chat_request_body(
            request,
            thinking_enabled=effective_thinking_enabled,
            policy=_REQUEST_POLICY,
        )
