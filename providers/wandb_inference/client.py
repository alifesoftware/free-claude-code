"""W&B Inference provider implementation (OpenAI-compatible Chat Completions)."""

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import WANDB_INFERENCE_DEFAULT_BASE
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="WANDB_INFERENCE",
    include_extra_body=True,
)


class WandbInferenceProvider(OpenAIChatTransport):
    """W&B Inference provider using the OpenAI-compatible endpoint at api.inference.wandb.ai/v1.

    Hosts open-source models (Qwen3, DeepSeek, Llama 4, GPT OSS, etc.) via CoreWeave.
    Auth: W&B API key as Bearer token (same key used for W&B Weave tracing).
    See https://docs.wandb.ai/guides/inference for available models.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="WANDB_INFERENCE",
            base_url=config.base_url or WANDB_INFERENCE_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )
