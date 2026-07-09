"""W&B Inference OpenAI-compatible adapter."""

from providers.defaults import WANDB_INFERENCE_DEFAULT_BASE

from .client import WandbInferenceProvider

__all__ = [
    "WANDB_INFERENCE_DEFAULT_BASE",
    "WandbInferenceProvider",
]
