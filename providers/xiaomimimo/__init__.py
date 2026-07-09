"""Xiaomi MiMo Anthropic-compatible Messages provider."""

from providers.defaults import XIAOMIMIMO_DEFAULT_BASE

from .client import XiaomiMiMoProvider

__all__ = [
    "XIAOMIMIMO_DEFAULT_BASE",
    "XiaomiMiMoProvider",
]
