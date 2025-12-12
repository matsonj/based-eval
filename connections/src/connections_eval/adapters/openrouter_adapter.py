"""OpenRouter API adapter - delegates to shared infrastructure.

This module re-exports the shared OpenRouter adapter for backwards compatibility.
"""

# Import from shared infrastructure
from shared.adapters.openrouter_adapter import chat, OpenRouterAdapter, resolve_model_id

__all__ = ["chat", "OpenRouterAdapter", "resolve_model_id"]
