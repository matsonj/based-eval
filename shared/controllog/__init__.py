"""Controllable logging SDK (events + balanced postings).

Designed to be embedded now and easily extracted into a standalone library.
This provides double-entry accounting for:
- Token usage (resource.tokens)
- Time tracking (resource.time_ms)
- Cost tracking (resource.money)
- State transitions (truth.state)
- Utility/reward (value.utility)
"""

from .sdk import init, event, post, new_id
from .builders import (
    agent_run,
    model_response,
    model_prompt,
    model_completion,
    state_move,
    utility,
)

__all__ = [
    "init",
    "event",
    "post",
    "new_id",
    "agent_run",
    "model_response",
    "model_prompt",
    "model_completion",
    "state_move",
    "utility",
]

