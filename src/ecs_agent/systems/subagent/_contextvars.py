"""Process-context channels shared across the subagent package.

These ContextVars carry per-call flags into deeply nested delegation calls. They are
a known smell earmarked for replacement by an explicit typed launch context in a
later refactor (plan Q2 — deferred). They live in this leaf module so cycle-free
submodules (child_world, background, delegation) can share them without importing the
package ``__init__``.
"""

from __future__ import annotations

from contextvars import ContextVar

_PUBLISH_COMPLETION_EVENT: ContextVar[bool] = ContextVar(
    "_PUBLISH_COMPLETION_EVENT",
    default=True,
)
_BACKGROUND_RESULT_ENVELOPE_ENABLED: ContextVar[bool] = ContextVar(
    "_BACKGROUND_RESULT_ENVELOPE_ENABLED",
    default=False,
)
_BACKGROUND_LAUNCH_OBSERVABILITY_CONTEXT: ContextVar[
    tuple[str | None, str | None, str | None] | None
] = ContextVar(
    "_BACKGROUND_LAUNCH_OBSERVABILITY_CONTEXT",
    default=None,
)

__all__ = [
    "_PUBLISH_COMPLETION_EVENT",
    "_BACKGROUND_RESULT_ENVELOPE_ENABLED",
    "_BACKGROUND_LAUNCH_OBSERVABILITY_CONTEXT",
]
