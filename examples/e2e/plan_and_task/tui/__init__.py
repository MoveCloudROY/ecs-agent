"""Textual TUI for the plan-and-task E2E example.

Layers:

* ``view_model`` — pure view state; consumes ECS events, no textual imports.
* ``bridge`` — subscribes the view model to a ``World``'s event bus and owns
  the pending ``UserInputRequestedEvent`` future.
* ``app`` — the Textual application rendering the view model.

Run with ``python -m examples.e2e.plan_and_task.tui`` (same environment
variables as ``examples/e2e/plan_and_task/main.py``).
"""

from examples.e2e.plan_and_task.tui.view_model import (
    PlanTaskViewModel,
    TranscriptEntry,
    UiChange,
)

__all__ = [
    "PlanTaskViewModel",
    "TranscriptEntry",
    "UiChange",
]
