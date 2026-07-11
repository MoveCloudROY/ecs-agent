"""Wiring of view model, bridge, and app over a plan-and-task world.

Shared by the ``__main__`` entrypoint and the headless end-to-end tests so
both exercise the exact same object graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ecs_agent.components import StreamingComponent, UserPromptConfigComponent
from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH
from examples.e2e.plan_and_task.tui.app import PlanTaskTuiApp
from examples.e2e.plan_and_task.tui.bridge import PlanTaskTuiBridge
from examples.e2e.plan_and_task.tui.view_model import PlanTaskViewModel

if TYPE_CHECKING:
    from ecs_agent.core import World
    from ecs_agent.types import EntityId
    from examples.e2e.plan_and_task.state_models import RuntimeState


@dataclass(slots=True)
class TuiSession:
    """The wired TUI object graph for one plan-and-task world."""

    app: PlanTaskTuiApp
    bridge: PlanTaskTuiBridge
    view_model: PlanTaskViewModel


def _slash_commands(world: World, agent_id: EntityId) -> tuple[str, ...]:
    prompt_config = world.get_component(agent_id, UserPromptConfigComponent)
    if prompt_config is None:
        return ()
    return tuple(sorted(trigger.pattern for trigger in prompt_config.triggers))


def create_tui_session(
    world: World,
    agent_id: EntityId,
    runtime_state_ref: list[RuntimeState | None],
) -> TuiSession:
    """Wire view model, bridge, and app over an already-built world.

    Enables system-level streaming on the agent entity (the TUI renders
    token-level output) and attaches the bridge, which installs the
    interactive input loop. The caller runs ``Runner.run`` and
    ``app.run_async``/``app.run_test`` concurrently.
    """
    # non_blocking_delta_publish must stay False: the TUI folds deltas into an
    # ordered live buffer, and non-blocking publish delivers StreamEndEvent
    # before the queued deltas, which would flush an empty assistant message.
    world.add_component(
        agent_id,
        StreamingComponent(enabled=True, non_blocking_delta_publish=False),
    )
    view_model = PlanTaskViewModel(
        agent_id=agent_id, phase_ids=tuple(PLAN_TASK_PHASE_GRAPH.phases_by_id)
    )
    # The bridge needs the app to deliver changes and the app needs the bridge
    # as its input sink; break the cycle with a one-slot holder populated
    # before bridge.attach() subscribes anything.
    app_holder: list[PlanTaskTuiApp] = []
    bridge = PlanTaskTuiBridge(
        world=world,
        agent_id=agent_id,
        view_model=view_model,
        runtime_state_ref=runtime_state_ref,
        on_change=lambda change: app_holder[0].dispatch_change(change),
    )
    app = PlanTaskTuiApp(
        view_model=view_model,
        sink=bridge,
        commands=_slash_commands(world, agent_id),
    )
    app_holder.append(app)
    bridge.attach()
    return TuiSession(app=app, bridge=bridge, view_model=view_model)
