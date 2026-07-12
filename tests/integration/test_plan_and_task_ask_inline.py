"""ask_question answers must stay inline in the conversation, even with the
tool-results scratchbook sink enabled.

Regression: the TUI runs with ``enable_tool_sink=True``, which externalizes
every tool result to a file and puts the *record path* in the conversation. A
user's ask_question answer (small, critical, conversational input — especially
free-text that redirects the task) then never reaches the model verbatim: it
reads the path lazily or not at all, so custom input appears ignored. The
answer must be inlined so the model always sees it.
"""

from __future__ import annotations

import asyncio

import pytest

from ecs_agent.components import (
    ConversationComponent,
    PendingToolCallsComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import World
from ecs_agent.scratchbook import ArtifactRegistry
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import ToolCall
from examples.e2e.plan_and_task.ask_tool import (
    QuestionAnswer,
    UserQuestionRequestedEvent,
    install_ask_question_tool,
)

_CUSTOM = "Pi is a LLM Agent framework, please search web for more info"


@pytest.mark.asyncio
async def test_ask_question_answer_inlined_with_tool_sink(tmp_path) -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(entity_id, ConversationComponent(messages=[]))
    install_ask_question_tool(world, entity_id)

    async def frontend(event: UserQuestionRequestedEvent) -> None:
        # User picks nothing and types a free-text redirect.
        event.answer_future.set_result(
            [
                QuestionAnswer(
                    header=q.header,
                    question=q.question,
                    custom_text=_CUSTOM,
                )
                for q in event.questions
            ]
        )

    world.event_bus.subscribe(UserQuestionRequestedEvent, frontend)
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(
                    id="ask-1",
                    name="ask_question",
                    arguments={
                        "questions": [
                            {
                                "header": "Scope",
                                "question": "What should the report cover?",
                                "options": [{"label": "Single"}, {"label": "Multi"}],
                            }
                        ]
                    },
                )
            ]
        ),
    )

    # Sink enabled — same as the real TUI entrypoint.
    registry = ArtifactRegistry(root=tmp_path / ".scratchbook")
    await ToolExecutionSystem(registry=registry).process(world)

    conv = world.get_component(entity_id, ConversationComponent)
    assert conv is not None
    tool_msg = conv.messages[-1]
    assert tool_msg.role == "tool"
    # The model must see the user's actual answer, not a scratchbook path.
    assert _CUSTOM in (tool_msg.content or ""), (
        f"answer was externalized to a path: {tool_msg.content!r}"
    )
    assert not (tool_msg.content or "").startswith("scratchbook/")


@pytest.mark.asyncio
async def test_non_inline_tool_result_still_externalized(tmp_path) -> None:
    """A normal (non-inline) tool result is still externalized to a path."""
    world = World()
    entity_id = world.create_entity()

    async def echo(msg: str = "") -> str:
        return f"echoed: {msg}"

    from ecs_agent.types import ToolSchema

    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "echo": ToolSchema(
                    name="echo", description="", parameters={}
                )
            },
            handlers={"echo": echo},
        ),
    )
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="e1", name="echo", arguments={"msg": "hi"})]
        ),
    )

    registry = ArtifactRegistry(root=tmp_path / ".scratchbook")
    await ToolExecutionSystem(registry=registry).process(world)

    conv = world.get_component(entity_id, ConversationComponent)
    assert conv is not None
    assert conv.messages[-1].content.startswith("scratchbook/records/tool/tool_")
