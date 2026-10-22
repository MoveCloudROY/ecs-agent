"""Live LLM scenarios selected explicitly with --live and a model profile."""

from __future__ import annotations


import pytest

from ecs_agent.components import (
    CompactionConfigComponent,
    ContextTrimConfig,
    ConversationArchiveComponent,
    ConversationComponent,
    CurrentCompactionSummaryComponent,
    LLMComponent,
)
from ecs_agent.core import World
from ecs_agent.providers.claude_model import ClaudeModel
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.types import Message, ToolCall

from tests.support.models import LiveModelFactory


def _claude_model(*, live_model_factory: LiveModelFactory) -> ClaudeModel:
    return live_model_factory.claude(max_tokens=256)


@pytest.mark.live(formats=["anthropic_messages"], capabilities=["text"])
@pytest.mark.compaction
@pytest.mark.asyncio
async def test_real_trim_then_summarize_pipeline(
    *, live_model_factory: LiveModelFactory
) -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        LLMComponent(model=_claude_model(live_model_factory=live_model_factory)),
    )
    # Oldest turn is a tool span (trimmable); the rest is essential prose that
    # cannot be trimmed, so trimming alone won't fit -> real summary is produced.
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                Message(role="user", content="Look up the capital of France."),
                Message(
                    role="assistant",
                    content="Using the lookup tool.",
                    tool_calls=[ToolCall(id="c1", name="lookup", arguments={})],
                ),
                Message(role="tool", content="Paris. " * 200, tool_call_id="c1"),
                Message(
                    role="assistant",
                    content="The capital of France is Paris. " * 40,
                ),
                Message(
                    role="user", content="Now summarize our whole discussion. " * 40
                ),
            ]
        ),
    )
    world.add_component(entity_id, ConversationArchiveComponent())
    world.add_component(entity_id, CompactionConfigComponent(threshold_tokens=50))
    world.add_component(
        entity_id, ContextTrimConfig(max_tokens=50, protect_recent_turns=0)
    )

    await CompactionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    # The oldest tool span was permanently dropped by the trim step.
    assert all("Paris. Paris." not in (m.content or "") for m in conversation.messages)
    # Trimming wasn't enough -> the real model produced a non-empty summary.
    summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)
    assert summary is not None
    assert summary.summary.strip()
