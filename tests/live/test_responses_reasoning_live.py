"""Explicit live integration coverage migrated from mixed/offline directories."""

from __future__ import annotations
import pytest
from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import (
    Message,
)

from tests.support.models import LiveModelFactory


@pytest.mark.live(formats=["openai_responses"], capabilities=["text", "tools"])
@pytest.mark.workflow
async def test_responses_api_with_real_llm(
    *, live_model_factory: LiveModelFactory
) -> None:
    """Test reasoning against an explicitly selected Responses profile."""

    world = World()

    # Keep the declared Responses protocol throughout the scenario.
    model = live_model_factory.openai()

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(
        entity,
        ConversationComponent(
            messages=[Message(role="user", content="Say 'hello' and nothing else")]
        ),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=3)

    # Verify response was generated
    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) >= 2  # user + assistant
    assert conv.messages[-1].role == "assistant"
    assert len(conv.messages[-1].content) > 0
