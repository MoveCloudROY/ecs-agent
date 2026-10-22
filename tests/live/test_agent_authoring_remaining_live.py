"""Opt-in Responses API checks for complete authoring and controlled edits.

Select a profile with --live; see docs/testing.md.
"""

import asyncio
from pathlib import Path

import pytest

from ecs_agent import AgentDefinition, build_agent, build_agent_from_specs, tool
from ecs_agent.components import SessionRefComponent
from ecs_agent.dsl import AgentSpec
from ecs_agent.prompts import (
    PlaceholderSpec,
    PromptTemplateSource,
    SystemPromptConfigSpec,
)
from ecs_agent.sessions import (
    InMemorySessionStore,
    SessionService,
    release_session_writer,
)

from tests.support.models import LiveModelFactory


@pytest.mark.live(
    formats=["openai_responses"], capabilities=["text", "tools", "streaming"]
)
@pytest.mark.authoring
@pytest.mark.parametrize("stream", [False, True])
async def test_complete_authoring_tools_and_context_edits(
    stream: bool, *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.create()
    service = SessionService(InMemorySessionStore())
    calls = []

    @tool()
    async def echo_value(value: int) -> str:
        """Return an integer with a fixed test marker; no external effects."""
        calls.append(value)
        return f"TOOL_VALUE={value}"

    agent = build_agent(
        AgentDefinition(
            model,
            prompt="Follow the requested tool call, then give a concise final answer.",
            tools=[echo_value],
        ),
        session_service=service,
    )
    try:
        async with asyncio.timeout(live_model_factory.profile.timeouts.case_seconds):
            request = "Call echo_value exactly once with value=7. Use the actual tool, then report its result."
            if stream:
                async with agent.stream_turn(request, max_steps=5) as events:
                    async for event in events:
                        assert event.entity_id == agent.entity
                result = agent.last_result
            else:
                result = await agent.run_turn(request, max_steps=5)
            assert result is not None and result.success, result
            assert calls == [7]
            assert any(
                message.role == "tool" and message.content == "TOOL_VALUE=7"
                for message in agent.history_snapshot().messages
            )

            agent.set_prompt(
                SystemPromptConfigSpec(
                    PromptTemplateSource(
                        inline="On a greeting, answer exactly ${marker}. Do not use any tools."
                    ),
                    [PlaceholderSpec("marker", "AFTER_EDIT_29")],
                )
            )
            snapshot = agent.history_snapshot()
            agent.replace_history(
                [],
                reason="isolate new prompt",
                expected_history_id=snapshot.history_id,
                expected_revision=snapshot.revision,
            )
            result = await agent.run_turn("Hello", max_steps=3)
            assert result.success, result
            assert "AFTER_EDIT_29" in result.final_message.content
            assert "AFTER_EDIT_29" in agent.inspect_prompt().text
            assert calls == [7]

            previous = agent.world.get_component(
                agent.entity, SessionRefComponent
            ).session_id
            agent.reset_session(reason="new live session")
            current = agent.world.get_component(
                agent.entity, SessionRefComponent
            ).session_id
            assert current != previous and agent.history_snapshot().messages == ()
            assert service.replay_history(current) == []
            assert any(
                record.kind == "prompt_config_changed"
                for record in service.store.records(previous)
            )
    finally:
        release_session_writer(agent.world, agent.entity)
        service.close()


@pytest.mark.live(formats=["openai_responses"], capabilities=["text", "tools"])
@pytest.mark.authoring
async def test_dsl_subagent_skill_and_trigger_use_responses(
    tmp_path: Path, *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.create()
    service = SessionService(InMemorySessionStore())
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: live-evidence\ndescription: A fixed verification marker\n---\nThe verified marker is CHILD_SKILL_47.\n"
    )
    specs = {
        "main": AgentSpec(
            "primary",
            "configured",
            "Use the requested subagent and report its final answer.",
        ),
        "researcher": AgentSpec(
            "subagent",
            "configured",
            "You are ${role}. Follow the user task exactly.",
            placeholders=[{"name": "role", "value": "a verification researcher"}],
            skills=[{"path": "skill"}],
            triggers=[
                {
                    "pattern": "/probe",
                    "match_mode": "prefix",
                    "action": "replace",
                    "content": "Call load_skill_details with skill_name='live-evidence' exactly once. Then output only the verified marker in its result.",
                }
            ],
        ),
    }
    agent = build_agent_from_specs(
        specs,
        lambda *_: model,
        source_dir=tmp_path,
        session_service=service,
        workspace_id="authoring-live",
    )
    try:
        async with asyncio.timeout(live_model_factory.profile.timeouts.case_seconds):
            result = await agent.run_turn(
                "Call subagent with category='researcher', prompt='/probe', background=false. Report the returned marker.",
                max_steps=6,
            )
            assert result.success, result
            assert "CHILD_SKILL_47" in result.final_message.content
            root_id = agent.world.get_component(
                agent.entity, SessionRefComponent
            ).session_id
            children = [
                info
                for info in service.store.sessions()
                if info.parent_session_id == root_id
            ]
            assert children
            records = [
                record
                for info in children
                for record in service.store.records(info.session_id)
            ]
            assert any(
                record.kind == "prompt_render"
                and "Call load_skill_details" in record.payload.get("content", "")
                for record in records
            )
            assert any(
                record.kind == "message"
                and record.payload.get("role") == "tool"
                and "CHILD_SKILL_47" in record.payload.get("content", "")
                for record in records
            )
    finally:
        release_session_writer(agent.world, agent.entity)
        service.close()
