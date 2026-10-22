"""Offline scenario coverage extracted from formerly credential-gated files."""

from __future__ import annotations
from typing import Any
import pytest
from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import (
    CompletionResult,
    Message,
)


@pytest.mark.asyncio
async def test_real_read_file_returns_clean_content(tmp_path: Any) -> None:
    from ecs_agent.tools.builtins.file_tools import read_file

    test_file = tmp_path / "test.txt"
    raw_lines = ["Alpha", "Beta", "Gamma"]
    test_file.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    result = await read_file(file_path="test.txt", workspace_root=str(tmp_path))

    assert result == "\n".join(raw_lines)
    assert "#" not in result
    assert "|" not in result


@pytest.mark.asyncio
async def test_real_glob_finds_files_in_workspace(tmp_path: Any) -> None:
    from ecs_agent.tools.builtins.glob_tool import glob

    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.py").write_text("print('x')", encoding="utf-8")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "c.txt").write_text("c", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "d.txt").write_text("d", encoding="utf-8")
    (tmp_path / "src" / "e.py").write_text("print('y')", encoding="utf-8")

    result = await glob(pattern="**/*.txt", base_path=".", workspace_root=str(tmp_path))
    matched = result.splitlines()

    assert set(matched) == {"a.txt", "notes/c.txt", "src/d.txt"}
    assert all(not path.endswith(".py") for path in matched)
    assert all(not path.startswith("/") for path in matched)

    empty_result = await glob(
        pattern="**/*.xyz", base_path=".", workspace_root=str(tmp_path)
    )
    assert empty_result == ""


@pytest.mark.asyncio
async def test_real_llm_builtin_tools_read_file_smoke(tmp_path: Any) -> None:
    from ecs_agent import BuiltinToolsSkill, SkillManager
    from ecs_agent.providers import FakeModel
    from ecs_agent.systems.tool_execution import ToolExecutionSystem
    from ecs_agent.types import ToolCall

    class RecordingFakeModel(FakeModel):
        def __init__(self, responses: list[CompletionResult]) -> None:
            super().__init__(responses)
            self.recorded_messages: list[list[Message]] = []

        async def complete(
            self,
            messages: list[Message],
            tools: list[Any] | None = None,
            stream: bool = False,
            response_format: dict[str, Any] | None = None,
        ) -> CompletionResult | Any:
            self.recorded_messages.append(list(messages))
            return await super().complete(
                messages,
                tools=tools,
                stream=stream,
                response_format=response_format,
            )

    note = tmp_path / "note.txt"
    note.write_text("one\ntwo\nthree\n", encoding="utf-8")

    model = RecordingFakeModel(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tc-read-1",
                            name="read_file",
                            arguments={"file_path": "note.txt"},
                        )
                    ],
                )
            ),
            CompletionResult(
                message=Message(
                    role="assistant", content="Read complete from tool output"
                )
            ),
        ]
    )

    world = World()
    entity = world.create_entity()
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="Read note.txt")]),
    )

    skill = BuiltinToolsSkill()
    skill.bind_workspace(str(tmp_path))
    SkillManager().install(world, entity, skill)

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    await Runner().run(world, max_ticks=5)

    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    assert any(msg.role == "tool" for msg in conv.messages)
    assert conv.messages[-1].role == "assistant"
    assert len(conv.messages[-1].content) > 0

    assert len(model.recorded_messages) >= 2
    second_turn_messages = model.recorded_messages[1]
    tool_messages = [msg for msg in second_turn_messages if msg.role == "tool"]
    assert len(tool_messages) >= 1
    assert tool_messages[-1].content == "one\ntwo\nthree"
    assert "#" not in tool_messages[-1].content
    assert "|" not in tool_messages[-1].content
