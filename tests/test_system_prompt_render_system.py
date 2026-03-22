from __future__ import annotations

from pathlib import Path

import pytest

from ecs_agent.components import (
    LLMComponent,
    RenderedSystemPromptComponent,
    RenderedUserPromptComponent,
    SkillComponent,
    SkillMetadata,
    SubagentRegistryComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import World
from ecs_agent.prompts.contracts import (
    PlaceholderSpec,
    PromptConfigSpec,
    PromptTemplateSource,
    TriggerSpec,
)
from ecs_agent.providers import FakeProvider
from ecs_agent.prompts.registry import resolve_placeholder_values
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.types import SubagentConfig, ToolSchema


def test_contract_prompt_template_source_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PromptTemplateSource()

    with pytest.raises(ValueError, match="exactly one"):
        PromptTemplateSource(inline="inline", file_path="prompt.md")


def test_contract_prompt_template_source_allows_inline_or_file_path() -> None:
    inline = PromptTemplateSource(inline="hello")
    by_file = PromptTemplateSource(file_path="prompts/system.md")

    assert inline.inline == "hello"
    assert inline.file_path is None
    assert by_file.inline is None
    assert by_file.file_path == "prompts/system.md"


def test_contract_placeholder_accepts_static_and_callable_values() -> None:
    static = PlaceholderSpec(name="project_name", value="ecs-agent")
    dynamic = PlaceholderSpec(name="release", value=lambda: "v1")

    assert static.value == "ecs-agent"
    assert callable(dynamic.value)


def test_invalid_placeholder_name_rejects_illegal_identifier() -> None:
    with pytest.raises(ValueError, match="Invalid placeholder name"):
        PlaceholderSpec(name="invalid-name", value="x")


def test_invalid_placeholder_name_rejects_reserved_prefix() -> None:
    with pytest.raises(ValueError, match="reserved"):
        PlaceholderSpec(name="_installed_tools", value="x")


def test_invalid_placeholder_callable_return_type_raises() -> None:
    with pytest.raises(ValueError, match="must return str"):
        resolve_placeholder_values([PlaceholderSpec(name="n", value=lambda: 123)])


def test_invalid_placeholder_callable_exception_bubbles() -> None:
    def _explode() -> str:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        resolve_placeholder_values([PlaceholderSpec(name="n", value=_explode)])


def test_contract_placeholder_callable_is_invoked_once_per_render() -> None:
    calls = 0

    def _value() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    snapshot = resolve_placeholder_values([PlaceholderSpec(name="run", value=_value)])

    assert snapshot == {"run": "ok"}
    assert calls == 1


def test_contract_prompt_config_spec_keeps_user_placeholders() -> None:
    spec = PromptConfigSpec(
        template_source=PromptTemplateSource(inline="Hello ${user_name}"),
        placeholders=[PlaceholderSpec(name="user_name", value="roy")],
    )

    assert spec.placeholders[0].name == "user_name"


def test_contract_trigger_spec_accepts_supported_fields() -> None:
    trigger = TriggerSpec(
        pattern="@code",
        match_mode="prefix",
        action="replace",
        content="be terse",
        priority=7,
    )

    assert trigger.pattern == "@code"
    assert trigger.priority == 7


def test_component_rendered_system_prompt_component_has_text_and_snapshot() -> None:
    component = RenderedSystemPromptComponent(
        text="system prompt",
        placeholder_snapshot={"user_name": "roy"},
    )

    assert component.text == "system prompt"
    assert component.placeholder_snapshot["user_name"] == "roy"


def test_component_rendered_user_prompt_component_has_text_and_turn_id() -> None:
    component = RenderedUserPromptComponent(
        text="normalized user prompt", turn_id="turn-1"
    )

    assert component.text == "normalized user prompt"
    assert component.turn_id == "turn-1"


@pytest.mark.asyncio
async def test_render_system_renders_inline_template_and_bridges_to_llm() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(
            template_source=PromptTemplateSource(
                inline="Hello ${user_name}\n${_installed_tools}"
            ),
            placeholders=[PlaceholderSpec(name="user_name", value="roy")],
        ),
    )
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "write_file": ToolSchema(
                    name="write_file",
                    description="write",
                    parameters={"type": "object", "properties": {}},
                )
            },
            handlers={},
        ),
    )
    world.add_component(entity_id, LLMComponent(provider=object(), model="demo"))

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    llm = world.get_component(entity_id, LLMComponent)
    assert rendered is not None
    assert llm is not None
    assert rendered.text == "Hello roy\n- write_file"
    assert rendered.placeholder_snapshot == {
        "user_name": "roy",
        "_installed_tools": "- write_file",
        "_installed_skills": "- none",
        "_installed_mcps": "- none",
        "_installed_subagents": "- none",
    }
    assert llm.system_prompt == rendered.text


@pytest.mark.asyncio
async def test_render_system_writes_component_for_inline_template_without_placeholders() -> (
    None
):
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(template_source=PromptTemplateSource(inline="Always terse.")),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "Always terse."


@pytest.mark.asyncio
async def test_render_system_loads_template_from_file(tmp_path: Path) -> None:
    template_path = tmp_path / "system.txt"
    template_path.write_text("Skills:\n${_installed_skills}", encoding="utf-8")

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(
            template_source=PromptTemplateSource(file_path=str(template_path)),
        ),
    )
    world.add_component(
        entity_id,
        SkillComponent(
            skills={
                "alpha": SkillMetadata(
                    name="alpha",
                    description="a",
                    tool_names=[],
                    has_system_prompt=False,
                ),
                "beta": SkillMetadata(
                    name="beta",
                    description="b",
                    tool_names=[],
                    has_system_prompt=False,
                ),
            }
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "Skills:\n- alpha\n- beta"


@pytest.mark.asyncio
async def test_render_system_renders_all_builtin_placeholders() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(
            template_source=PromptTemplateSource(
                inline=(
                    "${_installed_tools}\n${_installed_skills}\n"
                    "${_installed_mcps}\n${_installed_subagents}"
                )
            ),
        ),
    )
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "bash": ToolSchema(
                    name="bash",
                    description="shell",
                    parameters={"type": "object", "properties": {}},
                ),
                "read": ToolSchema(
                    name="read",
                    description="read",
                    parameters={"type": "object", "properties": {}},
                ),
            },
            handlers={},
        ),
    )
    world.add_component(
        entity_id,
        SkillComponent(
            skills={
                "zeta": SkillMetadata(
                    name="zeta",
                    description="z",
                    tool_names=[],
                    has_system_prompt=False,
                ),
                "alpha": SkillMetadata(
                    name="alpha",
                    description="a",
                    tool_names=[],
                    has_system_prompt=False,
                ),
            }
        ),
    )
    world.add_component(
        entity_id,
        SubagentRegistryComponent(
            subagents={
                "planner": SubagentConfig(name="planner", provider=object(), model="m"),
                "researcher": SubagentConfig(
                    name="researcher", provider=object(), model="m"
                ),
            }
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == (
        "- bash\n- read\n- alpha\n- zeta\n- none\n- planner\n- researcher"
    )


@pytest.mark.asyncio
async def test_render_system_rejects_unknown_placeholder() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(template_source=PromptTemplateSource(inline="${missing}")),
    )

    with pytest.raises(ValueError, match="unknown placeholders"):
        await SystemPromptRenderSystem().process(world)


@pytest.mark.asyncio
async def test_render_system_rejects_missing_template_file() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(
            template_source=PromptTemplateSource(file_path="/tmp/does-not-exist.prompt")
        ),
    )

    with pytest.raises(ValueError, match="missing template file"):
        await SystemPromptRenderSystem().process(world)


@pytest.mark.asyncio
async def test_render_system_rejects_non_string_callable_return() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(
            template_source=PromptTemplateSource(inline="${dynamic}"),
            placeholders=[PlaceholderSpec(name="dynamic", value=lambda: 123)],
        ),
    )

    with pytest.raises(ValueError, match="must return str"):
        await SystemPromptRenderSystem().process(world)


@pytest.mark.asyncio
async def test_render_system_callable_raises_propagates() -> None:
    world = World()
    entity_id = world.create_entity()

    def _explode() -> str:
        raise RuntimeError("boom")

    world.add_component(
        entity_id,
        PromptConfigSpec(
            template_source=PromptTemplateSource(inline="${dynamic}"),
            placeholders=[PlaceholderSpec(name="dynamic", value=_explode)],
        ),
    )

    with pytest.raises(RuntimeError, match="boom"):
        await SystemPromptRenderSystem().process(world)


@pytest.mark.asyncio
async def test_render_system_inline_no_placeholders() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(
            template_source=PromptTemplateSource(inline="Hello world"),
            placeholders=[],
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "Hello world"


@pytest.mark.asyncio
async def test_render_system_user_placeholder_static() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(
            template_source=PromptTemplateSource(inline="Hello ${name}"),
            placeholders=[PlaceholderSpec(name="name", value="Alice")],
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "Hello Alice"


@pytest.mark.asyncio
async def test_render_system_built_in_installed_tools() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_tools}")
        ),
    )
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "read_file": ToolSchema(
                    name="read_file",
                    description="read file",
                    parameters={"type": "object", "properties": {}},
                ),
                "bash": ToolSchema(
                    name="bash",
                    description="bash",
                    parameters={"type": "object", "properties": {}},
                ),
            },
            handlers={},
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "- bash\n- read_file"
    assert "- bash" in rendered.text
    assert "- read_file" in rendered.text


@pytest.mark.asyncio
async def test_render_system_empty_inventory_renders_none() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_tools}")
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "- none"


@pytest.mark.asyncio
async def test_render_system_unknown_placeholder_raises() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(template_source=PromptTemplateSource(inline="${unknown_var}")),
    )

    with pytest.raises(ValueError):
        await SystemPromptRenderSystem().process(world)


@pytest.mark.asyncio
async def test_render_system_callable_placeholder() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(
            template_source=PromptTemplateSource(inline="today=${ts}"),
            placeholders=[PlaceholderSpec(name="ts", value=lambda: "2026-01-01")],
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert "2026-01-01" in rendered.text


@pytest.mark.asyncio
async def test_render_system_callable_non_string_raises() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(
            template_source=PromptTemplateSource(inline="${ts}"),
            placeholders=[PlaceholderSpec(name="ts", value=lambda: 42)],
        ),
    )

    with pytest.raises(ValueError):
        await SystemPromptRenderSystem().process(world)


@pytest.mark.asyncio
async def test_render_system_file_template(tmp_path: Path) -> None:
    file_path = tmp_path / "system-prompt.txt"
    file_path.write_text("Hello ${name}", encoding="utf-8")

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(
            template_source=PromptTemplateSource(file_path=str(file_path)),
            placeholders=[PlaceholderSpec(name="name", value="Alice")],
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "Hello Alice"


@pytest.mark.asyncio
async def test_render_system_missing_file_raises() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(
            template_source=PromptTemplateSource(file_path="/nonexistent/path.txt")
        ),
    )

    with pytest.raises((ValueError, FileNotFoundError)):
        await SystemPromptRenderSystem().process(world)


@pytest.mark.asyncio
async def test_render_system_mirrors_to_llm_component() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigSpec(
            template_source=PromptTemplateSource(inline="Hello ${name}"),
            placeholders=[PlaceholderSpec(name="name", value="Alice")],
        ),
    )
    world.add_component(
        entity_id,
        LLMComponent(provider=FakeProvider(responses=[]), model="fake"),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    llm = world.get_component(entity_id, LLMComponent)
    assert rendered is not None
    assert llm is not None
    assert llm.system_prompt == rendered.text
