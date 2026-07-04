from __future__ import annotations

from pathlib import Path

import pytest

from ecs_agent.components import (
    CompactionConfigComponent,
    CurrentCompactionSummaryComponent,
    LLMComponent,
    RenderedSystemPromptComponent,
    RenderedUserPromptComponent,
    SkillComponent,
    SkillMetadata,
    SubagentRegistryComponent,
    SystemPromptComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.observability import RecordingTelemetrySink, install_observability
from ecs_agent.prompts.contracts import (
    PlaceholderSpec,
    SystemPromptConfigSpec,
    PromptTemplateSource,
    TriggerSpec,
)
import ecs_agent.prompts.provider as prompt_provider_module
from ecs_agent.providers import FakeModel
from ecs_agent.prompts.registry import resolve_placeholder_values
from ecs_agent.scratchbook import (
    ScratchbookArtifactPromptDef,
    ScratchbookPromptConfig,
)
import ecs_agent.systems.system_prompt_render_system as render_module
import ecs_agent.prompts.template_render as template_render_module
from ecs_agent.systems.system_prompt_render_system import (
    SystemPromptRenderSystem,
    render_compaction_prompt,
)
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.types import FreeSubagentConfig, SubagentConfig, ToolSchema


class TerminatingSystem:
    """Terminates a runner after prompt systems have processed."""

    async def process(self, world: World) -> None:
        """Attach a terminal component to stop the runner."""
        world.add_component(world.create_entity(), TerminalComponent(reason="done"))


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
    spec = SystemPromptConfigSpec(
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


def test_component_rendered_user_prompt_component_has_text() -> None:
    component = RenderedUserPromptComponent(text="normalized user prompt")

    assert component.text == "normalized user prompt"


@pytest.mark.asyncio
async def test_render_system_renders_inline_template_and_bridges_to_llm() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
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
    world.add_component(entity_id, LLMComponent(model=object()))

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
        "_cache_key": "inventory:tools:write_file|skills:|subagents:|mcps:",
    }
    assert llm.system_prompt == rendered.text


@pytest.mark.asyncio
async def test_system_prompt_replacement_emits_prompt_telemetry() -> None:
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="Hello ${user_name}"),
            placeholders=[PlaceholderSpec(name="user_name", value="roy")],
        ),
    )
    world.register_system(SystemPromptRenderSystem(), priority=0)
    world.register_system(TerminatingSystem(), priority=10)

    await Runner().run(world, max_ticks=1)

    records = [record for record in sink.records if record.name == "prompt.system.replacement"]
    assert len(records) == 1
    record = records[0]
    assert record.kind == "event"
    assert record.entity_id == int(entity_id)
    assert record.input == {"text": "Hello ${user_name}"}
    assert record.output == {"text": "Hello roy"}
    assert record.metadata == {
        "prompt_kind": "system",
        "replacements": {"user_name": "roy"},
        "system_name": "ecs_agent.systems.system_prompt_render_system.SystemPromptRenderSystem",
    }


@pytest.mark.asyncio
async def test_static_system_prompt_emits_no_prompt_render_tracing() -> None:
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(template_source=PromptTemplateSource(inline="Always terse.")),
    )
    world.register_system(SystemPromptRenderSystem(), priority=0)
    world.register_system(TerminatingSystem(), priority=10)

    await Runner().run(world, max_ticks=1)

    assert not any(record.name == "prompt.system.replacement" for record in sink.records)
    assert not any(record.name.endswith("SystemPromptRenderSystem") for record in sink.records)


@pytest.mark.asyncio
async def test_render_system_writes_component_for_inline_template_without_placeholders() -> (
    None
):
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="Always terse.")
        ),
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
        SystemPromptConfigSpec(
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
    assert rendered.text == "Skills:\n- alpha: a\n- beta: b"


@pytest.mark.asyncio
async def test_render_system_renders_all_builtin_placeholders() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
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
                "planner": SubagentConfig(name="planner", model=object()),
                "researcher": SubagentConfig(
                    name="researcher", model=object()
                ),
            }
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == (
        "- bash\n- read\n- alpha: a\n- zeta: z\n- none\n- planner\n- researcher"
    )


@pytest.mark.asyncio
async def test_render_system_rejects_unknown_placeholder() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${missing}")
        ),
    )

    with pytest.raises(ValueError, match="unknown placeholders"):
        await SystemPromptRenderSystem().process(world)


@pytest.mark.asyncio
async def test_render_system_rejects_missing_template_file() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
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
        SystemPromptConfigSpec(
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
        SystemPromptConfigSpec(
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
        SystemPromptConfigSpec(
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
        SystemPromptConfigSpec(
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
        SystemPromptConfigSpec(
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
async def test_installed_tools_omit_descriptions_but_skills_keep_them() -> None:
    """ISSUE-7: tool descriptions are already carried by the native tool schema,
    so ${_installed_tools} lists names only; skills (no native schema) keep them."""
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline="Tools:\n${_installed_tools}\nSkills:\n${_installed_skills}"
            )
        ),
    )
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "search": ToolSchema(
                    name="search",
                    description="Search the web for a query",
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
                "navigator": SkillMetadata(
                    name="navigator",
                    description="Guide a design flow",
                    tool_names=[],
                    has_system_prompt=False,
                ),
            }
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    # Tool: name only (description is in the native schema, not duplicated here).
    assert "- search\n" in rendered.text
    assert "Search the web for a query" not in rendered.text
    # Skill: description retained (progressive disclosure, no native schema).
    assert "- navigator: Guide a design flow" in rendered.text


@pytest.mark.asyncio
async def test_render_system_empty_inventory_renders_none() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
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
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${unknown_var}")
        ),
    )

    with pytest.raises(ValueError):
        await SystemPromptRenderSystem().process(world)


@pytest.mark.asyncio
async def test_render_system_callable_placeholder() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
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
        SystemPromptConfigSpec(
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
        SystemPromptConfigSpec(
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
        SystemPromptConfigSpec(
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
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="Hello ${name}"),
            placeholders=[PlaceholderSpec(name="name", value="Alice")],
        ),
    )
    world.add_component(
        entity_id,
        LLMComponent(model=FakeModel(responses=[])),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    llm = world.get_component(entity_id, LLMComponent)
    assert rendered is not None
    assert llm is not None
    assert llm.system_prompt == rendered.text


@pytest.mark.asyncio
async def test_render_system_installed_skills_sorted() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_skills}")
        ),
    )
    world.add_component(
        entity_id,
        SkillComponent(
            skills={
                "code": SkillMetadata(
                    name="code",
                    description="code skill",
                    tool_names=[],
                    has_system_prompt=False,
                ),
                "analyze": SkillMetadata(
                    name="analyze",
                    description="analyze skill",
                    tool_names=[],
                    has_system_prompt=False,
                ),
            }
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "- analyze: analyze skill\n- code: code skill"


@pytest.mark.asyncio
async def test_render_system_empty_skills_renders_none() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_skills}")
        ),
    )
    world.add_component(entity_id, SkillComponent(skills={}))

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "- none"


@pytest.mark.asyncio
async def test_render_system_no_skill_component_renders_none() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_skills}")
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "- none"


@pytest.mark.asyncio
async def test_installed_skills_placeholder_renders_summary_only() -> None:
    """Contract: ${_installed_skills} renders Tier-1 name+description only."""
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="Skills:\n${_installed_skills}")
        ),
    )
    world.add_component(
        entity_id,
        SkillComponent(
            skills={
                "alpha": SkillMetadata(
                    name="alpha",
                    description="alpha summary",
                    tool_names=["tool_a", "tool_b"],
                    has_system_prompt=True,
                )
            }
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "Skills:\n- alpha: alpha summary"
    assert "##" not in rendered.text
    assert "parameters" not in rendered.text
    assert "tool_a" not in rendered.text


@pytest.mark.asyncio
async def test_render_system_installed_subagents_sorted() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_subagents}")
        ),
    )
    world.add_component(
        entity_id,
        SubagentRegistryComponent(
            subagents={
                "researcher": SubagentConfig(
                    name="researcher",
                    model=object(),
                ),
                "analyst": SubagentConfig(
                    name="analyst",
                    model=object(),
                ),
            }
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "- analyst\n- researcher"


@pytest.mark.asyncio
async def test_render_system_empty_subagents_renders_none() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_subagents}")
        ),
    )
    world.add_component(entity_id, SubagentRegistryComponent(subagents={}))

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "- none"


@pytest.mark.asyncio
async def test_render_system_free_subagents_mentions_unregistered_categories() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_subagents}")
        ),
    )
    world.add_component(
        entity_id,
        SubagentRegistryComponent(
            free_subagent_config=FreeSubagentConfig(enabled=True),
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert "Free-form subagents enabled" in rendered.text
    assert "unregistered" in rendered.text


@pytest.mark.asyncio
async def test_render_system_global_free_subagent_preinstall_exposes_prompt_hint() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=object()))
    world.add_component(entity_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_subagents}")
        ),
    )

    await SubagentSystem(allow_unregistered_subagents=True).process(world)
    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert "Free-form subagents enabled" in rendered.text


@pytest.mark.asyncio
async def test_render_system_all_builtins_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMCPClientComponent:
        def __init__(self, cached_tools: list[dict[str, str]]) -> None:
            self.cached_tools = cached_tools

    monkeypatch.setattr(
        prompt_provider_module,
        "_MCP_CLIENT_COMPONENT_CLASS",
        FakeMCPClientComponent,
    )

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline=(
                    "Tools:\n${_installed_tools}\n"
                    "Skills:\n${_installed_skills}\n"
                    "MCPs:\n${_installed_mcps}\n"
                    "Subagents:\n${_installed_subagents}"
                )
            )
        ),
    )
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "bash": ToolSchema(
                    name="bash",
                    description="bash",
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
                "python": SkillMetadata(
                    name="python",
                    description="python skill",
                    tool_names=[],
                    has_system_prompt=False,
                )
            }
        ),
    )
    world.add_component(
        entity_id,
        SubagentRegistryComponent(
            subagents={
                "child": SubagentConfig(name="child", model=object())
            }
        ),
    )

    world.add_component(
        entity_id,
        FakeMCPClientComponent(
            cached_tools=[
                {"name": "filesystem.read"},
                {"name": "filesystem.write"},
            ]
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == (
        "Tools:\n- bash\n- read\n"
        "Skills:\n- python: python skill\n"
        "MCPs:\n- filesystem.read\n- filesystem.write\n"
        "Subagents:\n- child"
    )


@pytest.mark.asyncio
async def test_render_system_skill_activate_then_render() -> None:
    world = World()
    entity_id = world.create_entity()
    skill_component = SkillComponent(skills={})
    world.add_component(entity_id, skill_component)
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_skills}")
        ),
    )

    skill_component.skills["python"] = SkillMetadata(
        name="python",
        description="python skill",
        tool_names=[],
        has_system_prompt=False,
        activated=True,
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "- python: python skill"


@pytest.mark.asyncio
async def test_system_prompt_render_system_invalidates_when_installed_skills_change() -> (
    None
):
    world = World()
    entity_id = world.create_entity()
    skill_component = SkillComponent(
        skills={
            "python": SkillMetadata(
                name="python",
                description="python skill",
                tool_names=[],
                has_system_prompt=False,
            ),
        }
    )
    world.add_component(entity_id, skill_component)
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_skills}")
        ),
    )

    await SystemPromptRenderSystem().process(world)
    first_render = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert first_render is not None
    assert first_render.text == "- python: python skill"

    skill_component.skills["shell"] = SkillMetadata(
        name="shell",
        description="shell skill",
        tool_names=[],
        has_system_prompt=False,
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "- python: python skill\n- shell: shell skill"


@pytest.mark.asyncio
async def test_system_prompt_render_system_renders_once_when_no_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_skills}"),
        ),
    )
    world.add_component(entity_id, SkillComponent(skills={}))

    call_count = 0
    original_render = render_module._render_system_prompt

    def _counting_render(
        target_world: World,
        target_entity_id: object,
        prompt_config: SystemPromptConfigSpec,
    ) -> tuple[str, dict[str, str]]:
        nonlocal call_count
        call_count += 1
        return original_render(target_world, target_entity_id, prompt_config)

    monkeypatch.setattr(render_module, "_render_system_prompt", _counting_render)

    system = SystemPromptRenderSystem()
    await system.process(world)
    await system.process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "- none"
    assert call_count == 1


def test_render_compaction_prompt_does_not_mutate_runtime_prompt_state() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline="Hello ${name}\n${_installed_tools}"
            ),
            placeholders=[PlaceholderSpec(name="name", value="Compaction")],
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
    world.add_component(
        entity_id,
        RenderedSystemPromptComponent(
            text="runtime cache",
            placeholder_snapshot={"_cache_key": "cached"},
        ),
    )
    world.add_component(
        entity_id,
        LLMComponent(
            model=FakeModel(responses=[]),
            system_prompt="live llm prompt",
        ),
    )
    world.add_component(
        entity_id,
        SystemPromptComponent(content="legacy runtime prompt"),
    )

    rendered = render_compaction_prompt(
        template="Hello ${name}\n${_installed_tools}",
        world=world,
        entity=entity_id,
    )

    runtime_cache = world.get_component(entity_id, RenderedSystemPromptComponent)
    llm = world.get_component(entity_id, LLMComponent)
    legacy_prompt = world.get_component(entity_id, SystemPromptComponent)
    assert rendered == "Hello Compaction\n- write_file"
    assert runtime_cache is not None
    assert runtime_cache.text == "runtime cache"
    assert runtime_cache.placeholder_snapshot == {"_cache_key": "cached"}
    assert llm is not None
    assert llm.system_prompt == "live llm prompt"
    assert legacy_prompt is not None
    assert legacy_prompt.content == "legacy runtime prompt"


@pytest.mark.asyncio
async def test_system_prompt_render_system_cached_process_still_bridges_runtime_state() -> (
    None
):
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_tools}"),
        ),
    )
    world.add_component(entity_id, SkillComponent(skills={}))
    world.add_component(
        entity_id,
        LLMComponent(
            model=FakeModel(responses=[]),
            system_prompt="stale llm prompt",
        ),
    )
    world.add_component(
        entity_id,
        SystemPromptComponent(content="stale legacy prompt"),
    )

    system = SystemPromptRenderSystem()
    await system.process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "- none"

    llm = world.get_component(entity_id, LLMComponent)
    legacy_prompt = world.get_component(entity_id, SystemPromptComponent)
    assert llm is not None
    assert legacy_prompt is not None
    llm.system_prompt = "changed after initial render"
    legacy_prompt.content = "legacy changed after initial render"

    await system.process(world)

    assert llm.system_prompt == "- none"
    assert legacy_prompt.content == "- none"


@pytest.mark.asyncio
async def test_system_prompt_render_system_bridges_legacy_system_prompt_component() -> (
    None
):
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="Hello ${name}"),
            placeholders=[PlaceholderSpec(name="name", value="Bridge")],
        ),
    )
    world.add_component(entity_id, SystemPromptComponent())

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    legacy_prompt = world.get_component(entity_id, SystemPromptComponent)
    assert rendered is not None
    assert legacy_prompt is not None
    assert legacy_prompt.content == rendered.text == "Hello Bridge"


@pytest.mark.asyncio
async def test_compaction_summary_xml_empty_block_for_new_entity() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline="Base prompt\n${_chat_history_summary_xml}"
            )
        ),
    )
    world.add_component(entity_id, CompactionConfigComponent(threshold_tokens=100))

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text.endswith("<chat_history_summary></chat_history_summary>")
    assert rendered.text.count("<chat_history_summary>") == 1
    assert rendered.placeholder_snapshot["_chat_history_summary_xml"] == (
        "<chat_history_summary></chat_history_summary>"
    )


@pytest.mark.asyncio
async def test_compaction_summary_xml_with_content() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline="Base prompt\n${_chat_history_summary_xml}"
            )
        ),
    )
    world.add_component(entity_id, CompactionConfigComponent(threshold_tokens=100))
    world.add_component(
        entity_id,
        CurrentCompactionSummaryComponent(summary="hello world"),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text.endswith(
        "<chat_history_summary>hello world</chat_history_summary>"
    )


@pytest.mark.asyncio
async def test_compaction_summary_xml_escapes_special_chars() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline="Base prompt\n${_chat_history_summary_xml}"
            )
        ),
    )
    world.add_component(entity_id, CompactionConfigComponent(threshold_tokens=100))
    world.add_component(
        entity_id,
        CurrentCompactionSummaryComponent(
            summary="<tag>&value></chat_history_summary>"
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text.endswith(
        "<chat_history_summary>"
        "&lt;tag&gt;&amp;value&gt;&lt;/chat_history_summary&gt;"
        "</chat_history_summary>"
    )


def test_compaction_summary_fingerprint_changes_with_summary() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, CompactionConfigComponent(threshold_tokens=100))
    model = prompt_provider_module.CompactionSummaryPlaceholderProvider()

    empty_fingerprint = model.provider_fingerprint(world, entity_id)

    world.add_component(
        entity_id,
        CurrentCompactionSummaryComponent(summary="first summary"),
    )
    first_fingerprint = model.provider_fingerprint(world, entity_id)

    world.add_component(
        entity_id,
        CurrentCompactionSummaryComponent(summary="second summary"),
    )
    second_fingerprint = model.provider_fingerprint(world, entity_id)

    world.remove_component(entity_id, CurrentCompactionSummaryComponent)
    cleared_fingerprint = model.provider_fingerprint(world, entity_id)

    assert empty_fingerprint != first_fingerprint
    assert first_fingerprint != second_fingerprint
    assert second_fingerprint != cleared_fingerprint


@pytest.mark.asyncio
async def test_compaction_summary_cache_invalidates_on_summary_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline="Base prompt\n${_chat_history_summary_xml}"
            )
        ),
    )
    world.add_component(entity_id, CompactionConfigComponent(threshold_tokens=100))

    call_count = 0
    original_render = render_module._render_system_prompt

    def _counting_render(
        target_world: World,
        target_entity_id: object,
        prompt_config: SystemPromptConfigSpec,
    ) -> tuple[str, dict[str, str]]:
        nonlocal call_count
        call_count += 1
        return original_render(target_world, target_entity_id, prompt_config)

    monkeypatch.setattr(render_module, "_render_system_prompt", _counting_render)

    system = SystemPromptRenderSystem()

    await system.process(world)
    assert call_count == 1

    await system.process(world)
    assert call_count == 1

    world.add_component(
        entity_id,
        CurrentCompactionSummaryComponent(summary="first summary"),
    )
    await system.process(world)
    assert call_count == 2

    await system.process(world)
    assert call_count == 2

    world.add_component(
        entity_id,
        CurrentCompactionSummaryComponent(summary="second summary"),
    )
    await system.process(world)
    assert call_count == 3

    world.remove_component(entity_id, CurrentCompactionSummaryComponent)
    await system.process(world)
    assert call_count == 4


@pytest.mark.asyncio
async def test_legacy_agent_gets_xml_tail_without_explicit_placeholder() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, CompactionConfigComponent(threshold_tokens=100))
    world.add_component(
        entity_id,
        LLMComponent(
            model=FakeModel(responses=[]),
            system_prompt="Legacy base",
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    llm = world.get_component(entity_id, LLMComponent)
    assert rendered is not None
    assert llm is not None
    # Full text keeps its historical shape; the split only adds stable/volatile.
    assert rendered.text == (
        "Legacy base\n<chat_history_summary></chat_history_summary>"
    )
    assert rendered.stable_text == "Legacy base"
    assert rendered.volatile_text == "<chat_history_summary></chat_history_summary>"
    assert llm.system_prompt == rendered.text


@pytest.mark.asyncio
async def test_rerender_legacy_agent_after_summary_change_no_double_block() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, CompactionConfigComponent(threshold_tokens=100))
    world.add_component(
        entity_id,
        LLMComponent(
            model=FakeModel(responses=[]),
            system_prompt="You are helpful.",
        ),
    )
    world.add_component(
        entity_id,
        CurrentCompactionSummaryComponent(summary="first summary"),
    )

    system = SystemPromptRenderSystem()
    await system.process(world)

    rendered_first = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered_first is not None
    first_text = rendered_first.text
    assert first_text.count("<chat_history_summary>") == 1
    assert "first summary" in first_text

    world.remove_component(entity_id, RenderedSystemPromptComponent)
    world.remove_component(entity_id, CurrentCompactionSummaryComponent)
    world.add_component(
        entity_id,
        CurrentCompactionSummaryComponent(summary="second summary"),
    )

    await system.process(world)

    rendered_second = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered_second is not None
    second_text = rendered_second.text
    assert second_text.count("<chat_history_summary>") == 1, (
        f"Expected exactly one <chat_history_summary> block, got:\n{second_text}"
    )
    assert "second summary" in second_text
    assert "first summary" not in second_text


class _ContractProvider:
    def __init__(
        self,
        provider_id: str,
        values: dict[str, str],
        fingerprint: str = "v1",
    ) -> None:
        self.provider_id = provider_id
        self._values = values
        self.fingerprint = fingerprint

    def resolve(self, _world: World, _entity_id: object) -> dict[str, str]:
        return dict(self._values)

    def resolve_placeholders(self, _world: World, _entity_id: object) -> dict[str, str]:
        return self.resolve(_world, _entity_id)

    def provider_fingerprint(self, _world: World, _entity_id: object) -> str:
        return self.fingerprint


def _require_provider_seam_contract_surface() -> None:
    assert hasattr(template_render_module, "_BUILTIN_PLACEHOLDER_PROVIDERS"), (
        "expected model seam registry: _BUILTIN_PLACEHOLDER_PROVIDERS"
    )


@pytest.mark.asyncio
async def test_builtin_provider_registration_requires_provider_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_provider_seam_contract_surface()

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_tools}"),
        ),
    )

    class _ProviderWithoutId:
        def resolve(self, _world: World, _entity_id: object) -> dict[str, str]:
            return {"_installed_tools": "- from-model"}

    monkeypatch.setattr(
        template_render_module,
        "_BUILTIN_PLACEHOLDER_PROVIDERS",
        [_ProviderWithoutId()],
        raising=False,
    )

    with pytest.raises(ValueError, match="provider_id"):
        await SystemPromptRenderSystem().process(world)


@pytest.mark.asyncio
async def test_duplicate_provider_keys_raise_with_provider_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_provider_seam_contract_surface()

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(template_source=PromptTemplateSource(inline="${_foo}")),
    )

    monkeypatch.setattr(
        template_render_module,
        "_BUILTIN_PLACEHOLDER_PROVIDERS",
        [
            _ContractProvider("alpha_provider", {"_foo": "a"}),
            _ContractProvider("beta_provider", {"_foo": "b"}),
        ],
        raising=False,
    )

    with pytest.raises(ValueError, match="duplicate built-in key") as exc_info:
        await SystemPromptRenderSystem().process(world)

    message = str(exc_info.value)
    assert "alpha_provider" in message
    assert "beta_provider" in message


@pytest.mark.asyncio
async def test_user_placeholder_collision_with_builtin_provider_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_provider_seam_contract_surface()

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${installed_tools}"),
            placeholders=[
                PlaceholderSpec(name="installed_tools", value="- user-installed")
            ],
        ),
    )
    monkeypatch.setattr(
        template_render_module,
        "_BUILTIN_PLACEHOLDER_PROVIDERS",
        [
            _ContractProvider(
                "collision_provider",
                {"installed_tools": "- model-installed"},
            )
        ],
        raising=False,
    )

    with pytest.raises(ValueError, match="installed_tools"):
        await SystemPromptRenderSystem().process(world)


@pytest.mark.asyncio
async def test_provider_exception_propagates_from_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_provider_seam_contract_surface()

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(template_source=PromptTemplateSource(inline="${_foo}")),
    )

    class _ExplodingProvider:
        provider_id = "exploding_provider"

        def resolve_placeholders(
            self,
            _world: World,
            _entity_id: object,
        ) -> dict[str, str]:
            raise RuntimeError("provider_exception")

        def provider_fingerprint(self, _world: World, _entity_id: object) -> str:
            return "v1"

    monkeypatch.setattr(
        template_render_module,
        "_BUILTIN_PLACEHOLDER_PROVIDERS",
        [_ExplodingProvider()],
        raising=False,
    )

    with pytest.raises(RuntimeError, match="provider_exception"):
        await SystemPromptRenderSystem().process(world)


@pytest.mark.asyncio
async def test_builtin_provider_merge_order_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_provider_seam_contract_surface()

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_first}|${_second}")
        ),
    )

    monkeypatch.setattr(
        template_render_module,
        "_BUILTIN_PLACEHOLDER_PROVIDERS",
        [
            _ContractProvider("provider_a", {"_first": "first"}),
            _ContractProvider("provider_b", {"_second": "second"}),
        ],
        raising=False,
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "first|second"


@pytest.mark.asyncio
async def test_provider_fingerprint_changes_force_rerender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_provider_seam_contract_surface()

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_tools}"),
        ),
    )

    model = _ContractProvider("inventory_provider", {"_installed_tools": "- none"})
    monkeypatch.setattr(
        template_render_module,
        "_BUILTIN_PLACEHOLDER_PROVIDERS",
        [model],
        raising=False,
    )

    call_count = 0
    original_render = render_module._render_system_prompt

    def _counting_render(
        target_world: World,
        target_entity_id: object,
        prompt_config: SystemPromptConfigSpec,
    ) -> tuple[str, dict[str, str]]:
        nonlocal call_count
        call_count += 1
        return original_render(target_world, target_entity_id, prompt_config)

    monkeypatch.setattr(render_module, "_render_system_prompt", _counting_render)

    system = SystemPromptRenderSystem()
    await system.process(world)
    model.fingerprint = "v2"
    await system.process(world)

    assert call_count == 2


def test_user_placeholder_name_with_underscore_prefix_still_raises() -> None:
    with pytest.raises(ValueError, match="reserved"):
        PlaceholderSpec(name="_provider_owned", value="x")


@pytest.mark.asyncio
async def test_scratchbook_placeholder_names_are_approved_builtin_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_provider_seam_contract_surface()

    class _ScratchbookConfig:
        path: str = ".sisyphus/notepads/scratchbook-prompt-model"
        artifact_types: tuple[str, ...] = ("plan", "report")

    class _ScratchbookProvider(_ContractProvider):
        def resolve(self, world: World, entity_id: object) -> dict[str, str]:
            config = world.get_component(entity_id, _ScratchbookConfig)
            if config is None:
                return {}
            return {
                "_scratchbook_overview": "overview",
                "_scratchbook_path": config.path,
                "_scratchbook_artifact_types": "plan,report",
                "_scratchbook_artifacts": "- plan\n- report",
                "_scratchbook_artifact_plan": "plan summary",
                "_scratchbook_artifact_report": "report summary",
                "_scratchbook_artifact_path_plan": "scratchbook/plan.md",
                "_scratchbook_artifact_path_report": "scratchbook/report.md",
            }

    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, _ScratchbookConfig())
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_scratchbook_path}"),
        ),
    )
    monkeypatch.setattr(
        template_render_module,
        "_BUILTIN_PLACEHOLDER_PROVIDERS",
        [_ScratchbookProvider("scratchbook", {})],
        raising=False,
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert {
        "_scratchbook_overview",
        "_scratchbook_path",
        "_scratchbook_artifact_types",
        "_scratchbook_artifacts",
        "_scratchbook_artifact_plan",
        "_scratchbook_artifact_report",
        "_scratchbook_artifact_path_plan",
        "_scratchbook_artifact_path_report",
    }.issubset(set(rendered.placeholder_snapshot))


@pytest.mark.asyncio
async def test_inventory_builtins_render_through_provider_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_provider_seam_contract_surface()

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline=(
                    "${_installed_tools}\n${_installed_skills}\n"
                    "${_installed_mcps}\n${_installed_subagents}"
                )
            ),
        ),
    )
    monkeypatch.setattr(
        template_render_module,
        "_BUILTIN_PLACEHOLDER_PROVIDERS",
        [
            _ContractProvider(
                "inventory_provider",
                {
                    "_installed_tools": "- from-model-tools",
                    "_installed_skills": "- from-model-skills",
                    "_installed_mcps": "- from-model-mcps",
                    "_installed_subagents": "- from-model-subagents",
                },
            )
        ],
        raising=False,
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == (
        "- from-model-tools\n"
        "- from-model-skills\n"
        "- from-model-mcps\n"
        "- from-model-subagents"
    )


@pytest.mark.asyncio
async def test_missing_builtin_placeholder_still_raises_after_provider_refactor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_provider_seam_contract_surface()

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_tools}"),
        ),
    )
    monkeypatch.setattr(
        template_render_module,
        "_BUILTIN_PLACEHOLDER_PROVIDERS",
        [],
        raising=False,
    )

    with pytest.raises(ValueError, match="unknown placeholders"):
        await SystemPromptRenderSystem().process(world)


@pytest.mark.asyncio
async def test_rendered_system_prompt_bridges_to_llm_and_legacy_components_after_provider_refactor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_provider_seam_contract_surface()

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_provider_text}"),
        ),
    )
    world.add_component(entity_id, LLMComponent(model=object()))
    world.add_component(entity_id, SystemPromptComponent())
    monkeypatch.setattr(
        template_render_module,
        "_BUILTIN_PLACEHOLDER_PROVIDERS",
        [_ContractProvider("provider_bridge", {"_provider_text": "bridge me"})],
        raising=False,
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    llm = world.get_component(entity_id, LLMComponent)
    legacy = world.get_component(entity_id, SystemPromptComponent)
    assert rendered is not None
    assert llm is not None
    assert legacy is not None
    assert rendered.text == "bridge me"
    assert llm.system_prompt == "bridge me"
    assert legacy.content == "bridge me"


@pytest.mark.asyncio
async def test_absent_scratchbook_provider_does_not_change_existing_render_behavior() -> (
    None
):
    _require_provider_seam_contract_surface()

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_tools}"),
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "- none"
    assert "_scratchbook_path" not in rendered.placeholder_snapshot


@pytest.mark.asyncio
async def test_scratchbook_provider_fingerprint_changes_cache_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_provider_seam_contract_surface()

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_installed_tools}"),
        ),
    )
    scratchbook_config = ScratchbookPromptConfig(
        overview_default_template="Overview ${scratchbook_path}\n${artifact_types}",
        scratchbook_root_path="scratchbook/a",
        artifacts=[
            ScratchbookArtifactPromptDef(
                artifact_type_id="plan",
                path="scratchbook/a/plan.md",
                purpose="Execution plan",
                readonly=False,
                read_when="When updating plan status.",
            )
        ],
    )
    world.add_component(entity_id, scratchbook_config)

    call_count = 0
    original_render = render_module._render_system_prompt

    def _counting_render(
        target_world: World,
        target_entity_id: object,
        prompt_config: SystemPromptConfigSpec,
    ) -> tuple[str, dict[str, str]]:
        nonlocal call_count
        call_count += 1
        return original_render(target_world, target_entity_id, prompt_config)

    monkeypatch.setattr(render_module, "_render_system_prompt", _counting_render)

    system = SystemPromptRenderSystem()
    await system.process(world)

    scratchbook_config.artifacts.append(
        ScratchbookArtifactPromptDef(
            artifact_type_id="report",
            path="scratchbook/a/report.md",
            purpose="Execution report",
            readonly=True,
            read_when="When summarizing outcomes.",
        )
    )
    await system.process(world)

    assert call_count == 2


@pytest.mark.asyncio
async def test_scratchbook_provider_placeholders_render_into_system_prompt() -> None:
    _require_provider_seam_contract_surface()

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="Path=${_scratchbook_path}"),
        ),
    )
    world.add_component(
        entity_id,
        ScratchbookPromptConfig(
            overview_default_template="Overview ${scratchbook_path}\n${artifact_types}",
            scratchbook_root_path=".sisyphus/notepads/demo",
            artifacts=[
                ScratchbookArtifactPromptDef(
                    artifact_type_id="plan",
                    path=".sisyphus/notepads/demo/plan.md",
                    purpose="Track plan state",
                    readonly=False,
                    read_when="While executing plan tasks.",
                )
            ],
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "Path=.sisyphus/notepads/demo"
    assert "_scratchbook_overview" in rendered.placeholder_snapshot
    assert "_scratchbook_path" in rendered.placeholder_snapshot
    assert "_scratchbook_artifact_types" in rendered.placeholder_snapshot
    assert "_scratchbook_artifacts" in rendered.placeholder_snapshot
    assert "_scratchbook_artifact_plan" in rendered.placeholder_snapshot
    assert "_scratchbook_artifact_path_plan" in rendered.placeholder_snapshot


@pytest.mark.asyncio
async def test_system_prompt_values_can_reference_builtins_recursively() -> None:
    _require_provider_seam_contract_surface()

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${outer}"),
            placeholders=[
                PlaceholderSpec(name="outer", value="Tools:\n${_installed_tools}"),
            ],
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "Tools:\n- none"


@pytest.mark.asyncio
async def test_recursive_system_prompt_cycle_raises_instead_of_leaking_placeholder() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${outer}"),
            placeholders=[
                PlaceholderSpec(name="outer", value="${inner}"),
                PlaceholderSpec(name="inner", value="${outer}"),
            ],
        ),
    )

    with pytest.raises(ValueError, match="recursive placeholder expansion"):
        await SystemPromptRenderSystem().process(world)


@pytest.mark.asyncio
async def test_recursive_system_prompt_preserves_escaped_placeholders() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="literal=$${_installed_tools}"),
        ),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "literal=${_installed_tools}"


@pytest.mark.asyncio
async def test_provider_resolution_preserves_existing_installed_placeholder_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_provider_seam_contract_surface()

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline=(
                    "${_installed_tools}\n${_installed_skills}\n"
                    "${_installed_mcps}\n${_installed_subagents}"
                )
            )
        ),
    )
    monkeypatch.setattr(
        template_render_module,
        "_BUILTIN_PLACEHOLDER_PROVIDERS",
        [
            _ContractProvider(
                "inventory_provider",
                {
                    "_installed_tools": "- none",
                    "_installed_skills": "- none",
                    "_installed_mcps": "- none",
                    "_installed_subagents": "- none",
                },
            )
        ],
        raising=False,
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert rendered.text == "- none\n- none\n- none\n- none"


# ---------------------------------------------------------------------------
# ISSUE-6: stable/volatile split for prompt-cache prefix stability
# ---------------------------------------------------------------------------


def test_split_stable_volatile_relocates_volatile_to_tail() -> None:
    from ecs_agent.systems.system_prompt_render_system import _split_stable_volatile

    template = (
        "Base instructions\n"
        "${_installed_tools}\n"
        "${_phase_prompt}\n"
        "${_chat_history_summary_xml}"
    )
    snapshot = {
        "_installed_tools": "- toolA",
        "_phase_prompt": "WF-STATE-BODY",
        "_chat_history_summary_xml": "<chat_history_summary>S</chat_history_summary>",
    }

    stable, volatile = _split_stable_volatile(template, snapshot)

    # Stable prefix keeps static content, drops volatile values entirely.
    assert "Base instructions" in stable
    assert "- toolA" in stable
    assert "WF-STATE-BODY" not in stable
    assert "chat_history_summary" not in stable
    # Volatile tail carries both, workflow first (deterministic order).
    assert volatile == (
        "WF-STATE-BODY\n\n<chat_history_summary>S</chat_history_summary>"
    )


def test_split_stable_volatile_is_byte_stable_across_volatile_changes() -> None:
    from ecs_agent.systems.system_prompt_render_system import _split_stable_volatile

    template = (
        "Base instructions\n"
        "${_installed_tools}\n"
        "${_phase_prompt}\n"
        "${_chat_history_summary_xml}"
    )
    base_snapshot = {
        "_installed_tools": "- toolA",
        "_phase_prompt": "state-1",
        "_chat_history_summary_xml": "<chat_history_summary>first</chat_history_summary>",
    }
    changed_snapshot = {
        "_installed_tools": "- toolA",
        "_phase_prompt": "state-2-different-length",
        "_chat_history_summary_xml": (
            "<chat_history_summary>second summary text</chat_history_summary>"
        ),
    }

    stable_a, _ = _split_stable_volatile(template, base_snapshot)
    stable_b, _ = _split_stable_volatile(template, changed_snapshot)

    # ISSUE-6 acceptance: only volatile values differ -> stable prefix identical.
    assert stable_a == stable_b


def test_split_stable_volatile_expands_nested_placeholder_in_volatile_value() -> None:
    from ecs_agent.systems.system_prompt_render_system import _split_stable_volatile

    # Whole prompt body comes from a volatile placeholder whose value itself
    # references a non-volatile placeholder (the workflow-prompt scenario).
    template = "${_phase_prompt}\n${_chat_history_summary_xml}"
    snapshot = {
        "_phase_prompt": "You are agent.\n${_installed_tools}",
        "_installed_tools": "- toolA",
        "_chat_history_summary_xml": "<chat_history_summary></chat_history_summary>",
    }

    stable, volatile = _split_stable_volatile(template, snapshot)

    # All-volatile body -> empty stable prefix (nothing to cache).
    assert stable == ""
    # Nested non-volatile placeholder is expanded, not left as literal.
    assert "${_installed_tools}" not in volatile
    assert "- toolA" in volatile
    assert volatile.startswith("You are agent.\n- toolA")


def test_split_stable_volatile_no_volatile_keys_leaves_full_equal_stable() -> None:
    from ecs_agent.systems.system_prompt_render_system import _split_stable_volatile

    template = "Base instructions\n${_installed_tools}"
    snapshot = {"_installed_tools": "- toolA"}

    stable, volatile = _split_stable_volatile(template, snapshot)

    assert volatile == ""
    assert stable == "Base instructions\n- toolA"


@pytest.mark.asyncio
async def test_render_system_populates_stable_and_volatile_fields() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline="Base prompt\n${_chat_history_summary_xml}"
            )
        ),
    )
    world.add_component(entity_id, CompactionConfigComponent(threshold_tokens=100))
    world.add_component(
        entity_id,
        CurrentCompactionSummaryComponent(summary="hello world"),
    )

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    # Stable prefix excludes the summary; volatile tail carries it.
    assert "chat_history_summary" not in rendered.stable_text
    assert rendered.stable_text == "Base prompt"
    assert rendered.volatile_text == (
        "<chat_history_summary>hello world</chat_history_summary>"
    )
    # Full text is the faithful render (unchanged historical whitespace).
    assert rendered.text == (
        "Base prompt\n<chat_history_summary>hello world</chat_history_summary>"
    )
