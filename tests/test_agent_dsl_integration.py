"""End-to-end integration tests for Agent DSL pipeline.

Tests full workflow: discover → load → resolve → compile → execute.
Uses FakeProvider for deterministic, network-free testing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    PermissionComponent,
    SubagentRegistryComponent,
)
from ecs_agent.prompts.contracts import SystemPromptConfigSpec
from ecs_agent.core import Runner
from ecs_agent.dsl import (
    compile_agent_specs,
    discover_agent_sources,
    load_json_agents,
    load_markdown_agent,
    resolve_agent_specs,
)
from ecs_agent.dsl.schema import AgentSpec
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.types import CompletionResult, Message


# ============================================================================
# Test Helper Functions
# ============================================================================


def create_temp_json_file(directory: Path, filename: str, content: str) -> Path:
    """Create a temporary JSON agent file."""
    filepath = directory / filename
    filepath.write_text(content, encoding="utf-8")
    return filepath


def create_temp_markdown_file(directory: Path, filename: str, content: str) -> Path:
    """Create a temporary Markdown agent file."""
    filepath = directory / filename
    filepath.write_text(content, encoding="utf-8")
    return filepath


def create_fake_provider_factory(responses: list[CompletionResult]):
    """Create a provider factory that returns FakeProvider instances."""

    def factory(model: str, system_prompt: str):
        # Each subagent gets its own provider instance
        return FakeProvider(responses=responses.copy())

    return factory


# ============================================================================
# Test 1: JSON DSL → World → Run (Basic Happy Path)
# ============================================================================


@pytest.mark.asyncio
async def test_json_dsl_to_world_to_run_basic_workflow() -> None:
    """Load JSON file with primary agent spec, compile to World, run with FakeProvider."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create JSON agent file
        json_content = """{
            "assistant": {
                "mode": "primary",
                "model": "fake-model",
                "prompt": "You are a helpful assistant."
            }
        }"""
        create_temp_json_file(tmpdir_path, "agents.json", json_content)

        # Load and resolve
        specs = load_json_agents(tmpdir_path / "agents.json")
        resolved = resolve_agent_specs(specs)

        # Compile to World
        responses = [
            CompletionResult(
                message=Message(role="assistant", content="Hello! I can help you.")
            )
        ]
        factory = create_fake_provider_factory(responses)
        primary_entity, world = compile_agent_specs(resolved, factory)

        # Add conversation and register systems
        world.add_component(
            primary_entity,
            ConversationComponent(messages=[Message(role="user", content="Hi there!")]),
        )
        world.register_system(ReasoningSystem(priority=0), priority=0)

        # Run
        runner = Runner()
        await runner.run(world, max_ticks=3)

        # Verify conversation contains expected response
        conv = world.get_component(primary_entity, ConversationComponent)
        assert conv is not None
        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"
        assert conv.messages[0].content == "Hi there!"
        assert conv.messages[1].role == "assistant"
        assert conv.messages[1].content == "Hello! I can help you."


# ============================================================================
# Test 2: Markdown DSL → World → Run
# ============================================================================


@pytest.mark.asyncio
async def test_markdown_dsl_to_world_to_run_basic_workflow() -> None:
    """Load Markdown file with frontmatter config, compile to World, run with FakeProvider."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create Markdown agent file
        markdown_content = """---
mode: primary
model: fake-model
---

You are a code reviewer. Provide constructive feedback.
"""
        create_temp_markdown_file(tmpdir_path, "reviewer.md", markdown_content)

        # Load and resolve
        spec = load_markdown_agent(tmpdir_path / "reviewer.md")
        resolved = resolve_agent_specs([spec])

        # Compile to World
        responses = [
            CompletionResult(
                message=Message(role="assistant", content="Code looks good overall.")
            )
        ]
        factory = create_fake_provider_factory(responses)
        primary_entity, world = compile_agent_specs(resolved, factory)

        # Verify system prompt from markdown body
        llm = world.get_component(primary_entity, LLMComponent)
        assert llm is not None
        assert (
            llm.system_prompt
            == "You are a code reviewer. Provide constructive feedback."
        )

        # Add conversation and register systems
        world.add_component(
            primary_entity,
            ConversationComponent(
                messages=[Message(role="user", content="Review this code.")]
            ),
        )
        world.register_system(ReasoningSystem(priority=0), priority=0)

        # Run
        runner = Runner()
        await runner.run(world, max_ticks=3)

        # Verify conversation contains expected response
        conv = world.get_component(primary_entity, ConversationComponent)
        assert conv is not None
        assert len(conv.messages) == 2
        assert conv.messages[1].content == "Code looks good overall."


def test_subagent_runtime_prompt_component_uses_compiled_prompt_template() -> None:
    resolved = resolve_agent_specs(
        [
            AgentSpec(
                name="orchestrator",
                mode="primary",
                model="primary-model",
                prompt="Primary prompt",
            ),
            AgentSpec(
                name="researcher",
                mode="subagent",
                model="research-model",
                prompt="Research prompt from DSL",
            ),
        ]
    )
    primary_entity, world = compile_agent_specs(
        resolved,
        create_fake_provider_factory([]),
    )

    registry = world.get_component(primary_entity, SubagentRegistryComponent)
    assert registry is not None
    config = registry.subagents["researcher"]

    subagent_system = SubagentSystem()
    child_world, child_entity = subagent_system._assemble_child_world(
        world,
        primary_entity,
        config,
    )

    child_spec = child_world.get_component(child_entity, SystemPromptConfigSpec)
    assert child_spec is not None
    assert child_spec.template_source.inline is not None
    assert child_spec.template_source.inline.startswith("Research prompt from DSL")


# ============================================================================
# Test 3: Mixed JSON + Markdown → Run (Last-One-Wins)
# ============================================================================


@pytest.mark.asyncio
async def test_mixed_json_markdown_last_one_wins_workflow() -> None:
    """Load both JSON and Markdown files with same agent name, verify last-one-wins."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create JSON agent file
        json_content = """{
            "assistant": {
                "mode": "primary",
                "model": "json-model",
                "prompt": "JSON system prompt"
            }
        }"""
        create_temp_json_file(tmpdir_path, "agents.json", json_content)

        # Create Markdown agent file with same name
        markdown_content = """---
mode: primary
model: markdown-model
---

Markdown system prompt
"""
        create_temp_markdown_file(tmpdir_path, "assistant.md", markdown_content)

        # Discover and load (Markdown loaded after JSON)
        sources = discover_agent_sources(tmpdir_path)
        all_specs = []
        for source in sources:
            if str(source).endswith(".json"):
                all_specs.extend(load_json_agents(source))
            else:
                all_specs.append(load_markdown_agent(source))

        # Resolve (last-one-wins)
        resolved = resolve_agent_specs(all_specs)

        # Verify winner's config
        assert "assistant" in resolved
        winner = resolved["assistant"]
        assert winner.model == "markdown-model"
        assert winner.prompt == "Markdown system prompt"

        # Compile and run
        responses = [
            CompletionResult(
                message=Message(
                    role="assistant", content="Response from markdown config"
                )
            )
        ]
        factory = create_fake_provider_factory(responses)
        primary_entity, world = compile_agent_specs(resolved, factory)

        # Verify LLMComponent has markdown config
        llm = world.get_component(primary_entity, LLMComponent)
        assert llm is not None
        assert llm.model == "markdown-model"
        assert llm.system_prompt == "Markdown system prompt"


# ============================================================================
# Test 4: Subagent Registry Integration
# ============================================================================


@pytest.mark.asyncio
async def test_subagent_registry_integration() -> None:
    """Load DSL with 1 primary + 2 subagents, verify SubagentRegistryComponent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create JSON with primary and 2 subagents
        json_content = """{
            "orchestrator": {
                "mode": "primary",
                "model": "primary-model",
                "prompt": "I coordinate tasks."
            },
            "researcher": {
                "mode": "subagent",
                "model": "research-model",
                "prompt": "I research topics."
            },
            "writer": {
                "mode": "subagent",
                "model": "write-model",
                "prompt": "I write content."
            }
        }"""
        create_temp_json_file(tmpdir_path, "agents.json", json_content)

        # Load and resolve
        specs = load_json_agents(tmpdir_path / "agents.json")
        resolved = resolve_agent_specs(specs)

        # Compile to World
        responses = [
            CompletionResult(
                message=Message(role="assistant", content="Task coordinated.")
            )
        ]
        factory = create_fake_provider_factory(responses)
        primary_entity, world = compile_agent_specs(resolved, factory)

        # Verify SubagentRegistryComponent contains 2 configs
        registry = world.get_component(primary_entity, SubagentRegistryComponent)
        assert registry is not None
        assert len(registry.subagents) == 2
        assert "researcher" in registry.subagents
        assert "writer" in registry.subagents

        # Verify subagent names and models match DSL
        researcher = registry.subagents["researcher"]
        assert researcher.name == "researcher"
        assert researcher.model == "research-model"
        assert researcher.system_prompt == "I research topics."

        writer = registry.subagents["writer"]
        assert writer.name == "writer"
        assert writer.model == "write-model"
        assert writer.system_prompt == "I write content."


# ============================================================================
# Test 5: Permission Enforcement - Allowed Tools
# ============================================================================


@pytest.mark.asyncio
async def test_permission_enforcement_allowed_tools() -> None:
    """Load DSL with tools: {allowed_tool: true}, verify PermissionComponent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create JSON with allowed tools
        json_content = """{
            "assistant": {
                "mode": "primary",
                "model": "fake-model",
                "prompt": "You are a helpful assistant.",
                "tools": {
                    "search": true,
                    "calculate": true
                }
            }
        }"""
        create_temp_json_file(tmpdir_path, "agents.json", json_content)

        # Load and resolve
        specs = load_json_agents(tmpdir_path / "agents.json")
        resolved = resolve_agent_specs(specs)

        # Compile to World
        responses = [
            CompletionResult(message=Message(role="assistant", content="Tools ready."))
        ]
        factory = create_fake_provider_factory(responses)
        primary_entity, world = compile_agent_specs(resolved, factory)

        # Verify PermissionComponent.allowed_tools
        permission = world.get_component(primary_entity, PermissionComponent)
        assert permission is not None
        assert set(permission.allowed_tools) == {"search", "calculate"}
        assert permission.denied_tools == []


# ============================================================================
# Test 6: Permission Enforcement - Mixed Allow/Deny
# ============================================================================


@pytest.mark.asyncio
async def test_permission_enforcement_mixed_tools() -> None:
    """Load DSL with tools: {allowed: true, denied: false}, verify filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create JSON with mixed tools
        json_content = """{
            "assistant": {
                "mode": "primary",
                "model": "fake-model",
                "prompt": "You are restricted.",
                "tools": {
                    "read_file": true,
                    "write_file": false,
                    "execute_code": false
                }
            }
        }"""
        create_temp_json_file(tmpdir_path, "agents.json", json_content)

        # Load and resolve
        specs = load_json_agents(tmpdir_path / "agents.json")
        resolved = resolve_agent_specs(specs)

        # Compile to World
        responses = [
            CompletionResult(
                message=Message(role="assistant", content="Permission granted.")
            )
        ]
        factory = create_fake_provider_factory(responses)
        primary_entity, world = compile_agent_specs(resolved, factory)

        # Verify PermissionComponent only includes allowed_tools
        permission = world.get_component(primary_entity, PermissionComponent)
        assert permission is not None
        assert permission.allowed_tools == ["read_file"]
        assert permission.denied_tools == []


# ============================================================================
# Test 7: Full Pipeline - Discover → Load → Resolve → Compile → Run
# ============================================================================


@pytest.mark.asyncio
async def test_full_pipeline_discover_to_run() -> None:
    """Test complete pipeline from discovery to execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create multiple agent files
        json_content = """{
            "primary_agent": {
                "mode": "primary",
                "model": "fake-model",
                "prompt": "I am the primary agent."
            }
        }"""
        create_temp_json_file(tmpdir_path, "agents.json", json_content)

        markdown_content = """---
mode: subagent
model: helper-model
---

I am a helper agent.
"""
        create_temp_markdown_file(tmpdir_path, "helper.md", markdown_content)

        # Full pipeline: discover
        sources = discover_agent_sources(tmpdir_path)
        assert len(sources) == 2

        # Load
        all_specs = []
        for source in sources:
            if str(source).endswith(".json"):
                all_specs.extend(load_json_agents(source))
            else:
                all_specs.append(load_markdown_agent(source))
        assert len(all_specs) == 2

        # Resolve
        resolved = resolve_agent_specs(all_specs)
        assert len(resolved) == 2
        assert "primary_agent" in resolved
        assert "helper" in resolved

        # Compile
        responses = [
            CompletionResult(
                message=Message(role="assistant", content="Pipeline complete.")
            )
        ]
        factory = create_fake_provider_factory(responses)
        primary_entity, world = compile_agent_specs(resolved, factory)

        # Verify World state
        llm = world.get_component(primary_entity, LLMComponent)
        assert llm is not None
        assert llm.model == "fake-model"

        registry = world.get_component(primary_entity, SubagentRegistryComponent)
        assert registry is not None
        assert len(registry.subagents) == 1
        assert "helper" in registry.subagents

        # Run
        world.add_component(
            primary_entity,
            ConversationComponent(
                messages=[Message(role="user", content="Test pipeline.")]
            ),
        )
        world.register_system(ReasoningSystem(priority=0), priority=0)

        runner = Runner()
        await runner.run(world, max_ticks=3)

        # Verify execution
        conv = world.get_component(primary_entity, ConversationComponent)
        assert conv is not None
        assert len(conv.messages) == 2
        assert conv.messages[1].content == "Pipeline complete."


# ============================================================================
# Test 8: Error Handling - Multiple Primary Agents
# ============================================================================


def test_compile_error_multiple_primaries() -> None:
    """Test compile error when multiple primary agents are provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create JSON with multiple primaries
        json_content = """{
            "agent1": {
                "mode": "primary",
                "model": "model1",
                "prompt": "Primary one"
            },
            "agent2": {
                "mode": "primary",
                "model": "model2",
                "prompt": "Primary two"
            }
        }"""
        create_temp_json_file(tmpdir_path, "agents.json", json_content)

        # Load and resolve
        specs = load_json_agents(tmpdir_path / "agents.json")
        resolved = resolve_agent_specs(specs)

        # Attempt to compile (should raise ValueError)
        factory = create_fake_provider_factory([])
        with pytest.raises(
            ValueError, match="Expected exactly one primary agent, found 2"
        ):
            compile_agent_specs(resolved, factory)


# ============================================================================
# Test 9: Error Handling - Missing Primary Agent
# ============================================================================


def test_compile_error_missing_primary() -> None:
    """Test compile error when no primary agent is provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create JSON with only subagents
        json_content = """{
            "helper1": {
                "mode": "subagent",
                "model": "helper-model",
                "prompt": "I help"
            },
            "helper2": {
                "mode": "subagent",
                "model": "helper-model2",
                "prompt": "I also help"
            }
        }"""
        create_temp_json_file(tmpdir_path, "agents.json", json_content)

        # Load and resolve
        specs = load_json_agents(tmpdir_path / "agents.json")
        resolved = resolve_agent_specs(specs)

        # Attempt to compile (should raise ValueError)
        factory = create_fake_provider_factory([])
        with pytest.raises(
            ValueError, match="Expected exactly one primary agent, found 0"
        ):
            compile_agent_specs(resolved, factory)


# ============================================================================
# Test 10: No Tools Specified - No PermissionComponent
# ============================================================================


@pytest.mark.asyncio
async def test_no_tools_no_permission_component() -> None:
    """Verify that PermissionComponent is not added when tools field is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create JSON without tools
        json_content = """{
            "assistant": {
                "mode": "primary",
                "model": "fake-model",
                "prompt": "No tools here."
            }
        }"""
        create_temp_json_file(tmpdir_path, "agents.json", json_content)

        # Load and resolve
        specs = load_json_agents(tmpdir_path / "agents.json")
        resolved = resolve_agent_specs(specs)

        # Compile to World
        responses = [
            CompletionResult(
                message=Message(role="assistant", content="No tools configured.")
            )
        ]
        factory = create_fake_provider_factory(responses)
        primary_entity, world = compile_agent_specs(resolved, factory)

        # Verify PermissionComponent is not present
        permission = world.get_component(primary_entity, PermissionComponent)
        assert permission is None


# ============================================================================
# Test 11: Empty Tools Dict - No PermissionComponent
# ============================================================================


@pytest.mark.asyncio
async def test_empty_tools_dict_no_permission_component() -> None:
    """Verify that PermissionComponent is not added when tools dict is empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create JSON with empty tools
        json_content = """{
            "assistant": {
                "mode": "primary",
                "model": "fake-model",
                "prompt": "Empty tools.",
                "tools": {}
            }
        }"""
        create_temp_json_file(tmpdir_path, "agents.json", json_content)

        # Load and resolve
        specs = load_json_agents(tmpdir_path / "agents.json")
        resolved = resolve_agent_specs(specs)

        # Compile to World
        responses = [
            CompletionResult(
                message=Message(role="assistant", content="No tools allowed.")
            )
        ]
        factory = create_fake_provider_factory(responses)
        primary_entity, world = compile_agent_specs(resolved, factory)

        # Verify PermissionComponent is not present
        permission = world.get_component(primary_entity, PermissionComponent)
        assert permission is None


# ============================================================================
# Test 12: Subagent With Tools - No PermissionComponent on Primary
# ============================================================================


@pytest.mark.asyncio
async def test_subagent_tools_no_permission_on_primary() -> None:
    """Verify that tools on subagents do not affect primary PermissionComponent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create JSON with primary (no tools) and subagent (with tools)
        json_content = """{
            "orchestrator": {
                "mode": "primary",
                "model": "primary-model",
                "prompt": "I orchestrate."
            },
            "worker": {
                "mode": "subagent",
                "model": "worker-model",
                "prompt": "I work.",
                "tools": {
                    "hammer": true,
                    "drill": true
                }
            }
        }"""
        create_temp_json_file(tmpdir_path, "agents.json", json_content)

        # Load and resolve
        specs = load_json_agents(tmpdir_path / "agents.json")
        resolved = resolve_agent_specs(specs)

        # Compile to World
        responses = [
            CompletionResult(
                message=Message(role="assistant", content="Orchestration ready.")
            )
        ]
        factory = create_fake_provider_factory(responses)
        primary_entity, world = compile_agent_specs(resolved, factory)

        # Verify PermissionComponent is not on primary
        permission = world.get_component(primary_entity, PermissionComponent)
        assert permission is None

        # Verify subagent in registry
        registry = world.get_component(primary_entity, SubagentRegistryComponent)
        assert registry is not None
        assert "worker" in registry.subagents


def test_compile_attaches_system_prompt_config_spec_not_legacy_component() -> None:
    from ecs_agent.components import SystemPromptComponent

    resolved = resolve_agent_specs(
        [AgentSpec(name="bot", mode="primary", model="m", prompt="You are a bot.")]
    )
    primary_entity, world = compile_agent_specs(
        resolved, create_fake_provider_factory([])
    )

    config_spec = world.get_component(primary_entity, SystemPromptConfigSpec)
    assert config_spec is not None
    assert config_spec.template_source.inline == "You are a bot."
    assert config_spec.placeholders == []

    legacy = world.get_component(primary_entity, SystemPromptComponent)
    assert legacy is None


def test_compile_with_placeholders_builds_correct_placeholder_specs() -> None:
    from ecs_agent.prompts.contracts import PlaceholderSpec

    spec = AgentSpec(
        name="bot",
        mode="primary",
        model="m",
        prompt="You are ${role}.",
        placeholders=[{"name": "role", "value": "a helpful assistant"}],
    )
    primary_entity, world = compile_agent_specs(
        resolve_agent_specs([spec]), create_fake_provider_factory([])
    )

    config_spec = world.get_component(primary_entity, SystemPromptConfigSpec)
    assert config_spec is not None
    assert len(config_spec.placeholders) == 1
    assert isinstance(config_spec.placeholders[0], PlaceholderSpec)
    assert config_spec.placeholders[0].name == "role"
    assert config_spec.placeholders[0].value == "a helpful assistant"


def test_compile_with_triggers_attaches_user_prompt_config_component() -> None:
    from ecs_agent.components.definitions import UserPromptConfigComponent
    from ecs_agent.prompts.contracts import TriggerSpec

    spec = AgentSpec(
        name="bot",
        mode="primary",
        model="m",
        prompt="You are an assistant.",
        triggers=[
            {
                "pattern": "@help",
                "match_mode": "keyword",
                "action": "inject",
                "content": "Show help.",
                "priority": 0,
            }
        ],
    )
    primary_entity, world = compile_agent_specs(
        resolve_agent_specs([spec]), create_fake_provider_factory([])
    )

    upc = world.get_component(primary_entity, UserPromptConfigComponent)
    assert upc is not None
    assert len(upc.triggers) == 1
    t = upc.triggers[0]
    assert isinstance(t, TriggerSpec)
    assert t.pattern == "@help"
    assert t.match_mode == "keyword"
    assert t.action == "inject"
    assert t.content == "Show help."
    assert t.priority == 0


def test_compile_without_triggers_no_user_prompt_config_component() -> None:
    from ecs_agent.components.definitions import UserPromptConfigComponent

    resolved = resolve_agent_specs(
        [AgentSpec(name="bot", mode="primary", model="m", prompt="You are a bot.")]
    )
    primary_entity, world = compile_agent_specs(
        resolved, create_fake_provider_factory([])
    )

    upc = world.get_component(primary_entity, UserPromptConfigComponent)
    assert upc is None


def test_compile_auto_registers_system_prompt_render_system() -> None:
    from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem

    resolved = resolve_agent_specs(
        [AgentSpec(name="bot", mode="primary", model="m", prompt="You are a bot.")]
    )
    _primary_entity, world = compile_agent_specs(
        resolved, create_fake_provider_factory([])
    )

    world._systems.apply_queued_operations()
    registered_types = [type(entry.system) for entry in world._systems._systems]
    assert SystemPromptRenderSystem in registered_types


def test_compile_with_triggers_registers_user_prompt_normalization_system() -> None:
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )

    spec = AgentSpec(
        name="bot",
        mode="primary",
        model="m",
        prompt="You are an assistant.",
        triggers=[
            {
                "pattern": "@help",
                "match_mode": "keyword",
                "action": "inject",
                "content": "Show help.",
                "priority": 0,
            }
        ],
    )
    _primary_entity, world = compile_agent_specs(
        resolve_agent_specs([spec]), create_fake_provider_factory([])
    )

    world._systems.apply_queued_operations()
    registered_types = [type(entry.system) for entry in world._systems._systems]
    assert UserPromptNormalizationSystem in registered_types


def test_compile_without_triggers_no_user_prompt_normalization_system() -> None:
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )

    resolved = resolve_agent_specs(
        [AgentSpec(name="bot", mode="primary", model="m", prompt="You are a bot.")]
    )
    _primary_entity, world = compile_agent_specs(
        resolved, create_fake_provider_factory([])
    )

    world._systems.apply_queued_operations()
    registered_types = [type(entry.system) for entry in world._systems._systems]
    assert UserPromptNormalizationSystem not in registered_types
