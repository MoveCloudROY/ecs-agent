"""Behavioral regressions for DSL assembly and controlled agent editing."""

import json
from copy import deepcopy
from contextlib import contextmanager
import subprocess
import sys
import asyncio
from pathlib import Path

import pytest

from ecs_agent.components import RenderedSystemPromptComponent, ToolRegistryComponent
from ecs_agent.core import World
from ecs_agent.dsl import AgentSpec, compile_agent_specs, load_json_agents, load_markdown_agent
from ecs_agent.prompts.contracts import PlaceholderSpec, PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.providers import FakeModel
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.tools import register_tools, tool
from ecs_agent import AgentDefinition, ModelRef, RuntimeConfig, SkillRef, SystemStep, build_agent, build_agent_from_specs, inspect_runtime
from ecs_agent.types import CompletionResult, Message, ToolCall


class RecordingModel(FakeModel):
    """Capture the actual provider boundary for authoring parity tests."""

    def __init__(self, responses, model_id="fake"):
        super().__init__(responses, model_id=model_id)
        self.requests = []

    async def complete(self, messages, tools=None, **kwargs):
        self.requests.append((deepcopy(messages), deepcopy(tools)))
        return await super().complete(messages, tools=tools, **kwargs)


def make_skill(root: Path, name: str = "research") -> None:
    """Create a deterministic file skill with useful loadable instructions."""
    directory = root / "skills" / "demo"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: research instructions\n---\nVerified evidence only.\n"
    )


async def test_prompt_updates_template_values_and_dynamic_values_once() -> None:
    world = World()
    entity = world.create_entity()
    render = SystemPromptRenderSystem()
    world.add_component(entity, SystemPromptConfigSpec(
        PromptTemplateSource(inline="Hello ${name}"), [PlaceholderSpec("name", "Ada")],
    ))
    await render.process(world)
    assert world.get_component(entity, RenderedSystemPromptComponent).text == "Hello Ada"
    calls = 0

    def dynamic() -> str:
        nonlocal calls
        calls += 1
        return f"Bob {calls}"

    world.add_component(entity, SystemPromptConfigSpec(
        PromptTemplateSource(inline="Goodbye ${name}"), [PlaceholderSpec("name", dynamic)],
    ))
    await render.process(world)
    assert world.get_component(entity, RenderedSystemPromptComponent).text == "Goodbye Bob 1"
    await render.process(world)
    assert world.get_component(entity, RenderedSystemPromptComponent).text == "Goodbye Bob 2"
    assert calls == 2


@pytest.mark.parametrize("kind", ["markdown", "json"])
async def test_loaded_skill_keeps_origin_and_tools_merge(tmp_path: Path, kind: str) -> None:
    make_skill(tmp_path)
    if kind == "markdown":
        path = tmp_path / "main.md"
        path.write_text("---\nmode: primary\nmodel: fake\nskills:\n  - path: skills/demo\n---\nYou research.\n")
        spec = load_markdown_agent(path)
    else:
        path = tmp_path / "agents.json"
        path.write_text(json.dumps({"main": {
            "mode": "primary", "model": "fake", "prompt": "You research.",
            "skills": [{"path": "skills/demo"}],
        }}))
        spec = load_json_agents(path)[0]
    entity, world = compile_agent_specs(
        {"main": spec}, lambda *_: FakeModel([]), source_dir=tmp_path / "wrong-fallback",
    )

    @tool()
    def extra() -> str:
        """Return an extra result."""
        return "extra"

    register_tools(world, entity, [extra])
    registry = world.get_component(entity, ToolRegistryComponent)
    assert registry is not None
    assert set(registry.tools) == set(registry.handlers) == {"load_skill_details", "extra"}
    assert "Verified evidence only" in await registry.handlers["load_skill_details"](skill_name="research")
    assert await registry.handlers["extra"]() == "extra"


@pytest.mark.parametrize("base", [None, "missing"])
def test_invalid_skill_fails_before_provider_creation(tmp_path: Path, base: str | None) -> None:
    calls = []

    def factory(*args: str) -> FakeModel:
        calls.append(args)
        return FakeModel([])

    spec = AgentSpec("primary", "fake", "Research", name="main", skills=[{"path": "absent"}])
    with pytest.raises(ValueError, match=r"main\.skills\[0\]\.path"):
        compile_agent_specs({"main": spec}, factory, source_dir=None if base is None else tmp_path)
    assert calls == []


def test_skill_symlink_cannot_escape_source(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    make_skill(tmp_path / "outside")
    (root / "shortcut").symlink_to(tmp_path / "outside" / "skills" / "demo")
    spec = AgentSpec("primary", "fake", "Research", skills=[{"path": "shortcut"}])
    with pytest.raises(ValueError, match="escapes"):
        compile_agent_specs({"main": spec}, lambda *_: FakeModel([]), source_dir=root)


async def test_complete_builder_executes_tools_without_manual_systems() -> None:
    calls = []

    @tool()
    def total(values: list[int]) -> str:
        """Sum integer inputs."""
        calls.append(values)
        return str(sum(values))

    model = RecordingModel([
        CompletionResult(Message("assistant", "", tool_calls=[ToolCall("sum-1", "total", {"values": [2, 3]})])),
        CompletionResult(Message("assistant", "5")),
    ])
    agent = build_agent(AgentDefinition(model, prompt="Compute carefully.", tools=[total]))
    result = await agent.run_turn("Add 2 and 3")
    assert result.success and result.final_message.content == "5"
    assert calls == [[2, 3]]
    assert any(message.role == "tool" and message.content == "5" for message in model.requests[1][0])
    report = inspect_runtime(agent)
    assert report.complete
    assert dict(report.tools) == {"total": "Python tool"}
    assert [step.name for step in report.systems] == ["normalize", "render", "reason", "execute", "errors"]
    assert dict(report.component_sources)["LLMComponent"].startswith("AgentDefinition.model")
    assert report.concurrency_groups == ((-25, ("normalize",)), (-20, ("render",)), (0, ("reason",)), (5, ("execute",)), (99, ("errors",)))


async def test_python_and_dsl_build_equivalent_skill_requests(tmp_path: Path) -> None:
    make_skill(tmp_path)
    responses = [CompletionResult(Message("assistant", "ready"))]
    python_model, dsl_model = RecordingModel(responses), RecordingModel(responses)
    python = build_agent(AgentDefinition(
        python_model, prompt=SystemPromptConfigSpec(PromptTemplateSource(inline="Hello ${name}"), [PlaceholderSpec("name", "Ada")]),
        skills=[SkillRef("skills/demo", tmp_path)], permissions={"load_skill_details": True}, name="main",
    ))
    spec = AgentSpec(
        "primary", "fake", "Hello ${name}", name="main", tools={"load_skill_details": True},
        placeholders=[{"name": "name", "value": "Ada"}], skills=[{"path": "skills/demo"}],
    )
    dsl = build_agent_from_specs({"main": spec}, lambda *_: dsl_model, source_dir=tmp_path)
    assert (await python.run_turn("hello")).success
    assert (await dsl.run_turn("hello")).success
    python_messages, python_tools = python_model.requests[0]
    dsl_messages, dsl_tools = dsl_model.requests[0]
    assert [(m.role, m.content) for m in python_messages] == [(m.role, m.content) for m in dsl_messages]
    assert python_tools == dsl_tools
    assert inspect_runtime(python).systems == inspect_runtime(dsl).systems


async def test_dsl_child_fields_reach_actual_delegated_request(tmp_path: Path) -> None:
    from ecs_agent.components import SkillComponent

    make_skill(tmp_path)
    parent = RecordingModel([
        CompletionResult(Message("assistant", "", tool_calls=[ToolCall("delegate-1", "subagent", {"category": "researcher", "prompt": "/ask question"})])),
        CompletionResult(Message("assistant", "parent completed")),
    ], model_id="parent")
    child = RecordingModel([
        CompletionResult(Message("assistant", "", tool_calls=[ToolCall("details-1", "load_skill_details", {"skill_name": "research"})])),
        CompletionResult(Message("assistant", "child completed")),
    ], model_id="child")
    specs = {
        "main": AgentSpec("primary", "parent", "Delegate."),
        "researcher": AgentSpec(
            "subagent", "child", "You are ${role}.", placeholders=[{"name": "role", "value": "a researcher"}],
            skills=[{"path": "skills/demo"}], triggers=[{"pattern": "/ask", "match_mode": "prefix", "action": "replace", "content": "normalized task"}],
        ),
    }
    agent = build_agent_from_specs(specs, lambda name, _: parent if name == "parent" else child, source_dir=tmp_path)
    result = await agent.run_turn("start", max_steps=6)
    assert result.success
    assert child.requests
    messages, schemas = child.requests[0]
    assert any(m.role == "system" and "You are a researcher." in m.content for m in messages)
    assert any(m.role == "user" and "normalized task" in m.content for m in messages)
    assert "load_skill_details" in {schema.name for schema in schemas}
    assert any(m.role == "tool" and "Verified evidence only" in m.content for m in child.requests[1][0])
    assert not agent.world.has_component(agent.entity, SkillComponent)


def test_bad_child_definition_is_rejected_before_any_model_factory(tmp_path: Path) -> None:
    calls = []

    def factory(*args):
        calls.append(args)
        return FakeModel([])

    specs = {
        "main": AgentSpec("primary", "parent", "Delegate."),
        "child": AgentSpec("subagent", "child", "You are ${missing}."),
    }
    with pytest.raises(ValueError, match="Unknown prompt placeholders"):
        build_agent_from_specs(specs, factory)
    assert calls == []


async def test_named_runtime_dependencies_and_repeat_installation() -> None:
    from ecs_agent.authoring.scheduling import install_steps

    trace = []

    class Write:
        async def process(self, world):
            trace.append("write")

    class Read:
        async def process(self, world):
            assert trace[-1] == "write"
            trace.append("read")

    steps = (SystemStep("read", Read, after=("write",)), SystemStep("write", Write, priority=-5))
    world = World()
    assert install_steps(world, steps) == install_steps(world, steps)
    await world.process()
    assert trace == ["write", "read"]
    with pytest.raises(ValueError, match="Conflicting installed"):
        install_steps(world, (SystemStep("write", Read, priority=-5),))


@pytest.mark.parametrize("steps", [
    (),
    (SystemStep("a", lambda: None, after=("missing",)),),
    (SystemStep("a", lambda: None, after=("b",)), SystemStep("b", lambda: None, after=("a",))),
])
def test_incomplete_or_cyclic_runtime_never_creates_provider(steps) -> None:
    calls = []
    with pytest.raises(ValueError):
        build_agent(AgentDefinition(ModelRef("test")), runtime=RuntimeConfig(steps=steps), model_factory=lambda *args: calls.append(args))
    assert calls == []


def test_definition_cycle_is_rejected() -> None:
    definition = AgentDefinition(FakeModel([]))
    definition.subagents["self"] = definition
    with pytest.raises(ValueError, match="Recursive"):
        build_agent(definition)


def test_base_import_does_not_load_optional_sdks() -> None:
    subprocess.run([sys.executable, "-c", "import sys; from ecs_agent import build_agent; assert not {'mcp', 'langfuse', 'prometheus_client'}.intersection(sys.modules)"], check=True)


@pytest.fixture(params=["memory", "sqlite"])
def recording_service(request, tmp_path):
    from ecs_agent.sessions import InMemorySessionStore, SQLiteSessionStore, SessionService
    store = InMemorySessionStore() if request.param == "memory" else SQLiteSessionStore(tmp_path / "sessions.sqlite")
    service = SessionService(store)
    yield service
    service.close()


async def test_prompt_edit_changes_actual_next_request_and_checkpoint(tmp_path: Path) -> None:
    from ecs_agent.core import AgentHandle
    from ecs_agent.serialization import WorldSerializer
    from ecs_agent.authoring.runtime import core_steps
    from ecs_agent.authoring.scheduling import install_steps

    model = RecordingModel([CompletionResult(Message("assistant", "first")), CompletionResult(Message("assistant", "second"))])
    agent = build_agent(AgentDefinition(model, prompt="Hello Ada"))
    await agent.run_turn("hello")
    assert agent.inspect_prompt().text == "Hello Ada"
    assert agent.inspect_prompt().cache_key is not None
    agent.set_prompt("Goodbye Bob")
    inspection = agent.inspect_prompt()
    assert inspection.config_revision == 1 and inspection.text is None
    path = tmp_path / "agent.json"
    WorldSerializer.save(agent.world, path)
    restored = WorldSerializer.load(path, {"fake": model}, {})
    install_steps(restored, core_steps())
    loaded = AgentHandle(restored, agent.entity)
    assert (await loaded.run_turn("hello again")).success
    assert any(m.role == "system" and m.content == "Goodbye Bob" for m in model.requests[-1][0])
    assert loaded.inspect_prompt().rendered_revision == 1


async def test_file_prompt_reload_and_inspection_do_not_run_callbacks(tmp_path: Path) -> None:
    path = tmp_path / "prompt.md"
    path.write_text("Old ${name}")
    calls = []

    def value():
        calls.append(True)
        return "Ada"

    config = SystemPromptConfigSpec(PromptTemplateSource(file_path=str(path)), [PlaceholderSpec("name", value)])
    model = RecordingModel([CompletionResult(Message("assistant", "ok")) for _ in range(3)])
    agent = build_agent(AgentDefinition(model, prompt=config))
    assert agent.inspect_prompt().source == str(path.resolve())
    assert calls == []
    await agent.run_turn("one")
    path.write_text("New ${name}")
    await agent.run_turn("two")
    assert agent.inspect_prompt().text == "Old Ada"
    assert calls == [True, True]
    agent.set_prompt(config)
    assert calls == [True, True]
    await agent.run_turn("three")
    assert agent.inspect_prompt().text == "New Ada"
    assert calls == [True, True, True]


def test_prompt_callback_requires_explicit_persistent_binding(tmp_path: Path) -> None:
    from ecs_agent.prompts import PromptBinding, register_prompt_binding
    from ecs_agent.serialization import WorldSerializer

    name = str(tmp_path / "callback")
    register_prompt_binding(name, lambda: "bound")
    agent = build_agent(AgentDefinition(FakeModel([]), prompt=SystemPromptConfigSpec(
        PromptTemplateSource(inline="Hello ${value}"), [PlaceholderSpec("value", PromptBinding(name))],
    )))
    path = tmp_path / "agent.json"
    WorldSerializer.save(agent.world, path)
    restored = WorldSerializer.load(path, {"fake": FakeModel([])}, {})
    config = restored.get_component(agent.entity, SystemPromptConfigSpec)
    assert config.placeholders[0].value() == "bound"
    agent.set_prompt(SystemPromptConfigSpec(PromptTemplateSource(inline="${value}"), [PlaceholderSpec("value", lambda: "unbound")]))
    with pytest.raises(ValueError, match="registered PromptBinding"):
        WorldSerializer.to_dict(agent.world)


def test_history_snapshot_and_revision_guard(recording_service) -> None:
    from ecs_agent.components import SessionRefComponent
    from ecs_agent.types import ImageUrlPart

    agent = build_agent(AgentDefinition(FakeModel([])), session_service=recording_service)
    report = inspect_runtime(agent)
    assert "SessionRefComponent" in report.components
    assert dict(report.component_sources)["SessionRefComponent"] == "SessionService"
    original = [Message("user", "look", parts=[ImageUrlPart("https://example.test/image")]), Message("assistant", "seen")]
    first = agent.history_snapshot()
    agent.replace_history(original, reason="import", expected_history_id=first.history_id, expected_revision=first.revision)
    original[0].content = "caller mutation"
    snapshot = agent.history_snapshot()
    snapshot.messages[0].parts[0].url = "changed"
    assert agent.history_snapshot().messages[0].content == "look"
    assert agent.history_snapshot().messages[0].parts[0].url == "https://example.test/image"
    with pytest.raises(ValueError, match="stale"):
        agent.replace_history([], reason="stale edit", expected_revision=first.revision)
    ref = agent.world.get_component(agent.entity, SessionRefComponent)
    assert recording_service.replay_history(ref.session_id) == list(agent.history_snapshot().messages)


@pytest.mark.parametrize("failure", ["append", "writer", "reset"])
def test_reset_failure_rolls_back_records_catalog_and_writer(recording_service, monkeypatch, failure) -> None:
    from ecs_agent.components import SessionRefComponent

    agent = build_agent(AgentDefinition(FakeModel([])), session_service=recording_service)
    agent.replace_history([Message("user", "original")], reason="seed")
    old_ref = deepcopy(agent.world.get_component(agent.entity, SessionRefComponent))
    old_history = agent.history_snapshot()
    old_catalog = recording_service.store.sessions()
    old_records = list(recording_service.store.records(old_ref.session_id))
    original_append = recording_service.store.append
    original_claim = recording_service.store.claim_writer

    def append(draft, **kwargs):
        if (failure == "append" and draft.session_id != old_ref.session_id) or (failure == "reset" and draft.kind == "reset"):
            raise OSError("injected append failure")
        return original_append(draft, **kwargs)

    def claim(session_id, writer_id):
        if failure == "writer" and session_id != old_ref.session_id:
            raise OSError("injected writer failure")
        return original_claim(session_id, writer_id)

    with monkeypatch.context() as patch:
        patch.setattr(recording_service.store, "append", append)
        patch.setattr(recording_service.store, "claim_writer", claim)
        with pytest.raises(OSError, match="injected"):
            agent.reset_session(reason="new task")
    assert agent.world.get_component(agent.entity, SessionRefComponent) == old_ref
    assert agent.history_snapshot() == old_history
    assert recording_service.store.sessions() == old_catalog
    assert list(recording_service.store.records(old_ref.session_id)) == old_records
    agent.replace_history([Message("user", "writer still works")], reason="check writer")


def test_reset_lost_acknowledgement_recovers_committed_switch(recording_service, monkeypatch) -> None:
    from ecs_agent.components import SessionRefComponent

    agent = build_agent(AgentDefinition(FakeModel([])), session_service=recording_service)
    old_ref = deepcopy(agent.world.get_component(agent.entity, SessionRefComponent))
    original = recording_service.store.transaction

    @contextmanager
    def lost_ack():
        with original():
            yield
        raise OSError("commit acknowledgement lost")

    monkeypatch.setattr(recording_service.store, "transaction", lost_ack)
    agent.reset_session(reason="new task")
    new_ref = agent.world.get_component(agent.entity, SessionRefComponent)
    assert new_ref.session_id != old_ref.session_id
    assert recording_service.store.get_session(new_ref.session_id).source_session_id == old_ref.session_id
    assert recording_service.replay_history(new_ref.session_id) == []
    agent.replace_history([Message("user", "next")], reason="check new writer")
    assert recording_service.replay_history(new_ref.session_id) == [Message("user", "next")]


@pytest.mark.parametrize("operation", ["prompt", "replace", "reset"])
async def test_editing_rejects_unfinished_turns_without_mutation(operation) -> None:
    agent = build_agent(AgentDefinition(FakeModel([]), prompt="original"))
    await agent.run_turn("waiting", max_steps=0)
    snapshot = agent.history_snapshot()
    prompt = agent.world.get_component(agent.entity, SystemPromptConfigSpec)
    with pytest.raises(ValueError, match="current turn"):
        if operation == "prompt":
            agent.set_prompt("changed")
        elif operation == "replace":
            agent.replace_history([], reason="change")
        else:
            agent.reset_session(reason="change")
    assert agent.history_snapshot() == snapshot
    assert agent.world.get_component(agent.entity, SystemPromptConfigSpec) == prompt


async def test_restored_dsl_skills_bind_to_loaded_world_without_resetting_turn(tmp_path: Path) -> None:
    from ecs_agent import restore_runtime
    from ecs_agent.core import AgentHandle
    from ecs_agent.serialization import WorldSerializer
    from ecs_agent.components import SubagentRegistryComponent

    make_skill(tmp_path)
    model = RecordingModel([
        CompletionResult(Message("assistant", "", tool_calls=[ToolCall("restore-details", "load_skill_details", {"skill_name": "research"})])),
        CompletionResult(Message("assistant", "loaded")),
        CompletionResult(Message("assistant", "", tool_calls=[ToolCall("restored-delegate", "subagent", {"category": "child", "prompt": "/task"})])),
        CompletionResult(Message("assistant", "", tool_calls=[ToolCall("child-details", "load_skill_details", {"skill_name": "research"})])),
        CompletionResult(Message("assistant", "restored child answered")),
        CompletionResult(Message("assistant", "restored delegation completed")),
    ])
    specs = {
        "main": AgentSpec("primary", "fake", "Use the skill.", skills=[{"path": "skills/demo"}]),
        "child": AgentSpec("subagent", "fake", "Child ${role}", placeholders=[{"name": "role", "value": "researcher"}], skills=[{"path": "skills/demo"}], triggers=[{"pattern": "/task", "match_mode": "prefix", "action": "replace", "content": "after_restore_normalized_task"}]),
    }
    agent = build_agent_from_specs(specs, lambda *_: model, source_dir=tmp_path)
    paused = await agent.run_turn("load the skill", max_steps=0)
    path = tmp_path / "checkpoint.json"
    WorldSerializer.save(agent.world, path)
    restored = WorldSerializer.load(path, {"fake": model}, {})
    restore_runtime(restored)
    loaded = AgentHandle(restored, agent.entity)
    assert loaded.history_snapshot() == agent.history_snapshot()
    result = await loaded.resume(max_steps=3)
    assert result.success and result.turn_id == paused.turn_id
    assert any(message.role == "tool" and "Verified evidence only" in message.content for message in loaded.history_snapshot().messages)
    child = restored.get_component(agent.entity, SubagentRegistryComponent).subagents["child"]
    assert child.prompt_config.placeholders[0].value == "researcher"
    assert child.local_skills[0].name == "research"
    assert (await loaded.run_turn("delegate after restore", max_steps=4)).success
    assert any(message.role == "system" and "Child researcher" in message.content for message in model.requests[3][0])
    assert any(message.role == "user" and "after_restore_normalized_task" in message.content for message in model.requests[3][0])


def test_framework_collision_and_repeat_binding_are_explicit(tmp_path: Path) -> None:
    from ecs_agent.skills import Skill, SkillManager
    from ecs_agent.tools import get_tool_definition

    make_skill(tmp_path)
    world = World()
    entity = world.create_entity()

    @tool()
    def load_skill_details(skill_name: str) -> str:
        """A conflicting user function."""
        return skill_name

    register_tools(world, entity, [load_skill_details])
    original = world.get_component(entity, ToolRegistryComponent).handlers.copy()
    with pytest.raises(ValueError, match="collision"):
        SkillManager().install(world, entity, Skill(tmp_path / "skills" / "demo" / "SKILL.md"))
    assert world.get_component(entity, ToolRegistryComponent).handlers == original
    assert original["load_skill_details"] is get_tool_definition(load_skill_details).handler


async def test_bound_placeholder_callback_remains_host_owned() -> None:
    class State:
        def __init__(self):
            self.value = "first"
            self.calls = 0

        def read(self):
            self.calls += 1
            return self.value

    state = State()
    model = RecordingModel([CompletionResult(Message("assistant", "ok")) for _ in range(2)])
    agent = build_agent(AgentDefinition(model, prompt=SystemPromptConfigSpec(
        PromptTemplateSource(inline="${value}"), [PlaceholderSpec("value", state.read)],
    )))
    await agent.run_turn("one")
    assert state.calls == 1
    state.value = "second"
    await agent.run_turn("two")
    assert state.calls == 2 and agent.inspect_prompt().text == "second"


async def test_script_trigger_updates_phase_before_render() -> None:
    from ecs_agent.components import UserPromptConfigComponent
    from ecs_agent.prompts import TriggerSpec
    from ecs_agent.phases import PhaseSpec, advance, bind_phase_graph, build_graph

    async def next_phase(world, entity, text):
        await advance(world, entity, "SECOND", reason="user command")
        return "answer in the new phase"

    model = RecordingModel([CompletionResult(Message("assistant", "done"))])
    agent = build_agent(AgentDefinition(model, prompt="${_phase_prompt}"), runtime=RuntimeConfig(user_prompt=UserPromptConfigComponent(
        triggers=[TriggerSpec("/next", "prefix", "script", "next")], script_handlers={"next": next_phase},
    )))
    graph = build_graph("authoring-phase", initial="FIRST", phases=[
        PhaseSpec(phase_id="FIRST", prompts={"main": "first phase"}, to=("SECOND",)),
        PhaseSpec(phase_id="SECOND", prompts={"main": "second phase"}, to=("FIRST",)),
    ])
    await bind_phase_graph(agent.world, agent.entity, graph)
    assert (await agent.run_turn("/next")).success
    assert any(message.role == "system" and "second phase" in message.content for message in model.requests[0][0])


def test_failed_session_install_cleans_skill_resources(recording_service, monkeypatch) -> None:
    calls = []

    class ResourceSkill:
        name = "resource"
        description = "An owned resource"

        def tools(self):
            return {}

        def system_prompt(self):
            return "resource"

        def install(self, world, entity):
            calls.append("install")

        def uninstall(self, world, entity):
            calls.append("uninstall")

    def fail(*args, **kwargs):
        raise OSError("writer unavailable")

    monkeypatch.setattr(recording_service.store, "claim_writer", fail)
    with pytest.raises(OSError, match="writer unavailable"):
        build_agent(AgentDefinition(FakeModel([]), skills=[ResourceSkill()]), session_service=recording_service)
    assert calls == ["install", "uninstall"]
    assert recording_service.store.sessions() == []


async def test_internal_history_change_and_resume_do_not_repeat_input_script() -> None:
    from ecs_agent.components import UserPromptConfigComponent
    from ecs_agent.history import append_conversation_message
    from ecs_agent.prompts import TriggerSpec

    effects = []

    async def script(world, entity, text):
        effects.append(text)
        return "normalized input"

    @tool()
    def noop() -> str:
        """Return a fixed value."""
        return "ok"

    model = RecordingModel([
        CompletionResult(Message("assistant", "", tool_calls=[ToolCall("noop", "noop", {})])),
        CompletionResult(Message("assistant", "done")),
    ])
    agent = build_agent(AgentDefinition(model, tools=[noop]), runtime=RuntimeConfig(user_prompt=UserPromptConfigComponent(
        triggers=[TriggerSpec("/next", "prefix", "script", "next")], script_handlers={"next": script},
    )))
    first = await agent.run_turn("/next", max_steps=1)
    assert not first.success and effects == ["/next"]
    append_conversation_message(agent.world, agent.entity, Message("user", "/next synthetic context"))
    assert (await agent.resume(max_steps=2)).success
    assert effects == ["/next"]


async def test_restore_custom_runtime_requires_original_roles_and_types() -> None:
    from ecs_agent import restore_runtime
    from ecs_agent.core import AgentHandle
    from ecs_agent.serialization import WorldSerializer

    calls = []

    class Capture:
        async def process(self, world):
            calls.append("capture")

    runtime = RuntimeConfig(extra_steps=(SystemStep("capture", Capture, after=("render",), before=("reason",)),))
    model = FakeModel([CompletionResult(Message("assistant", "done"))])
    agent = build_agent(AgentDefinition(model), runtime=runtime)
    restored = WorldSerializer.from_dict(WorldSerializer.to_dict(agent.world), {"fake": model}, {})
    with pytest.raises(ValueError, match="checkpoint manifest"):
        restore_runtime(restored)
    assert not restored._authoring_instances
    restore_runtime(restored, runtime=runtime)
    assert (await AgentHandle(restored, agent.entity).run_turn("hello")).success
    assert calls == ["capture"]


@pytest.mark.parametrize("operation", ["prompt", "history", "reset"])
def test_context_edits_cannot_discard_unknown_tool_effects(operation) -> None:
    from ecs_agent.components import AgentTurnComponent

    agent = build_agent(AgentDefinition(FakeModel([])))
    turn = AgentTurnComponent("pending-effects", status="blocked", unconfirmed_tool_calls=["external-call"])
    agent.world.add_component(agent.entity, turn)
    original = agent.history_snapshot()
    with pytest.raises(ValueError, match="Finish or safely abandon"):
        if operation == "prompt":
            agent.set_prompt("changed")
        elif operation == "history":
            agent.replace_history([], reason="edit")
        else:
            agent.reset_session(reason="reset")
    assert agent.history_snapshot() == original
    assert turn.status == "blocked" and turn.unconfirmed_tool_calls == ["external-call"]


async def test_cancelled_input_script_cannot_be_replayed_from_checkpoint(tmp_path: Path) -> None:
    from ecs_agent.components import AgentTurnComponent, UserPromptConfigComponent
    from ecs_agent.prompts import TriggerSpec
    from ecs_agent.serialization import WorldSerializer

    entered = asyncio.Event()
    effects = []

    async def script(world, entity, text):
        effects.append(text)
        entered.set()
        await asyncio.Future()

    agent = build_agent(AgentDefinition(FakeModel([])), runtime=RuntimeConfig(user_prompt=UserPromptConfigComponent(
        triggers=[TriggerSpec("/next", "prefix", "script", "next")], script_handlers={"next": script},
    )))
    task = asyncio.create_task(agent.run_turn("/next"))
    await asyncio.wait_for(entered.wait(), 2)
    await agent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert effects == ["/next"]
    turn = agent.world.get_component(agent.entity, AgentTurnComponent)
    assert turn.status == "blocked"
    with pytest.raises(ValueError, match="unconfirmed"):
        agent.abandon_turn(reason="do not replay")
    path = tmp_path / "interrupted.json"
    WorldSerializer.save(agent.world, path)
    restored = WorldSerializer.load(path, {"fake": FakeModel([])}, {})
    assert restored.get_component(agent.entity, AgentTurnComponent).status == "blocked"


@pytest.mark.parametrize("filename", ["agent_dsl_json.py", "agent_dsl_markdown.py", "subagent_delegation_basic.py", "prompt_normalization_demo.py", "agent_session.py"])
def test_authoring_examples_run_offline(filename) -> None:
    import os

    environment = dict(os.environ)
    environment.pop("LLM_API_KEY", None)
    process = subprocess.run([sys.executable, "examples/" + filename], env=environment, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "status: completed" in process.stdout or "Assistant: Demo reply:" in process.stdout


async def test_complete_builder_background_wait_reaches_child_result() -> None:
    from ecs_agent.components import SubagentSessionTableComponent

    class Parent(RecordingModel):
        async def complete(self, messages, tools=None, **kwargs):
            if not self.requests:
                response = CompletionResult(Message("assistant", "", tool_calls=[ToolCall("background", "subagent", {
                    "category": "worker", "prompt": "produce a background answer", "background": True,
                })]))
            elif len(self.requests) == 1:
                result = next(message for message in reversed(messages) if message.role == "tool")
                self.session_id = json.loads(result.content)["session_id"]
                response = CompletionResult(Message("assistant", "", tool_calls=[ToolCall("wait", "subagent_wait", {"session_ids": [self.session_id], "timeout": 5})]))
            elif len(self.requests) == 2:
                assert any(self.session_id in message.content and "succeeded" in message.content for message in messages), [(message.role, message.content) for message in messages]
                response = CompletionResult(Message("assistant", "", tool_calls=[ToolCall("result", "subagent_result", {"session_id": self.session_id})]))
            else:
                response = CompletionResult(Message("assistant", "background work received"))
            self._responses.append(response)
            return await super().complete(messages, tools=tools, **kwargs)

    parent = Parent([])
    child = FakeModel([CompletionResult(Message("assistant", "<subagent_background_result>\n<summary>background answer</summary>\n<full_result>background answer is ready</full_result>\n</subagent_background_result>"))])
    agent = build_agent(AgentDefinition(parent, subagents={"worker": AgentDefinition(child)}))
    async with asyncio.timeout(10):
        result = await agent.run_turn("start background work", max_steps=5)
    assert result.success, result.error
    table = agent.world.get_component(agent.entity, SubagentSessionTableComponent)
    assert len(table.sessions) == 1 and all(session.status == "succeeded" for session in table.sessions.values())
    final_results = [message.content for message in parent.requests[-1][0] if message.role == "tool"]
    assert any("background answer" in content for content in final_results), final_results


async def test_stale_checkpoint_cannot_repeat_durably_started_input_script(recording_service) -> None:
    from ecs_agent import restore_runtime
    from ecs_agent.components import AgentTurnComponent, UserPromptConfigComponent
    from ecs_agent.core import AgentHandle
    from ecs_agent.prompts import TriggerSpec
    from ecs_agent.serialization import WorldSerializer
    from ecs_agent.sessions import install_session_recording, release_session_writer

    effects = []

    async def script(world, entity, text):
        effects.append(text)
        return "accepted input"

    model = FakeModel([CompletionResult(Message("assistant", "done"))])
    agent = build_agent(AgentDefinition(model), runtime=RuntimeConfig(user_prompt=UserPromptConfigComponent(
        triggers=[TriggerSpec("/next", "prefix", "script", "next")], script_handlers={"next": script},
    )), session_service=recording_service, workspace_id="test")
    await agent.run_turn("/next", max_steps=0)
    before_execution = WorldSerializer.to_dict(agent.world)
    assert (await agent.resume()).success
    release_session_writer(agent.world, agent.entity)
    restored = WorldSerializer.from_dict(before_execution, {"fake": model}, {})
    restored.get_component(agent.entity, UserPromptConfigComponent).script_handlers["next"] = script
    restore_runtime(restored)
    loaded = AgentHandle(restored, agent.entity)
    with pytest.raises(ValueError, match="Rebind session recording"):
        await loaded.resume()
    assert effects == ["/next"]
    install_session_recording(restored, agent.entity, recording_service, workspace_id="test", agent_name="agent", restore=True, register_tools=False)
    assert not (await loaded.resume()).success
    assert effects == ["/next"]
    assert restored.get_component(agent.entity, AgentTurnComponent).status == "blocked"


@pytest.mark.parametrize("cache", ["matching", "missing", "different_turn"])
async def test_v3_checkpoint_requires_evidence_before_resuming_input_script(cache) -> None:
    from ecs_agent import restore_runtime
    from ecs_agent.components import AgentTurnComponent, UserPromptConfigComponent
    from ecs_agent.core import AgentHandle
    from ecs_agent.prompts import TriggerSpec
    from ecs_agent.serialization import WorldSerializer

    effects = []

    async def script(world, entity, text):
        effects.append(text)
        return "accepted input"

    @tool()
    def noop() -> str:
        """Provide a deterministic tool result."""
        return "ok"

    model = FakeModel([
        CompletionResult(Message("assistant", "", tool_calls=[ToolCall("noop", "noop", {})])),
        CompletionResult(Message("assistant", "done")),
    ])
    agent = build_agent(AgentDefinition(model, tools=[noop]), runtime=RuntimeConfig(user_prompt=UserPromptConfigComponent(
        triggers=[TriggerSpec("/next", "prefix", "script", "next")], script_handlers={"next": script},
    )))
    await agent.run_turn("/next", max_steps=1)
    saved = WorldSerializer.to_dict(agent.world)
    saved["schema_version"] = 3
    components = saved["entities"][str(agent.entity)]
    components["AgentTurnComponent"].pop("input_normalization_state")
    if cache == "missing":
        components.pop("RenderedUserPromptComponent")
    elif cache == "different_turn":
        components["ConversationComponent"]["messages"][0]["turn_id"] = "another-turn"
    handlers = agent.world.get_component(agent.entity, ToolRegistryComponent).handlers
    restored = WorldSerializer.from_dict(saved, {"fake": model}, handlers)
    restored.get_component(agent.entity, UserPromptConfigComponent).script_handlers["next"] = script
    restore_runtime(restored)
    loaded = AgentHandle(restored, agent.entity)
    if cache == "matching":
        assert (await loaded.resume()).success
    else:
        assert restored.get_component(agent.entity, AgentTurnComponent).status == "blocked"
        with pytest.raises(ValueError, match="unconfirmed|blocked"):
            await loaded.resume()
    assert effects == ["/next"]


async def test_prompt_commit_failure_preserves_config_cache_and_next_request(recording_service, monkeypatch) -> None:
    from ecs_agent.components import SessionRefComponent

    model = RecordingModel([CompletionResult(Message("assistant", "ok")) for _ in range(2)])
    agent = build_agent(AgentDefinition(model, prompt="accepted prompt"), session_service=recording_service)
    await agent.run_turn("first")
    before = agent.inspect_prompt()
    ref = deepcopy(agent.world.get_component(agent.entity, SessionRefComponent))
    append = recording_service.store.append

    def fail(draft, **kwargs):
        if draft.kind == "prompt_config_changed":
            raise OSError("injected config commit failure")
        return append(draft, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(recording_service.store, "append", fail)
        with pytest.raises(OSError, match="config commit failure"):
            agent.set_prompt("uncommitted prompt")
    assert agent.inspect_prompt() == before
    assert agent.world.get_component(agent.entity, SessionRefComponent) == ref
    assert (await agent.run_turn("second")).success
    assert any(message.role == "system" and message.content == "accepted prompt" for message in model.requests[-1][0])


async def test_multiple_dsl_origins_resolve_same_relative_skill_independently(tmp_path: Path) -> None:
    from ecs_agent.components import SkillComponent

    primary_dir, child_dir = tmp_path / "primary", tmp_path / "child"
    make_skill(primary_dir, "alpha")
    make_skill(child_dir, "beta")
    specs = {}
    for directory, name, mode in [(primary_dir, "main", "primary"), (child_dir, "worker", "subagent")]:
        path = directory / "agent.json"
        path.write_text(json.dumps({name: {"mode": mode, "model": name, "prompt": "Research.", "skills": [{"path": "skills/demo"}]}}))
        specs[name] = load_json_agents(path)[0]
    parent = RecordingModel([
        CompletionResult(Message("assistant", "", tool_calls=[ToolCall("delegate", "subagent", {"category": "worker", "prompt": "load beta"})])),
        CompletionResult(Message("assistant", "done")),
    ])
    child = RecordingModel([
        CompletionResult(Message("assistant", "", tool_calls=[ToolCall("details", "load_skill_details", {"skill_name": "beta"})])),
        CompletionResult(Message("assistant", "child done")),
    ])
    agent = build_agent_from_specs(specs, lambda name, _: parent if name == "main" else child, source_dir=tmp_path / "unused")
    assert (await agent.run_turn("delegate")).success
    assert set(agent.world.get_component(agent.entity, SkillComponent).skills) == {"alpha"}
    assert any(message.role == "tool" and "Verified evidence only" in message.content for message in child.requests[-1][0])


async def test_nested_child_profile_reuses_explicit_delegation_roles(tmp_path: Path) -> None:
    from ecs_agent.systems.subagent import SubagentSystem
    from ecs_agent.systems.subagent_wait import SubagentWaitSystem
    from ecs_agent.systems.subagent.runtime_profiles import ChildSystemSpec, default_child_system_specs, register_child_runtime_profile, reset_child_runtime_profiles

    def profile(context):
        return [*default_child_system_specs(context), ChildSystemSpec(SubagentSystem, -2), ChildSystemSpec(SubagentWaitSystem, -1)]

    profile_name = str(tmp_path / "nested")
    register_child_runtime_profile(profile_name, profile)
    try:
        parent = RecordingModel([
            CompletionResult(Message("assistant", "", tool_calls=[ToolCall("child", "subagent", {"category": "worker", "prompt": "delegate to leaf"})])),
            CompletionResult(Message("assistant", "parent done")),
        ])
        child = RecordingModel([
            CompletionResult(Message("assistant", "", tool_calls=[ToolCall("leaf", "subagent", {"category": "leaf", "prompt": "answer"})])),
            CompletionResult(Message("assistant", "child done")),
        ])
        leaf = RecordingModel([CompletionResult(Message("assistant", "leaf done"))])
        agent = build_agent(AgentDefinition(parent, subagents={"worker": AgentDefinition(
            child, runtime_profile=profile_name, subagents={"leaf": AgentDefinition(leaf)},
        )}))
        assert (await agent.run_turn("delegate")).success
        assert len(leaf.requests) == 1
        assert any(message.role == "tool" and "leaf done" in message.content for message in child.requests[-1][0])
    finally:
        reset_child_runtime_profiles()


async def test_legacy_prompt_adoption_requires_explicit_conflict_resolution() -> None:
    from ecs_agent.authoring.runtime import core_steps
    from ecs_agent.authoring.scheduling import install_steps
    from ecs_agent.components import LLMComponent, SystemPromptComponent
    from ecs_agent.core import AgentHandle

    world = World()
    entity = world.create_entity()
    model = RecordingModel([CompletionResult(Message("assistant", "done"))])
    world.add_component(entity, LLMComponent(model, system_prompt="LLM field"))
    world.add_component(entity, SystemPromptComponent("selected system field"))
    install_steps(world, core_steps())
    agent = AgentHandle(world, entity)
    before = agent.inspect_prompt()
    with pytest.raises(ValueError, match="Conflicting prompt sources"):
        agent.adopt_prompt()
    assert agent.inspect_prompt() == before
    agent.adopt_prompt(source="system")
    agent.adopt_prompt()  # Compatibility copies agree with the canonical source.
    assert (await agent.run_turn("hello")).success
    assert any(message.role == "system" and message.content == "selected system field" for message in model.requests[0][0])


@pytest.mark.parametrize("committed", [False, True])
def test_history_replace_failure_uses_durable_commit_as_authority(recording_service, monkeypatch, committed) -> None:
    from ecs_agent.components import RenderedUserPromptComponent, ResponsesAPIStateComponent, SessionRefComponent

    agent = build_agent(AgentDefinition(FakeModel([])), session_service=recording_service)
    agent.replace_history([Message("user", "original")], reason="seed")
    original = agent.history_snapshot()
    old_ref = deepcopy(agent.world.get_component(agent.entity, SessionRefComponent))
    agent.world.add_component(agent.entity, RenderedUserPromptComponent("old rendered text"))
    agent.world.add_component(agent.entity, ResponsesAPIStateComponent())
    append = recording_service.store.append

    def fail(draft, **kwargs):
        if draft.kind == "history_replaced":
            if committed:
                append(draft, **kwargs)
            raise OSError("history commit acknowledgement unavailable")
        return append(draft, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(recording_service.store, "append", fail)
        if committed:
            agent.replace_history([Message("user", "replacement")], reason="edit")
        else:
            with pytest.raises(OSError, match="acknowledgement unavailable"):
                agent.replace_history([Message("user", "replacement")], reason="edit")
    snapshot = agent.history_snapshot()
    assert snapshot.messages == ((Message("user", "replacement"),) if committed else original.messages)
    assert recording_service.replay_history(old_ref.session_id) == list(snapshot.messages)
    assert agent.world.has_component(agent.entity, ResponsesAPIStateComponent) is not committed
    assert agent.world.has_component(agent.entity, RenderedUserPromptComponent) is not committed
    if not committed:
        assert snapshot == original
        assert agent.world.get_component(agent.entity, SessionRefComponent) == old_ref
