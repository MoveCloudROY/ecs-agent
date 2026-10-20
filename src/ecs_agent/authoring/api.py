"""Complete agent construction and DSL adapters over one installation pipeline."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
import inspect
from typing import Any, cast

from ecs_agent.authoring.definitions import (
    AgentDefinition, AssemblyReport, InstalledStep, ModelRef, RuntimeConfig, SkillRef, SystemStep,
)
from ecs_agent.authoring.normalize import normalize_definition
from ecs_agent.authoring.runtime import (
    core_steps, delegation_steps, install_components, install_delegation, validate_complete_systems,
)
from ecs_agent.authoring.scheduling import install_steps, resolve_steps
from ecs_agent.components import (
    AgentTurnComponent, ChildStubComponent, CompactionConfigComponent, ConversationComponent,
    ConversationTreeComponent, EntityRegistryComponent, LLMComponent,
    PermissionComponent, SubagentRegistryComponent, SubagentSessionTableComponent,
    ToolRegistryComponent, UserPromptConfigComponent, SessionRefComponent, SkillComponent,
)
from ecs_agent.core.agent import AgentHandle
from ecs_agent.core.world import World
from ecs_agent.dsl.schema import AgentSpec, validate_agent_spec
from ecs_agent.logging import get_logger
from ecs_agent.prompts.contracts import PlaceholderSpec, PromptTemplateSource, SystemPromptConfigSpec, TriggerSpec
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.sessions.install import install_session_recording, release_session_writer
from ecs_agent.sessions.service import SessionService
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.tools.definitions import ToolDefinition
from ecs_agent.tools.permissions import compile_tool_permissions
from ecs_agent.types import EntityId, SubagentConfig

logger = get_logger(__name__)
ModelFactory = Callable[[str, str], LLMModel]


@dataclass(frozen=True, slots=True)
class AssemblyPlan:
    """Validated installation data; runtime state remains in the destination World."""

    definition: AgentDefinition
    steps: tuple[SystemStep, ...]
    components: tuple[object, ...]
    complete: bool
    user_prompt: UserPromptConfigComponent | None


def _plan(definition: AgentDefinition, runtime: RuntimeConfig, *, complete: bool) -> AssemblyPlan:
    prepared = normalize_definition(definition)
    if runtime.user_prompt is not None:
        if prepared.triggers:
            raise ValueError("Choose AgentDefinition.triggers or RuntimeConfig.user_prompt as the input configuration source")
        for trigger in runtime.user_prompt.triggers:
            if trigger.match_mode not in {"keyword", "prefix", "contains"} or trigger.action not in {"replace", "inject", "script"}:
                raise ValueError("Invalid user prompt trigger")
            if trigger.action == "script":
                handler = runtime.user_prompt.script_handlers.get(trigger.content)
                if handler is None or not callable(handler):
                    raise ValueError(f"Missing script handler for trigger '{trigger.content}'")
                inspect.signature(handler).bind(None, EntityId(0), "input")
    protected = {
        LLMComponent, ConversationComponent, ConversationTreeComponent, AgentTurnComponent,
        SystemPromptConfigSpec, UserPromptConfigComponent, ToolRegistryComponent,
        SubagentRegistryComponent, SubagentSessionTableComponent, EntityRegistryComponent,
    }
    if definition.permissions is not None:
        protected.add(PermissionComponent)
    seen = set(protected)
    for component in runtime.components:
        if not is_dataclass(component) or isinstance(component, type):
            raise ValueError("Runtime components must be dataclass instances")
        if type(component) in seen:
            raise ValueError(f"Conflicting component slot: {type(component).__name__}")
        seen.add(type(component))
    if complete:
        steps = core_steps(compaction=CompactionConfigComponent in seen)
        if prepared.subagents:
            steps += delegation_steps()
    else:
        from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
        from ecs_agent.systems.user_prompt_normalization_system import UserPromptNormalizationSystem
        steps = (
            SystemStep("render", SystemPromptRenderSystem, priority=-20),
            SystemStep("normalize", UserPromptNormalizationSystem, after=("render",), priority=-10),
        )
        if prepared.subagents:
            steps += delegation_steps(complete=False)
    steps = (steps if runtime.steps is None else runtime.steps) + runtime.extra_steps
    resolved = dict((step.name, priority) for step, priority in resolve_steps(steps))
    if complete:
        required = {"normalize", "render", "reason", "execute", "errors"}
        if prepared.subagents:
            required.update({"delegate", "wait"})
        if missing := required - resolved.keys():
            raise ValueError(f"Incomplete runtime: missing roles {sorted(missing)}")
        chain = ["normalize", "render", *( ["delegate", "wait"] if prepared.subagents else []), "reason", "execute", "errors"]
        for before, after in zip(chain, chain[1:]):
            if resolved[before] >= resolved[after]:
                raise ValueError(f"Runtime dependency requires '{before}' before '{after}'")
    _validate_children(prepared, compaction=CompactionConfigComponent in seen)
    user_prompt = replace(runtime.user_prompt, triggers=list(runtime.user_prompt.triggers), script_handlers=dict(runtime.user_prompt.script_handlers)) if runtime.user_prompt is not None else None
    return AssemblyPlan(prepared, steps, tuple(deepcopy(component) for component in runtime.components), complete, user_prompt)


def _validate_children(definition: AgentDefinition, *, compaction: bool) -> None:
    from ecs_agent.systems.subagent.runtime_profiles import ChildProfileContext, resolve_child_runtime_profile
    from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
    from ecs_agent.systems.user_prompt_normalization_system import UserPromptNormalizationSystem
    from ecs_agent.systems.reasoning import ReasoningSystem
    from ecs_agent.systems.tool_execution import ToolExecutionSystem
    from ecs_agent.systems.error_handling import ErrorHandlingSystem

    for child in definition.subagents.values():
        specs = resolve_child_runtime_profile(child.runtime_profile)(ChildProfileContext(compaction, normalize=True))
        instances = [(spec.factory(), spec.priority) for spec in specs]
        ordered = []
        for required in (UserPromptNormalizationSystem, SystemPromptRenderSystem, ReasoningSystem, ToolExecutionSystem, ErrorHandlingSystem):
            priorities = [priority for system, priority in instances if isinstance(system, required)]
            if len(priorities) != 1:
                raise ValueError(f"{child.name}: runtime profile must contain exactly one {required.__name__}")
            ordered.append(priorities[0])
        if ordered != sorted(set(ordered)):
            raise ValueError(f"{child.name}: child runtime profile has conflicting system dependencies")
        if child.subagents:
            from ecs_agent.systems.subagent import SubagentSystem
            from ecs_agent.systems.subagent_wait import SubagentWaitSystem

            delegation_priorities = []
            for role_type, default_priority in ((SubagentSystem, -2), (SubagentWaitSystem, -1)):
                priorities = [priority for system, priority in instances if isinstance(system, role_type)]
                if len(priorities) > 1:
                    raise ValueError(f"{child.name}: duplicate {role_type.__name__} in child runtime profile")
                delegation_priorities.append(priorities[0] if priorities else default_priority)
            combined = [*ordered[:2], *delegation_priorities, *ordered[2:]]
            if combined != sorted(set(combined)):
                raise ValueError(f"{child.name}: nested delegation must run after rendering and before reasoning")
        _validate_children(child, compaction=compaction)


def _bind_model(definition: AgentDefinition, factory: ModelFactory | None) -> LLMModel:
    if isinstance(definition.model, ModelRef):
        if factory is None:
            raise ValueError(f"{definition.name}: model_factory is required to resolve '{definition.model.name}'")
        prompt = cast(SystemPromptConfigSpec, definition.prompt)
        return factory(definition.model.name, prompt.template_source.inline or "")
    return definition.model


def _child_config(definition: AgentDefinition, factory: ModelFactory | None) -> SubagentConfig:
    prompt = cast(SystemPromptConfigSpec, definition.prompt)
    return SubagentConfig(
        name=definition.name, model=_bind_model(definition, factory),
        description=definition.description,
        system_prompt=prompt.template_source.inline or "", prompt_config=prompt,
        tools=definition.permissions, skills=definition.inherited_skills,
        local_skills=[cast(ScriptSkill, skill) for skill in definition.skills],
        tool_definitions=[cast(ToolDefinition, tool) for tool in definition.tools],
        triggers=definition.triggers, metadata=definition.metadata,
        subagents={name: _child_config(child, factory) for name, child in definition.subagents.items()},
        inheritance_policy=definition.inheritance_policy, max_ticks=definition.max_ticks,
        runtime_profile=definition.runtime_profile, authoring=True,
    )


def _install(plan: AssemblyPlan, factory: ModelFactory | None) -> AgentHandle:
    definition = plan.definition
    world = World(name=definition.name)
    entity = world.create_entity()
    try:
        systems = install_steps(world, plan.steps)
        if plan.complete:
            validate_complete_systems(world, delegation=bool(definition.subagents))
        model = _bind_model(definition, factory)
        children = {name: _child_config(child, factory) for name, child in definition.subagents.items()}
        for component in plan.components:
            world.add_component(entity, component)
        install_components(
            world, entity, model=model, prompt=cast(SystemPromptConfigSpec, definition.prompt),
            tools=[cast(ToolDefinition, tool) for tool in definition.tools],
            skills=[cast(ScriptSkill, skill) for skill in definition.skills],
            triggers=definition.triggers, conversation=plan.complete, user_prompt=plan.user_prompt,
        )
        permissions = compile_tool_permissions(definition.permissions)
        if permissions is not None:
            world.add_component(entity, permissions)
        install_delegation(world, entity, children)
        world.register_entity(entity, definition.name)
        world.add_component(entity, EntityRegistryComponent(entity, definition.name, metadata={
            **definition.metadata,
            "_ecs_authoring": {"systems": [asdict(step) for step in systems], "complete": plan.complete},
        }))
        registry = world.get_component(entity, ToolRegistryComponent)
        assert registry is not None
        origins = {cast(ToolDefinition, tool).schema.name: "Python tool" for tool in definition.tools}
        for skill in definition.skills:
            assert isinstance(skill, ScriptSkill)
            origins.update({name: f"skill:{skill.name}" for name in skill.tools()})
        component_names = tuple(sorted(kind.__name__ for kind, entities in world._components._components.items() if entity in entities))
        sources = {name: "framework installation" for name in component_names}
        sources.update({type(component).__name__: "RuntimeConfig.components" for component in plan.components})
        sources.update({
            "LLMComponent": f"AgentDefinition.model ({definition.source})",
            "SystemPromptConfigSpec": f"AgentDefinition.prompt ({definition.source})",
            "UserPromptConfigComponent": "RuntimeConfig.user_prompt" if plan.user_prompt is not None else f"AgentDefinition.triggers ({definition.source})",
            "ToolRegistryComponent": "merged Python, skill and framework tools",
        })
        if definition.skills:
            sources["SkillComponent"] = f"AgentDefinition.skills ({definition.source})"
        info = world.get_component(entity, EntityRegistryComponent)
        assert info is not None
        info.metadata["_ecs_authoring"]["component_sources"] = sources
        world._assembly_reports[entity] = AssemblyReport(
            definition.name,
            component_names,
            tuple((name, origins.get(name, "framework")) for name in sorted(registry.tools)),
            systems, plan.complete, tuple(sorted(sources.items())),
            () if plan.complete else ("Partial runtime: install reasoning, tool execution and error handling before driving",),
        )
        return AgentHandle(world, entity)
    except Exception as exc:
        logger.error("agent_build_failed", agent=definition.name, exception=str(exc))
        for skill in reversed(definition.skills):
            assert isinstance(skill, ScriptSkill)
            if world.skill_runtime.get_installed_skill(entity, skill.name) is not None:
                try:
                    SkillManager().uninstall(world, entity, skill.name)
                except Exception as cleanup_error:
                    logger.error("agent_build_cleanup_failed", skill=skill.name, exception=str(cleanup_error))
        release_session_writer(world, entity)
        raise


def build_agent(
    definition: AgentDefinition, *, runtime: RuntimeConfig | None = None,
    model_factory: ModelFactory | None = None, session_service: SessionService | None = None,
    workspace_id: str = "default",
) -> AgentHandle:
    """Build a complete fresh agent, borrowing host-owned providers and services.

    Args:
        definition: Agent and child definitions; all are validated before activation.
        runtime: Optional complete system selection, extra roles, and components.
        model_factory: Resolver for ModelRef inputs. Returned models remain host-owned.
        session_service: Optional host-owned durable recording service.
        workspace_id: Recording scope when a session service is supplied.
    """
    agent: AgentHandle | None = None
    plan: AssemblyPlan | None = None
    try:
        plan = _plan(definition, runtime or RuntimeConfig(), complete=True)
        agent = _install(plan, model_factory)
        if session_service is not None:
            try:
                with session_service.store.transaction():
                    install_session_recording(agent.world, agent.entity, session_service, workspace_id=workspace_id, agent_name=definition.name, register_tools=False)
            except Exception as install_error:
                logger.error("agent_session_install_failed", agent=definition.name, exception=str(install_error))
                ref = agent.world.get_component(agent.entity, SessionRefComponent)
                writer = agent.world._session_writer_tokens.get(agent.entity)
                if ref is None or writer is None:
                    raise
                try:
                    committed = session_service.store.get_session(ref.session_id)
                    if committed.head_record_id != ref.head_record_id:
                        raise RuntimeError("Session installation commit is not confirmed")
                    session_service.store.claim_writer(ref.session_id, writer)
                except Exception as probe_error:
                    logger.error("agent_session_install_unconfirmed", exception=str(probe_error))
                    raise install_error from probe_error
            report = agent.world._assembly_reports[agent.entity]
            sources = dict(report.component_sources)
            sources.update({"SessionRefComponent": "SessionService", "SessionPolicyComponent": "SessionService"})
            agent.world._assembly_reports[agent.entity] = replace(
                report,
                components=tuple(sorted(kind.__name__ for kind, entities in agent.world._components._components.items() if agent.entity in entities)),
                component_sources=tuple(sorted(sources.items())),
            )
            info = agent.world.get_component(agent.entity, EntityRegistryComponent)
            assert info is not None
            info.metadata["_ecs_authoring"]["component_sources"] = sources
        return agent
    except Exception as exc:
        logger.error("agent_authoring_failed", agent=definition.name, exception=str(exc))
        if agent is not None and plan is not None:
            for skill in reversed(plan.definition.skills):
                assert isinstance(skill, ScriptSkill)
                try:
                    SkillManager().uninstall(agent.world, agent.entity, skill.name)
                except Exception as cleanup_error:
                    logger.error("agent_build_cleanup_failed", skill=skill.name, exception=str(cleanup_error))
            release_session_writer(agent.world, agent.entity)
        raise


def inspect_runtime(agent: AgentHandle) -> AssemblyReport:
    """Return the immutable installation report for an authored agent."""
    report = agent.world._assembly_reports.get(agent.entity)
    if report is None:
        raise ValueError("This entity was not created by the authoring installer")
    return report


def restore_runtime(
    world: World, *, runtime: RuntimeConfig | None = None,
    skill_bindings: dict[str, ScriptSkill] | None = None,
) -> tuple[InstalledStep, ...]:
    """Rebind systems on a loaded World without replacing history or lifecycle state.

    Custom system factories must be supplied again via runtime. The saved
    installation manifest rejects missing or differently ordered roles.
    Providers, tool handlers, sessions and phase graphs retain their explicit
    host binding contracts; this operation does not recreate their state.
    """
    config = runtime or RuntimeConfig()
    if config.components or config.user_prompt is not None:
        raise ValueError("restore_runtime accepts system definitions only; bind resources explicitly without replacing saved components")
    if not list(world.query(LLMComponent)):
        raise ValueError("restore_runtime requires a loaded agent entity")
    world.apply_pending_system_operations()
    if world._systems._systems and not world._authoring_systems:
        raise ValueError("The World already has unmanaged systems; restore runtime before manually registering systems")
    delegation = any(registry.subagents or registry.free_subagent_config.enabled for _, (registry,) in world.query(SubagentRegistryComponent))
    steps = core_steps(compaction=bool(list(world.query(CompactionConfigComponent))))
    if delegation:
        steps += delegation_steps()
    steps = (steps if config.steps is None else config.steps) + config.extra_steps
    resolved = resolve_steps(steps)
    # Validate saved roles and priorities before calling factories. Types are
    # checked below once the fresh instances exist.
    manifests = [info.metadata["_ecs_authoring"] for _, (info,) in world.query(EntityRegistryComponent) if "_ecs_authoring" in info.metadata]
    for manifest in manifests:
        expected = sorted((item["name"], item["priority"]) for item in manifest["systems"])
        if expected != sorted((step.name, priority) for step, priority in resolved):
            raise ValueError("Runtime differs from checkpoint manifest; provide the original system configuration")
    expected_types = {item["name"]: item["system_type"] for item in manifests[0]["systems"]} if manifests else None
    installed = install_steps(world, steps, expected_types=expected_types)
    validate_complete_systems(world, delegation=delegation)
    for manifest in manifests:
        if sorted((item["name"], item["system_type"]) for item in manifest["systems"]) != sorted((item.name, item.system_type) for item in installed):
            raise ValueError("Restored system types differ from the checkpoint manifest")
    for entity, (registry,) in world.query(SubagentRegistryComponent):
        install_delegation(world, entity, registry.subagents, rebind=True)
    from ecs_agent.skills.skill import Skill

    for entity, (skills,) in world.query(SkillComponent):
        if world.has_component(entity, ChildStubComponent):
            continue
        for name, metadata in skills.skills.items():
            skill = (skill_bindings or {}).get(name)
            if skill is None:
                if metadata.skill_dir_path is None:
                    raise ValueError(f"Provide the restored runtime skill binding for '{name}'")
                file_skill = Skill(skill_path=Path(metadata.skill_dir_path) / "SKILL.md")
                if not file_skill.valid or file_skill.name != name:
                    raise ValueError(f"Invalid restored file skill '{name}'")
                skill = cast(ScriptSkill, file_skill)
            SkillManager().rebind(world, entity, skill)
    for entity, (registry,) in world.query(ToolRegistryComponent):
        if world.has_component(entity, ChildStubComponent):
            continue
        if set(registry.tools) != set(registry.handlers):
            raise ValueError(f"Missing or unexpected restored tool handlers on entity {entity}")
    for entity, (info,) in world.query(EntityRegistryComponent):
        if "_ecs_authoring" not in info.metadata:
            continue
        registry = world.get_component(entity, ToolRegistryComponent)
        components = tuple(sorted(kind.__name__ for kind, entities in world._components._components.items() if entity in entities))
        sources = info.metadata["_ecs_authoring"].get("component_sources", {})
        world._assembly_reports[entity] = AssemblyReport(
            info.name, components,
            tuple((name, "restored binding") for name in sorted(registry.tools)) if registry is not None else (), installed, True,
            tuple((name, sources.get(name, "checkpoint state")) for name in components),
        )
    return installed


def definition_from_specs(specs: dict[str, AgentSpec], *, source_dir: Path | None = None) -> AgentDefinition:
    """Translate strict DSL data and per-file origins into the shared definition."""
    primary = [name for name, spec in specs.items() if spec.mode == "primary"]
    if len(primary) != 1:
        raise ValueError(f"Expected exactly one primary agent, found {len(primary)}")
    definitions = {}
    for name, original in specs.items():
        data = {
            "mode": original.mode, "model": original.model, "prompt": original.prompt,
            "name": name, "metadata": original.metadata, "skills": original.skills,
            "placeholders": original.placeholders, "triggers": original.triggers,
        }
        if original.tools is not None:
            data["tools"] = original.tools
        spec = validate_agent_spec(data, source_name=str(original.source_path or name))
        base = original.source_path.parent if original.source_path is not None else source_dir
        prompt_text = spec.prompt
        if "{file:" in prompt_text:
            if base is None:
                raise ValueError(f"{name}.prompt: a source_dir is required for file references")
            from ecs_agent.dsl.prompt_resolver import resolve_prompt_file
            prompt_text = resolve_prompt_file(prompt_text, base)
        definitions[name] = AgentDefinition(
            model=ModelRef(spec.model), name=name,
            description=str(spec.metadata.get("description", "")),
            prompt=SystemPromptConfigSpec(PromptTemplateSource(inline=prompt_text), [PlaceholderSpec(ph["name"], ph["value"]) for ph in spec.placeholders], source_path=str(original.source_path) if original.source_path is not None else None),
            permissions=spec.tools, metadata=spec.metadata,
            skills=[SkillRef(entry["path"], base) for entry in spec.skills],
            triggers=[TriggerSpec(**cast(dict[str, Any], trigger)) for trigger in spec.triggers],
            source=str(original.source_path or "in-memory spec"),
        )
    root = definitions[primary[0]]
    root.subagents = {name: definition for name, definition in definitions.items() if name != primary[0]}
    return root


def build_agent_from_specs(
    specs: dict[str, AgentSpec], model_factory: ModelFactory, *, source_dir: Path | None = None,
    runtime: RuntimeConfig | None = None, session_service: SessionService | None = None,
    workspace_id: str = "default",
) -> AgentHandle:
    """Build a complete agent from DSL; unlike compile_agent_specs, no manual wiring is needed."""
    return build_agent(
        definition_from_specs(specs, source_dir=source_dir), model_factory=model_factory,
        runtime=runtime, session_service=session_service, workspace_id=workspace_id,
    )


def compile_partial_specs(specs: dict[str, AgentSpec], model_factory: ModelFactory, *, source_dir: Path | None = None) -> tuple[EntityId, World]:
    """Compatibility adapter retaining the original partial-world return contract."""
    plan = _plan(definition_from_specs(specs, source_dir=source_dir), RuntimeConfig(), complete=False)
    agent = _install(plan, model_factory)
    return agent.entity, agent.world
