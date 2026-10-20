"""Validate all authoring definitions before activating runtime resources."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from string import Template

from ecs_agent.authoring.definitions import AgentDefinition, ModelRef, SkillRef
from ecs_agent.core.world import World
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.skills.skill import Skill
from ecs_agent.tools.definitions import ToolDefinition
from ecs_agent.tools.registry import register_tools


def normalize_prompt(prompt: str | SystemPromptConfigSpec) -> SystemPromptConfigSpec:
    """Snapshot a template file and copy configuration without invoking callbacks."""
    config = SystemPromptConfigSpec(PromptTemplateSource(inline=prompt)) if isinstance(prompt, str) else prompt
    source = config.template_source
    source_path = config.source_path
    if source.file_path is not None:
        try:
            source_path = str(Path(source.file_path).resolve())
            source = PromptTemplateSource(inline=Path(source.file_path).read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"Cannot load prompt template '{source.file_path}': {exc}") from exc
    values = {placeholder.name: placeholder.value for placeholder in config.placeholders}
    if len(values) != len(config.placeholders):
        raise ValueError("Duplicate user placeholder name")
    templates = [source.inline or "", *(value for value in values.values() if isinstance(value, str))]
    for text in templates:
        template = Template(text)
        if not template.is_valid():
            raise ValueError("Invalid prompt template substitution")
        unknown = [name for name in template.get_identifiers() if not name.startswith("_") and name not in values]
        if unknown:
            raise ValueError(f"Unknown prompt placeholders: {unknown}")
    return replace(config, template_source=source, placeholders=list(config.placeholders), source_path=source_path)


def resolve_skill(reference: SkillRef, *, context: str) -> Skill:
    """Resolve a local skill while preserving source containment and diagnostics."""
    path = Path(reference.path)
    if not path.is_absolute():
        if reference.source_dir is None:
            raise ValueError(f"{context}: a source_dir is required for relative resources")
        root = reference.source_dir.resolve()
        path = (root / path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"{context}: path escapes its source directory")
    skill_file = (path / "SKILL.md").resolve()
    if reference.source_dir is not None and not skill_file.is_relative_to(reference.source_dir.resolve()):
        raise ValueError(f"{context}: path escapes its source directory")
    if not skill_file.is_file():
        raise ValueError(f"{context}: skill file does not exist: {skill_file}")
    skill = Skill(skill_path=skill_file)
    if not skill.valid:
        raise ValueError(f"{context}: invalid skill metadata")
    return skill


def normalize_definition(definition: AgentDefinition, *, ancestors: frozenset[int] = frozenset()) -> AgentDefinition:
    """Validate a complete definition tree, without installing skills or models."""
    if id(definition) in ancestors:
        raise ValueError(f"Recursive agent definition: {definition.name}")
    if not definition.name:
        raise ValueError("An agent definition requires a nonempty name")
    if "_ecs_authoring" in definition.metadata:
        raise ValueError("metadata._ecs_authoring is reserved for runtime restoration")
    if isinstance(definition.model, ModelRef) and not definition.model.name:
        raise ValueError(f"{definition.name}: model identifier must not be empty")
    if definition.max_ticks is not None and (isinstance(definition.max_ticks, bool) or definition.max_ticks < 0):
        raise ValueError(f"{definition.name}: max_ticks must be non-negative")
    if definition.permissions is not None and any(not isinstance(k, str) or not isinstance(v, bool) for k, v in definition.permissions.items()):
        raise ValueError(f"{definition.name}: permissions must map tool names to booleans")
    prompt = normalize_prompt(definition.prompt)
    for trigger in definition.triggers:
        if trigger.action not in ("replace", "inject"):
            raise ValueError(f"{definition.name}.triggers: script triggers require an explicitly installed script handler")
        if trigger.match_mode not in ("keyword", "prefix", "contains"):
            raise ValueError(f"{definition.name}.triggers: invalid match mode")
    probe = World()
    entity = probe.create_entity()
    registry = register_tools(probe, entity, definition.tools)
    tools = [ToolDefinition(schema, registry.handlers[name]) for name, schema in registry.tools.items()]
    skills: list[SkillRef | ScriptSkill] = []
    names: set[str] = set()
    for index, reference in enumerate(definition.skills):
        context = f"{definition.name}.skills[{index}].path ({definition.source})"
        skill = resolve_skill(reference, context=context) if isinstance(reference, SkillRef) else reference
        if not isinstance(skill, ScriptSkill):
            raise ValueError(f"{context}: expected a ScriptSkill or SkillRef")
        if skill.name in names:
            raise ValueError(f"{context}: duplicate skill name '{skill.name}'")
        names.add(skill.name)
        register_tools(probe, entity, skill.tools())
        skills.append(skill)
    reserved = {"load_skill_details"} if skills else set()
    if definition.subagents:
        reserved.update({"subagent", "subagent_status", "subagent_result", "subagent_cancel", "subagent_wait", "subagent_resume"})
    collisions = reserved.intersection(registry.tools)
    if collisions:
        raise ValueError(f"{definition.name}: tool names reserved by enabled capabilities: {sorted(collisions)}")
    children = {}
    for name, child in definition.subagents.items():
        if not name:
            raise ValueError(f"{definition.name}: empty subagent name")
        children[name] = replace(normalize_definition(child, ancestors=ancestors | {id(definition)}), name=name)
    return replace(
        definition, prompt=prompt, tools=list(tools), skills=skills,
        subagents=children, permissions=deepcopy(definition.permissions),
        triggers=list(definition.triggers), metadata=deepcopy(definition.metadata),
        inheritance_policy=deepcopy(definition.inheritance_policy),
        inherited_skills=list(definition.inherited_skills),
    )
