# Agent authoring and controlled context changes

Use `build_agent()` to create a complete runtime and an `AgentHandle`. Python
definitions and DSL files share validation and installation. The handle continues
to use the existing World, components, Runner, session recorder, and RunResult.

```python
from ecs_agent import AgentDefinition, build_agent, tool

@tool()
def total(values: list[int]) -> str:
    """Sum a list of integers."""
    return str(sum(values))

agent = build_agent(AgentDefinition(
    model=model,
    prompt="Use the available tools when needed.",
    tools=[total],
))
result = await agent.run_turn("Add 2 and 3", max_steps=20)
if result.success:
    print(result.final_message.content)
```

All construction, inspection, and restore names below are exported from
`ecs_agent.authoring` and `ecs_agent`. Providers and session services remain
host-owned. Close them when the application is finished; building an agent does
not transfer their ownership.

## Definitions and resources

| Input | Meaning |
| --- | --- |
| `AgentDefinition.model` | A provider instance, or a `ModelRef` resolved by `model_factory` after validation |
| `prompt` | A string or `SystemPromptConfigSpec` |
| `tools` | Decorated functions or `ToolDefinition` objects; registration does not grant permission |
| `permissions` | Optional tool-name-to-boolean policy, with the same semantics as DSL `tools` |
| `skills` | `SkillRef` local directories or ScriptSkill runtime objects |
| `subagents` | Named child AgentDefinition values; child Worlds install their own local assets |
| `triggers` | Declarative input replacement or injection rules |
| `inherited_skills`, `inheritance_policy` | Explicit child inheritance, separate from local skill installation |
| `runtime_profile` | A registered child runtime profile; required systems and ordering are validated |
| `name`, `description`, `metadata` | Identity and descriptive information; `_ecs_authoring` metadata is reserved |

```python
from pathlib import Path
from ecs_agent import AgentDefinition, SkillRef, build_agent

agent = build_agent(AgentDefinition(
    model=model,
    prompt="Available skills:\n${_installed_skills}",
    skills=[SkillRef("skills/research", source_dir=Path("/project"))],
    subagents={
        "researcher": AgentDefinition(
            model=research_model,
            prompt="Research the delegated task.",
            skills=[SkillRef("skills/research", source_dir=Path("/project"))],
        ),
    },
))
```

Relative skill references require a source directory. DSL loaders retain each
file's origin, including after name conflict resolution. A loader origin takes
precedence over the legacy `source_dir` fallback. Missing resources, source-path
escapes, unknown user placeholders, conflicting tools, and invalid dependencies
fail during preparation. File templates are snapshotted when accepted, and
changes on disk take effect through an explicit new configuration update.

Skill and delegation installers merge tools through the common registry
validator. Shared framework tools are installed once. Repeated installation
preserves their handler identity; a user tool occupying the same name is an
error. Explicit restore rebinding is separate from ordinary registration.

## Complete DSL construction

```python
from ecs_agent import build_agent_from_specs
from ecs_agent.dsl import load_json_agents, resolve_agent_specs

specs = resolve_agent_specs(load_json_agents("agents.json"))
agent = build_agent_from_specs(specs, model_factory)
result = await agent.run_turn("Research this topic")
```

Both primary and subagent declarations support skills, placeholders and
triggers. Child triggers transform the delegated input. Child-local skill
paths use the child's defining file and do not install those skills on the
parent. DSL `tools` remains a permission mapping, not a handler loader.

`compile_agent_specs()` retains its `(EntityId, World)` partial-assembly
contract. Existing callers that add their own reasoning and execution systems
can continue to do so. New examples use `build_agent_from_specs()` and do not
register a second runtime loop.

## Runtime order and inspection

The default complete loop installs input normalization, Prompt rendering,
reasoning, tool execution and error handling. Delegation adds its system,
control tools, session table and background-wait support. Configured compaction
runs before normalization. Tool authorization remains enforced at execution.

```python
from ecs_agent import inspect_runtime

report = inspect_runtime(agent)
for step in report.systems:
    print(step.name, step.priority, step.system_type)
print(report.tools)  # tool name and installation source
print(report.component_sources)
print(report.concurrency_groups)  # priority and the roles that share it
print(report.diagnostics)  # partial-runtime guidance; failures raise at construction
```

The report is an immutable installation snapshot. Registration priorities are
the scheduling authority. Legacy system constructor `priority` arguments remain
accepted; new assembly supplies priority only in the registration plan.

`SystemStep(name, factory, after=..., before=..., priority=...)` describes a
World-level role. `RuntimeConfig.extra_steps` extends the defaults; an explicit
`RuntimeConfig.steps` supplies a whole loop and must include its required roles.
Missing roles, cycles, conflicting duplicate names and incompatible priorities
are rejected. Dependencies always cross execution groups because systems at the
same priority run concurrently. Factories must create fresh system instances
without activating external resources.

`RuntimeConfig.components` accepts additional copyable configuration components
for a fresh entity. Core state slots and duplicate component types are rejected.
`RuntimeConfig.user_prompt` accepts a UserPromptConfigComponent for context-pool
settings or explicitly bound script triggers. Choose that input configuration
or `AgentDefinition.triggers`, rather than both. Normalization precedes Prompt
rendering, so a script's phase transition is reflected in the same request.

## Prompt updates and diagnostics

`SystemPromptConfigSpec` is the canonical authoring configuration. The rendered
Prompt is derived; legacy LLM/SystemPrompt fields are compatibility views.

```python
from ecs_agent.prompts import PlaceholderSpec, PromptTemplateSource, SystemPromptConfigSpec

agent.set_prompt(SystemPromptConfigSpec(
    PromptTemplateSource(inline="You are ${role}."),
    [PlaceholderSpec("role", "a reviewer")],
))
view = agent.inspect_prompt()
print(view.config_revision, view.source)
```

Updates are allowed only between finished or explicitly abandoned turns, with
closed tool history. A successful update advances the configuration revision,
invalidates the rendered body and Responses chain, and takes effect on the next
request. If recording fails, the previous configuration remains active.
For a manually assembled handle, install SystemPromptRenderSystem before using
this operation.

`inspect_prompt()` reads the last rendered text, stable prefix, volatile suffix,
placeholder snapshot and sources, cache key, template source and revisions. It never evaluates callbacks,
executes a trigger, or consumes queued context. Immediately after an update the
new revision is visible but rendered text is absent until rendering occurs.

`adopt_prompt(source=...)` migrates existing fields. The supported source names
are `config`, `system`, and `llm`. Omitting the source is allowed when there is
only one unambiguous value; conflicting legacy values require an explicit choice.

The cache key covers template content, configuration revision, resolved user
values and built-in provider output. User callbacks are evaluated once per
render attempt and remain host-owned, including bound methods. The stable
prefix/volatile suffix split remains intact.

For persistent dynamic values, register a stable binding before saving/loading:

```python
from ecs_agent.prompts import PromptBinding, register_prompt_binding

register_prompt_binding("application.current_project", current_project)
placeholder = PlaceholderSpec("project", PromptBinding("application.current_project"))
```

Unregistered executable values are not serialized as strings. Missing bindings
produce an explicit restore error.

## History snapshots and edits

```python
snapshot = agent.history_snapshot()
agent.replace_history(
    edited_messages,
    reason="remove obsolete context",
    expected_history_id=snapshot.history_id,
    expected_revision=snapshot.revision,
)
agent.reset_session(reason="start a separate task")
```

Snapshots copy messages and nested content. Replacement checks tool call/result
identity, ordering and closure; optional history/revision guards reject stale
edits. It records a complete ordered snapshot before updating the runtime view,
then invalidates the Responses chain and message-position-dependent views.

Reset creates a new history/session identity and preserves the old transcript
and source-session relationship. The catalog, new records, writer handoff and
reset record commit together. In-memory and SQLite stores support the required
`SessionStore.transaction()` contract. A failed transaction leaves the old
binding and history visible; a lost acknowledgement is checked against the
commit record before publishing memory. `SessionService.replay_history()` is a
host recovery API for reconstructing complete recorded history snapshots.

Running, paused, failed-but-unabandoned, legacy and blocked turns cannot be
edited through these operations. Unknown tool effects cannot be discarded by
resetting context. Input normalization is recorded per managed turn: continuation
and synthetic context changes do not execute its script triggers again. An
interrupted trigger with unconfirmed completion blocks automatic replay.

Internal compaction and context assembly continue to use the coordinated history
helpers. Direct mutation of ConversationComponent.messages remains a low-level
compatibility path and does not provide the public editing guarantees.

## Checkpoint restoration

```python
from ecs_agent import AgentHandle, WorldSerializer, restore_runtime

restored = WorldSerializer.load(path, providers, tool_handlers)
restore_runtime(restored)
agent = AgentHandle(restored, entity)
result = await agent.resume(max_steps=20)
```

Checkpoint schema 4 saves the additional authoring data, Prompt revisions, child
capabilities and input-normalization state. Older supported payloads still load
under their existing lifecycle migration rules. Rendered system Prompts are
discarded when a canonical configuration is available and rebuilt from current
bindings.

`restore_runtime()` installs systems without replacing conversations or turn
state. Its saved installation manifest rejects missing or differently ordered
custom roles. Supply the original system configuration through `runtime` when
needed; restore does not apply `RuntimeConfig.components` or `user_prompt` again.

File skills rebind from their saved sources without rerunning initialization
hooks. Saved tool schemas must match. Supply custom bound ScriptSkill adapters
through `skill_bindings`; standalone tool handlers and providers are resolved
through WorldSerializer as before. Handler maps are copied and filtered to each
entity's declared schemas; the host resolver dictionary is not a live registry.
Delegation handlers are rebound to the loaded World.

Rebind script-trigger handlers and phase graphs through the existing host APIs
before driving a restored entity. Missing script handlers are rejected before
managed input is committed. Session recording still requires explicit host
rebinding and writer ownership; managed execution and editing reject a saved
session identity without those bindings. Unconfirmed tools or interrupted input scripts
remain blocked after restore.

Schema 3 checkpoints did not record input-normalization execution state. A
matching rendered input and turn identity can establish completion; an unfinished
script-bearing turn without that evidence is blocked. With session recording,
stable script-attempt receipts also prevent an older checkpoint branch from
replaying an already started script. Reconcile external effects explicitly when
completion cannot be confirmed. Derived input text only substitutes its recorded
source message, so later background notifications retain their own content.

## Examples

- [JSON DSL](../../examples/agent_dsl_json.py) and [Markdown DSL](../../examples/agent_dsl_markdown.py) run deterministic delegation without credentials.
- [Basic delegation](../../examples/subagent_delegation_basic.py) uses one complete definition for the manager and researcher.
- [Prompt normalization](../../examples/prompt_normalization_demo.py) shows rendering, context injection and diagnostic snapshots.
- [Multiple turns and checkpoint continuation](../../examples/agent_session.py) uses build_agent and restore_runtime.

The native conversation tree write path and tool hot replacement remain outside
these interfaces. Existing low-level ECS APIs are available for advanced runtimes.
