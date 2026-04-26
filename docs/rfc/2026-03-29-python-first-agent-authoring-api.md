# RFC: Python-First Agent Authoring API

Status: Draft  
Date: 2026-03-29  
Authors: OpenCode / Hephaestus  
Target Area: Public API, examples, ergonomics, DSL interop

## Summary

This RFC proposes a new **Python-first agent authoring layer** for `ecs_agent`.
The goal is to make the common path for building agents feel like writing normal
Python, while preserving the existing ECS runtime (`World`, `EntityId`,
components, systems, `Runner`) as the actual execution model.

The proposal does **not** replace ECS. Instead, it adds a thin public façade that:

1. Reduces repeated setup boilerplate.
2. Makes the recommended entry point obvious.
3. Reuses the current runtime, components, systems, tools, DSL, and skills.
4. Preserves explicit escape hatches for advanced users.

In short: **simplify authoring, keep runtime power**.

---

## Problem Statement

The current project has strong runtime flexibility, but the public authoring
experience asks users to understand too many layers too early.

Today, a typical agent setup requires the user to manually orchestrate:

- `World()`
- `create_entity()`
- multiple `add_component(...)` calls
- multiple `register_system(...)` calls with explicit priorities
- provider wiring
- tool registry wiring
- optional prompt rendering systems
- optional skill installation
- optional subagent installation

This pattern appears repeatedly across:

- `README.md`
- `examples/chat_agent.py`
- `examples/tool_agent.py`
- `examples/skill_agent.py`
- `examples/mcp_agent.py`
- `examples/subagent_delegation.py`
- `examples/prompt_normalization_demo.py`

The issue is **not** that the ECS model is wrong. The issue is that the public API
surface lacks a single clear, low-friction authoring path.

### Key Findings From Current API Review

1. **The runtime is composable, but authoring is ceremony-heavy.**
   The same bootstrap sequence is repeated in most examples.
2. **The project already contains ergonomic building blocks**, but they are scattered:
   - `@tool` and `scan_module()`
   - `AgentSpec` and `compile_agent_specs()`
   - `SystemPromptConfigSpec`
   - `SkillManager`
   - `SubagentSystem`
3. **The current DSL is useful, but not sufficient as the main Python authoring path.**
   `compile_agent_specs()` currently attaches only a subset of runtime requirements
   and still leaves conversation + systems to user code.
4. **Priority ordering leaks into every example.**
   This is correct for the ECS runtime, but should not be mandatory for common use.

---

## Goals

This RFC aims to achieve the following:

1. Provide a **single recommended entry point** for authoring agents in Python.
2. Reduce setup code for the 80% cases:
   - chat agent
   - tool-using agent
   - plan-and-execute agent
   - prompt-normalized agent
   - skill-enabled agent
   - subagent manager
3. Keep the ECS runtime model explicit and accessible.
4. Reuse existing mechanisms rather than inventing a second hidden runtime.
5. Define clear mappings from high-level sugar to:
   - components
   - systems
   - system priorities
   - DSL interop

---

## Non-Goals

This RFC does **not** propose:

1. Removing or deprecating `World`, `Runner`, or raw component/system APIs.
2. Replacing the existing JSON/Markdown DSL.
3. Hiding all runtime structure behind an opaque black box.
4. Automatically exposing every internal runtime-managed component as a first-class public API.
5. Rewriting the ECS engine around a graph builder, async workflow engine, or new scheduler.

---

## External Framework Research

This section summarizes the most relevant external API designs studied during the
research phase, with a focus on authoring ergonomics.

### 1. PydanticAI

#### Representative API shape

```python
from pydantic_ai import Agent

agent = Agent(
    "openai:gpt-5.2",
    instructions="Be concise.",
)

result = agent.run_sync("Where does 'hello world' come from?")
print(result.output)
```

Tool registration is also local to the agent object:

```python
@agent.tool
async def customer_balance(ctx: RunContext[Deps], include_pending: bool) -> float:
    ...
```

#### Strengths

- Extremely clear main entry point: `Agent(...)`
- The happy path is obvious from the first example
- Tool registration is colocated with the agent definition
- `run()` / `run_sync()` are intuitive
- Great type-driven ergonomics

#### Weaknesses

- The single object model works best when the framework itself is already centered
  around an agent object; this project is centered around ECS runtime primitives.
- Rich typed output/dependency injection is opinionated; not all of it maps cleanly
  to `ecs_agent`.

#### What to borrow

- **One obvious entry point**
- **`Agent(...).run(...)` mental model**
- **Behavior registration near the agent definition**

---

### 2. LangChain

#### Representative API shape

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    return f"Weather in {location}"

agent = create_agent(model, tools=[get_weather])
```

#### Strengths

- Excellent function-first ergonomics
- `@tool` on ordinary Python functions is easy to teach
- Schema inference from function signatures is low friction

#### Weaknesses

- The global architecture can feel fragmented: tools, executors, prompts,
  chains, agents, graphs
- Easy for simple examples, but conceptual surface grows quickly

#### What to borrow

- **Decorator-first tool registration**
- **Passing regular Python callables as tools**
- **Automatic schema inference from type hints / docstrings**

---

### 3. smolagents

#### Representative API shape

```python
from smolagents import tool

@tool
def get_weather(location: str, celsius: bool | None = False) -> str:
    ...
```

#### Strengths

- Very low-friction tool authoring
- Plain Python functions remain the default
- Small surface area for simple tasks

#### Weaknesses

- The API is optimized for lightweight agents, not for a deeply composable ECS model
- Less suitable as a blueprint for advanced runtime composition

#### What to borrow

- **Use Python functions as the primary tool authoring unit**
- **Keep simple tools simple**

---

### 4. Microsoft agent-framework

#### Representative API shape

```python
writer = Agent(
    client=client,
    name="WriterAgent",
    instructions="You are a creative writer.",
)

writer_tool = writer.as_tool(
    name="creative_writer",
    description="Generate creative content",
    arg_name="request",
)

coordinator = Agent(
    client=client,
    name="CoordinatorAgent",
    instructions="Delegate to the writer when appropriate.",
    tools=[writer_tool],
)
```

#### Strengths

- Agent composition is explicit and readable
- `agent.as_tool()` is a strong abstraction for hierarchical systems
- Good fit for multi-agent orchestration

#### Weaknesses

- Less lightweight than a pure function-first API
- Still assumes the framework’s primary unit is an `Agent` object, not an ECS world

#### What to borrow

- **Agent-as-tool composition**
- **Natural multi-agent composition at the authoring layer**

---

### 5. Haystack

#### Representative API shape

```python
pipe = Pipeline()
pipe.add_component("builder", prompt_builder)
pipe.add_component("llm", llm)
pipe.connect("builder.prompt", "llm.messages")
```

#### Strengths

- Explicit flow graph
- Clear for complex deterministic pipelines
- Good observability of data flow

#### Weaknesses

- Too verbose for the common chat/tool-use authoring case
- Better for explicit flow composition than for quick-start agent creation

#### What to borrow

- **Optional advanced builder / graph semantics for complex compositions**
- Not as the primary public API

---

### 6. CrewAI

#### Representative API shape

```python
@CrewBase
class ResearchCrew:
    @agent
    def researcher(self) -> Agent:
        return Agent(config=self.agents_config["researcher"], tools=[SerperDevTool()])
```

#### Strengths

- Good for convention-driven project structure
- Bridges config files and code nicely
- Useful when teams want repeatable role-based assembly

#### Weaknesses

- Decorators + external config can obscure runtime behavior
- The magic level is higher than ideal for an ECS runtime that values explicit assembly

#### What to borrow

- **Config/code bridge patterns**
- **Role/preset-based assembly**

---

### 7. Guidance

#### Representative API shape

```python
@guidance
def qa_bot(lm, query):
    lm += f"Q: {query}\nA: {gen(name='answer', stop='Q:')}"
    return lm
```

#### Strengths

- Prompt-as-program is expressive
- Good for structured prompting and extraction

#### Weaknesses

- It introduces a different mental model
- Better for prompt programming than for complete agent runtime authoring

#### What to borrow

- **Prompts should be authorable declaratively in Python**
- But prompt programs should remain optional, not the main agent entry point

---

## Research Conclusion

Across all studied frameworks, the most consistent ergonomic lessons are:

1. **There should be one obvious public entry point.**
2. **Ordinary Python functions should be the default tool authoring unit.**
3. **The 80% path should not require manual runtime wiring.**
4. **Advanced composition should be available, but opt-in.**
5. **Config/DSL should complement code authoring, not replace it.**

For `ecs_agent`, this points to a Python-first façade over the existing ECS runtime.

---

## Design Principles

The new public API should follow these principles:

### 1. ECS Remains the Runtime Truth

The new authoring API must compile into the same runtime model:

- `World`
- `EntityId`
- component attachments
- system registration
- `Runner`

No second hidden runtime should be introduced.

### 2. New Users Get a Default Path

The first example in the docs should not start with `World()`.
It should start with a higher-level agent authoring object.

### 3. Advanced Users Keep Escape Hatches

Users must be able to drop down to raw ECS at any point.

### 4. Presets Beat Heuristics

The framework should not guess which systems to register.
Instead, it should expose explicit presets such as:

- `chat`
- `tool_use`
- `plan_execute`
- `subagent_manager`
- `prompted`

### 5. Runtime-Managed Components Stay Internal By Default

Components such as `PendingToolCallsComponent`, `ToolResultsComponent`,
`RenderedSystemPromptComponent`, and `ChildStubComponent` should remain generated
or managed by systems. They should not be the first thing the new authoring API asks
users to reason about.

---

## Options Considered

### Option A: Helper Functions Only

Examples:

- `create_chat_agent(...)`
- `create_tool_agent(...)`
- `create_subagent_manager(...)`

#### Pros

- Fastest to implement
- Very low risk
- Easy migration for examples

#### Cons

- Sprawls quickly as combinations grow
- Hard to compose features orthogonally
- Does not provide a single, scalable public API shape

#### Decision

Useful as thin convenience wrappers, but **not sufficient as the main design**.

---

### Option B: Expand `AgentSpec` / DSL Into the Main Runtime API

#### Pros

- Reuses existing config structures
- Good for file-driven definitions

#### Cons

- `AgentSpec` is currently file/config oriented, not runtime-object oriented
- It does not naturally model live providers, callables, installed skills, or tool functions
- Risks turning a strict serializable DSL into a kitchen-sink runtime object

#### Decision

Keep DSL important, but **do not make it the primary runtime authoring API**.

---

### Option C: Python-First `Agent` Façade Backed By `AgentBuilder`

#### Pros

- Clear single entry point for users
- Scales from simple to advanced
- Composes well with tools, skills, prompts, subagents, and presets
- Can return `world` and `entity_id` explicitly, preserving ECS transparency

#### Cons

- Requires careful boundary design
- Needs explicit mapping rules for every supported feature area

#### Decision

**Recommended.**

This RFC adopts **Option C**.

---

## Proposed Public API

This RFC proposes a new module:

```python
from ecs_agent.agent import Agent, AgentBuilder, BuiltAgent, AgentPreset
```

### Public Objects

#### `Agent`

The primary user-facing entry point.

Responsibilities:

- Provide the shortest path for common agents
- Offer `.run()` / `.run_sync()` style ergonomics
- Allow `.build()` to reveal the compiled ECS world
- Expose a `builder()` path for advanced configuration

#### `AgentBuilder`

The explicit composition surface behind `Agent`.

Responsibilities:

- Configure components and systems intentionally
- Apply presets
- Accept custom components / systems / tool registries / features
- Compile to a `BuiltAgent`

#### `BuiltAgent`

The compiled handle returned by `build()`.

Responsibilities:

- Hold `world`, `entity_id`, `runner`
- Provide execution helpers
- Provide ECS escape hatches directly

---

## API Sketch

### Minimal Chat Agent

```python
from ecs_agent.agent import Agent

agent = Agent(
    provider=provider,
    model="qwen3.5-flash",
    instructions="You are a helpful assistant.",
)

result = await agent.run("Hello, how are you?")
print(result.text)
```

### Tool-Using Agent

```python
from ecs_agent.agent import Agent
from ecs_agent.tools import tool

@tool
async def add(a: int, b: int) -> int:
    return a + b

agent = Agent(
    provider=provider,
    model="qwen3.5-plus",
    instructions="You are a calculator assistant.",
    tools=[add],
    preset="tool_use",
)

result = await agent.run("What is 2 + 3?")
```

### Advanced Builder Path

```python
built = (
    Agent.builder(provider=provider, model="qwen3.5-plus")
    .instructions("You are a research manager.")
    .messages("Investigate the topic and summarize the findings.")
    .tools(search_web, fetch_doc)
    .skills(my_skill)
    .preset("subagent_manager")
    .build()
)

await built.run(max_ticks=10)
world = built.world
entity_id = built.entity_id
```

---

## Detailed API Proposal

### `Agent`

```python
class Agent:
    def __init__(
        self,
        *,
        model_impl: LLMModel | None = None,
        model: str | None = None,
        instructions: str | None = None,
        prompt: str | None = None,
        messages: str | Message | list[Message] | None = None,
        tools: Sequence[ToolLike] | None = None,
        skills: Sequence[SkillLike] | None = None,
        subagents: Mapping[str, SubagentConfig] | None = None,
        preset: AgentPreset | str = "chat",
        streaming: bool = False,
        max_messages: int | None = None,
        approval_policy: ApprovalPolicy | None = None,
        retry_config: RetryConfig | None = None,
        sandbox: SandboxOptions | None = None,
        prompt_config: SystemPromptConfigSpec | None = None,
        user_prompt_config: UserPromptConfigComponent | None = None,
        permissions: PermissionComponent | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    @classmethod
    def builder(cls, *, model_impl: LLMModel | None = None, model: str | None = None) -> AgentBuilder: ...

    @classmethod
    def from_spec(cls, spec: AgentSpec, *, model_factory: ModelFactory) -> Agent: ...

    @classmethod
    def from_specs(cls, specs: dict[str, AgentSpec], *, model_factory: ModelFactory) -> BuiltAgent: ...

    def build(self) -> BuiltAgent: ...
    async def run(self, prompt: str | Message | None = None, *, max_ticks: int | None = None) -> AgentRunResult: ...
    def run_sync(self, prompt: str | Message | None = None, *, max_ticks: int | None = None) -> AgentRunResult: ...
```

### Design Intent

- `instructions` is the preferred user-facing term because it matches how users think.
- `prompt` is accepted as an alias for migration friendliness.
- `messages` allows one-message quick starts without exposing `ConversationComponent` immediately.
- `preset` is explicit because system ordering is important.
- `build()` reveals the ECS runtime rather than hiding it.
- `from_spec()` and `from_specs()` create a bridge to the existing DSL.

---

### `AgentBuilder`

```python
class AgentBuilder:
    def model_impl(self, model_impl: LLMModel, *, model: str | None = None) -> AgentBuilder: ...
    def model(self, model: str) -> AgentBuilder: ...
    def instructions(self, text: str) -> AgentBuilder: ...
    def system_prompt(self, text: str) -> AgentBuilder: ...  # legacy alias
    def prompt_config(self, spec: SystemPromptConfigSpec) -> AgentBuilder: ...
    def user_prompt_config(self, config: UserPromptConfigComponent) -> AgentBuilder: ...
    def messages(self, *messages: str | Message) -> AgentBuilder: ...
    def tools(self, *tools: ToolLike) -> AgentBuilder: ...
    def tool_module(self, module: ModuleType) -> AgentBuilder: ...
    def skills(self, *skills: SkillLike) -> AgentBuilder: ...
    def subagents(self, subagents: Mapping[str, SubagentConfig]) -> AgentBuilder: ...
    def preset(self, preset: AgentPreset | str) -> AgentBuilder: ...
    def streaming(self, enabled: bool = True) -> AgentBuilder: ...
    def memory(self, *, max_messages: int | None = None) -> AgentBuilder: ...
    def approval(self, *, policy: ApprovalPolicy, timeout: float | None = None) -> AgentBuilder: ...
    def sandbox(self, *, timeout: float | None = None, max_output_size: int | None = None) -> AgentBuilder: ...
    def permissions(self, allowed: Sequence[str] = (), denied: Sequence[str] = ()) -> AgentBuilder: ...
    def checkpointing(self, enabled: bool = True) -> AgentBuilder: ...
    def compaction(self, **kwargs: Any) -> AgentBuilder: ...
    def message_bus(self, **kwargs: Any) -> AgentBuilder: ...
    def rag(self, **kwargs: Any) -> AgentBuilder: ...
    def user_input(self, enabled: bool = True) -> AgentBuilder: ...
    def task_runtime(self, enabled: bool = True, **kwargs: Any) -> AgentBuilder: ...
    def scratchbook(self, enabled: bool = True, **kwargs: Any) -> AgentBuilder: ...
    def workspace(self, **kwargs: Any) -> AgentBuilder: ...
    def component(self, component: object, *, replace: bool = False) -> AgentBuilder: ...
    def system(self, system: object, *, priority: int) -> AgentBuilder: ...
    def build(self) -> BuiltAgent: ...
```

### Design Intent

- `instructions(...)` and `prompt_config(...)` deliberately coexist:
  - `instructions(...)` is the easy path
  - `prompt_config(...)` is the declarative advanced prompt path
- `tool_module(...)` explicitly embraces existing `scan_module()` behavior
- `component(...)` and `system(...)` are the primary ECS escape hatches
- `checkpointing()`, `compaction()`, `rag()`, `message_bus()`, `task_runtime()`,
  and `workspace()` acknowledge advanced features without forcing them into the minimal path

---

### `BuiltAgent`

```python
@dataclass(slots=True)
class BuiltAgent:
    world: World
    entity_id: EntityId
    runner: Runner

    async def run(self, prompt: str | Message | None = None, *, max_ticks: int | None = None) -> AgentRunResult: ...
    def run_sync(self, prompt: str | Message | None = None, *, max_ticks: int | None = None) -> AgentRunResult: ...
    def conversation(self) -> ConversationComponent | None: ...
```

### Design Intent

- `BuiltAgent` is intentionally small
- It should not become a second giant runtime surface
- Its main job is to make the compiled ECS world usable without hiding it

---

## Presets

Presets are explicit named stacks that register known-good component + system bundles.

### Proposed Presets

| Preset | Purpose | Systems Installed |
|---|---|---|
| `chat` | Basic assistant loop | `ReasoningSystem`, `MemorySystem`, `ErrorHandlingSystem` |
| `tool_use` | Tool-calling assistant | `ReasoningSystem`, `ToolExecutionSystem`, `MemorySystem`, `ErrorHandlingSystem` |
| `plan_execute` | Plan + act loop | `PlanningSystem`, `ToolExecutionSystem`, `ReplanningSystem`, `MemorySystem`, `ErrorHandlingSystem` |
| `prompted` | Prompt rendering + normalization | `SystemPromptRenderSystem`, `UserPromptNormalizationSystem`, `PromptContextCollectorSystem`, plus `chat` or `tool_use` stack |
| `approval` | Tool approval flow | `ToolApprovalSystem`, `ToolExecutionSystem`, plus reasoning stack |
| `subagent_manager` | Delegation-oriented manager agent | `SystemPromptRenderSystem`, `UserPromptNormalizationSystem`, `SubagentSystem`, `ReasoningSystem`, `ToolExecutionSystem`, `MemorySystem`, `ErrorHandlingSystem` |
| `message_bus` | Collaboration-heavy agent | `MessageBusSystem`, plus chosen reasoning stack |
| `rag` | Retrieval-augmented agent | `RAGSystem`, plus chosen reasoning stack |
| `durable` | Checkpoint / compaction capable stack | `CheckpointSystem`, `CompactionSystem`, plus chosen reasoning stack |

### Preset Design Rules

1. Presets must be explicit and documented.
2. Presets must publish their installed systems and priorities.
3. Presets may compose with one another, but composition rules must be deterministic.
4. Users may override by calling `.system(...)` explicitly.

---

## Mapping to Existing ECS Runtime

This section defines how the proposed API maps to existing runtime primitives.

### Input-Level Mapping

| High-Level API | Underlying ECS Mapping | Reason |
|---|---|---|
| `instructions(text)` | `SystemPromptComponent` for simple path, or `SystemPromptConfigSpec` for prompted presets | Keep simple prompts easy while nudging advanced users to declarative prompt config |
| `prompt_config(spec)` | `SystemPromptConfigSpec` | Reuse the newer prompt system instead of growing legacy prompt assembly |
| `messages(...)` | `ConversationComponent` | Most natural mapping |
| `tools(fn1, fn2)` | `ToolRegistryComponent` built from `@tool` metadata or adapted function schema | Reuse existing tool discovery + execution runtime |
| `tool_module(module)` | `scan_module(module)` -> `ToolRegistryComponent` | Reuse current discovery model |
| `skills(...)` | `SkillManager.install(...)`, `SkillComponent`, tool + prompt installation | Reuse existing skill lifecycle |
| `subagents({...})` | `SubagentRegistryComponent`, optional `SubagentSessionTableComponent` | Reuse current subagent runtime |
| `streaming(True)` | `StreamingComponent(enabled=True)` | Preserve current streaming behavior |
| `approval(policy=...)` | `ToolApprovalComponent` + `ToolApprovalSystem` | Approval is a separate concern and should remain explicit |
| `sandbox(...)` | `SandboxConfigComponent` | Direct mapping |
| `permissions(...)` | `PermissionComponent` | Preserve existing allow/deny model |
| `memory(max_messages=...)` | `ConversationComponent.max_messages` + `MemorySystem` | Avoid introducing a second memory abstraction |
| `checkpointing()` | `CheckpointComponent` + `CheckpointSystem` | Reuse durable snapshot flow |
| `compaction(...)` | `CompactionConfigComponent` + `CompactionSystem` | Reuse existing archival / compaction flow |
| `message_bus(...)` | `MessageBusConfigComponent`, `MessageBusSubscriptionComponent`, `MessageBusConversationComponent`, `MessageBusSystem` | Preserve collaboration model |
| `rag(...)` | `RAGTriggerComponent`, `EmbeddingComponent`, `VectorStoreComponent`, `RAGSystem` | Reuse retrieval runtime |
| `user_input()` | `UserInputComponent`, `UserInputSystem` | Preserve async human input flow |
| `task_runtime(...)` | `TaskComponent` + task services + subagent/tool runtime integration | Reuse task orchestration system |
| `scratchbook(...)` | `ScratchbookRefComponent`, `ScratchbookIndexComponent`, service wiring | Keep persistent artifacts aligned with current scratchbook model |
| `workspace(...)` | `WorkspaceBindingComponent` | Explicit workspace binding instead of hidden global state |

---

## Component Coverage Matrix

The following table describes how each currently documented component should be treated by the new authoring layer.

### Core / Author-Facing Components

| Component | Exposure in New API | Proposed Surface | Reason |
|---|---|---|---|
| `LLMComponent` | First-class | `provider(...)`, `model(...)` | Core identity of an agent |
| `ConversationComponent` | First-class | `messages(...)` | Required for almost all agents |
| `SystemPromptComponent` | Supported, but legacy | `instructions(...)` / `system_prompt(...)` | Migration path only |
| `SystemPromptConfigSpec` | First-class advanced | `prompt_config(...)` | New preferred prompt path |
| `RenderedSystemPromptComponent` | Runtime-managed | none | Produced by system; should stay internal by default |
| `RenderedUserPromptComponent` | Runtime-managed | none | Produced by system |
| `EntityRegistryComponent` | Advanced/manual | `.component(...)` only | Not needed in common path |

### Tooling Components

| Component | Exposure in New API | Proposed Surface | Reason |
|---|---|---|---|
| `ToolRegistryComponent` | First-class | `tools(...)`, `tool_module(...)` | Central to tool authoring |
| `PendingToolCallsComponent` | Runtime-managed | none | Internal handoff between systems |
| `ToolResultsComponent` | Runtime-managed | none | Internal execution artifact |
| `ToolApprovalComponent` | First-class advanced | `approval(...)` | Important but opt-in |
| `SandboxConfigComponent` | First-class advanced | `sandbox(...)` | Important safety control |
| `PermissionComponent` | First-class advanced | `permissions(...)` | Existing policy model should stay reachable |

### Planning / Search Components

| Component | Exposure in New API | Proposed Surface | Reason |
|---|---|---|---|
| `PlanComponent` | First-class via preset | `preset("plan_execute")`, optional `.component(...)` | Common enough to support indirectly |
| `PlanSearchComponent` | Advanced/manual | `.component(...)` only | Specialized planning feature |

### Collaboration / Multi-Agent Components

| Component | Exposure in New API | Proposed Surface | Reason |
|---|---|---|---|
| `MessageBusConfigComponent` | First-class advanced | `message_bus(...)` | Valuable but not default |
| `MessageBusSubscriptionComponent` | First-class advanced | `message_bus(...)` / explicit methods later | Needed for collaboration setups |
| `MessageBusConversationComponent` | Runtime-managed / advanced | `message_bus(...)` | Usually should be configured, not hand-authored |
| `OwnerComponent` | Runtime-managed | none | Parent-child linkage detail |
| `ChildStubComponent` | Runtime-managed | none | Internal subagent/runtime marker |
| `SubagentRegistryComponent` | First-class advanced | `subagents(...)` | Important feature area |
| `SubagentSessionTableComponent` | First-class advanced | enabled by `subagent_manager` preset or `subagents(..., background=True)` style API later | Needed for background control tools |

### Prompt / Context Components

| Component | Exposure in New API | Proposed Surface | Reason |
|---|---|---|---|
| `UserPromptConfigComponent` | First-class advanced | `user_prompt_config(...)` | Core to normalization pipeline |
| `PromptContextQueueComponent` | Advanced/manual | `.component(...)`, possibly future `.context_pool(...)` | Useful, but not minimal-path API |
| `PromptContextReservationComponent` | Runtime-managed | none | Internal reservation mechanism |

### Retrieval / Memory / Runtime Components

| Component | Exposure in New API | Proposed Surface | Reason |
|---|---|---|---|
| `KVStoreComponent` | Advanced/manual | `.component(...)` | Generic low-level store |
| `RAGTriggerComponent` | First-class advanced | `rag(...)` | Important for retrieval feature |
| `EmbeddingComponent` | First-class advanced | `rag(...)` | Needed to make retrieval ergonomic |
| `VectorStoreComponent` | First-class advanced | `rag(...)` | Same |
| `StreamingComponent` | First-class | `streaming(...)` | Small and useful switch |
| `CheckpointComponent` | First-class advanced | `checkpointing(...)` | Durable runtime feature |
| `CompactionConfigComponent` | First-class advanced | `compaction(...)` | Important but not always on |
| `ConversationArchiveComponent` | Runtime-managed / advanced | `compaction(...)` | Archival detail |
| `RunnerStateComponent` | Runtime-managed | none | Internal lifecycle state |
| `UserInputComponent` | First-class advanced | `user_input(...)` | Useful for HITL flows |
| `ConversationTreeComponent` | Advanced/manual | `.component(...)` | Powerful, but not main-path sugar |
| `ResponsesAPIStateComponent` | Advanced/manual | `.component(...)` or future feature wrapper | Responses API is important but separate enough to remain advanced initially |

### Error / Lifecycle Components

| Component | Exposure in New API | Proposed Surface | Reason |
|---|---|---|---|
| `ErrorComponent` | Runtime-managed | none | System-generated |
| `InterruptionComponent` | Advanced/manual | `.component(...)` | Specialized control flow |
| `TerminalComponent` | Runtime-managed | none | Runtime terminal marker |

### Task / Scratchbook / Workspace Components

| Component | Exposure in New API | Proposed Surface | Reason |
|---|---|---|---|
| `TaskComponent` | First-class advanced | `task_runtime(...)` or future task builder | Important orchestration feature |
| `ScratchbookRefComponent` | First-class advanced | `scratchbook(...)` or `.component(...)` | Persistent artifact addressing |
| `ScratchbookIndexComponent` | Runtime-managed / advanced | `scratchbook(...)` | Usually derived or maintained by services |
| `WorkspaceBindingComponent` | First-class advanced | `workspace(...)` | Useful for filesystem / repo binding |

---

## System Coverage Matrix

| System | Exposure in New API | Typical Path |
|---|---|---|
| `SystemPromptRenderSystem` | preset / advanced | `prompted`, `subagent_manager`, or explicit `.system(...)` |
| `UserPromptNormalizationSystem` | preset / advanced | `prompted`, or explicit `.system(...)` |
| `ReasoningSystem` | default | `chat`, `tool_use`, `approval`, `subagent_manager` |
| `MemorySystem` | default | `chat`, `tool_use`, `plan_execute` |
| `PlanningSystem` | preset | `plan_execute` |
| `ToolExecutionSystem` | preset | `tool_use`, `plan_execute`, `approval`, `subagent_manager` |
| `MessageBusSystem` | advanced feature | `message_bus(...)` |
| `ErrorHandlingSystem` | default | almost always installed |
| `ReplanningSystem` | preset | `plan_execute` |
| `ToolApprovalSystem` | preset / advanced | `approval(...)` |
| `TreeSearchSystem` | advanced feature | future planning/search preset |
| `RAGSystem` | advanced feature | `rag(...)` |
| `CheckpointSystem` | advanced feature | `checkpointing(...)` |
| `CompactionSystem` | advanced feature | `compaction(...)` |
| `UserInputSystem` | advanced feature | `user_input(...)` |
| `SubagentSystem` | preset / advanced | `subagent_manager`, `subagents(...)` |
| `PromptContextCollectorSystem` | preset / advanced | `prompted` |

---

## Why Each Major API Surface Exists

### `Agent(...)`

**Intent:** provide the obvious entry point that current docs lack.

**Why:** users should be able to start with one object instead of mentally assembling a world before they understand the framework.

### `AgentBuilder`

**Intent:** expose a scalable composition surface without forcing a giant constructor.

**Why:** helper functions alone will not scale to the number of combinations supported by this framework.

### `preset(...)`

**Intent:** encode system registration order explicitly.

**Why:** priority order matters, and examples currently leak that burden to every user.

### `tools(...)`

**Intent:** make tool registration feel like normal Python.

**Why:** the project already has `@tool` and `scan_module()`; the public API should lean into them instead of forcing manual `ToolSchema` assembly in common cases.

### `prompt_config(...)`

**Intent:** promote the newer prompt system instead of expanding legacy prompt strings forever.

**Why:** `SystemPromptConfigSpec` + `SystemPromptRenderSystem` is already the better architecture for prompt assembly.

### `skills(...)`

**Intent:** let users install capabilities by concept instead of wiring prompts + tools manually.

**Why:** skills are already a core feature and should be represented as such in the authoring API.

### `subagents(...)`

**Intent:** expose delegation as a first-class authoring concern.

**Why:** subagents are already a documented major feature, and current setup is too plumbing-heavy for common use.

### `component(...)` and `system(...)`

**Intent:** preserve raw ECS escape hatches.

**Why:** advanced users must never be trapped inside a closed sugar layer.

---

## DSL Interoperability

The proposal deliberately keeps the DSL as a parallel, complementary path.

### Proposed Mapping

| DSL Concept | New API Interop |
|---|---|
| `AgentSpec` | `Agent.from_spec(...)` |
| `dict[str, AgentSpec]` | `Agent.from_specs(...)` |
| `compile_agent_specs(...)` | Used internally by `from_specs(...)` where appropriate |
| Markdown / JSON files | Still loaded with existing DSL loaders |

### Important Boundary

`AgentSpec` should remain:

- serializable
- strict
- file/config friendly

It should **not** become a runtime object that accepts:

- live callables
- installed tool functions
- active provider instances
- live skills with behavior

Those belong in the new Python authoring layer.

---

## Example Migrations

### Current `chat_agent.py` style

```python
world = World()
agent_id = world.create_entity()
world.add_component(agent_id, LLMComponent(...))
world.add_component(agent_id, ConversationComponent(...))
world.register_system(ReasoningSystem(priority=0), priority=0)
world.register_system(MemorySystem(), priority=10)
world.register_system(ErrorHandlingSystem(priority=99), priority=99)
runner = Runner()
await runner.run(world, max_ticks=3)
```

### Proposed style

```python
agent = Agent(
    provider=provider,
    model=model,
    instructions="You are a helpful assistant.",
)
await agent.run("Hello, how are you?", max_ticks=3)
```

### Current `tool_agent.py` style

Today the example manually constructs:

- `ToolRegistryComponent`
- `ToolSchema`
- `handlers`
- system ordering

### Proposed style

```python
@tool
async def add(a: int, b: int) -> int:
    return a + b

@tool
async def multiply(a: int, b: int) -> int:
    return a * b

agent = Agent(
    provider=provider,
    model=model,
    instructions="You are a calculator assistant.",
    tools=[add, multiply],
    preset="tool_use",
)

await agent.run("What is 2 + 3? And what is 7 * 8?")
```

---

## Failure Modes and Guardrails

### 1. Duplicate Component Attachment

If the builder is asked to attach two components of the same semantic slot,
the behavior must be explicit:

- fail fast by default
- allow `replace=True` on `.component(...)`

### 2. Conflicting Presets

Preset composition must be deterministic. The framework must document:

- which systems each preset installs
- which priorities are used
- how duplicates are handled

### 3. Tool Registration Ambiguity

When a callable is not decorated with `@tool`, the builder should either:

1. adapt it with safe defaults, or
2. fail with a clear error explaining how to decorate it

The decision should be explicit and documented.

### 4. Legacy vs New Prompt Paths

If both `instructions(...)` and `prompt_config(...)` are provided, the API should fail fast unless explicitly told how to merge.

### 5. Existing Entity Targets

If future versions allow `build(into_entity=...)`, collision policy must be explicit.

---

## Documentation Impact

If this RFC is accepted, the docs should evolve as follows:

1. **`getting-started.md`** should lead with `Agent(...)`, not `World()`.
2. **`examples/`** should include paired examples:
   - sugar path
   - raw ECS path
3. **`api-reference.md`** should document:
   - `ecs_agent.agent.Agent`
   - `AgentBuilder`
   - `BuiltAgent`
   - presets
4. **`features/agent-dsl.md`** should explicitly describe how the DSL relates to the Python authoring layer.

---

## Recommended Rollout Plan

### Phase 1: Core Authoring Façade

Deliver:

- `Agent`
- `AgentBuilder`
- `BuiltAgent`
- presets: `chat`, `tool_use`, `plan_execute`
- `tools(...)` support for `@tool` and `scan_module()` interop

### Phase 2: Prompt / Skills / Subagent Integration

Deliver:

- `prompt_config(...)`
- `user_prompt_config(...)`
- `skills(...)`
- `subagents(...)`
- presets: `prompted`, `subagent_manager`, `approval`

### Phase 3: Advanced Runtime Features

Deliver:

- `rag(...)`
- `message_bus(...)`
- `checkpointing(...)`
- `compaction(...)`
- `task_runtime(...)`
- `scratchbook(...)`
- `workspace(...)`

---

## Final Recommendation

Adopt a **Python-first public authoring API** centered on:

1. `Agent` as the primary public entry point
2. `AgentBuilder` as the explicit advanced composition surface
3. `BuiltAgent` as the transparent compiled ECS handle
4. explicit named presets for system stacks
5. direct mapping to existing ECS runtime primitives
6. DSL interop as a bridge, not a replacement

This design best satisfies the project’s needs because it:

- reduces authoring friction
- preserves ECS power
- promotes the newer prompt / tool / skill / subagent features coherently
- creates a clear learning ladder from beginner to expert

In one sentence:

> **Make agent authoring feel simple, while keeping the runtime honestly ECS.**
