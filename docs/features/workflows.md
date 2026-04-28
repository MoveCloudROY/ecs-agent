# Workflow DSL

The ECS-based LLM Agent framework provides a declarative Domain Specific Language (DSL) for building stateful agents. Workflows allow you to define a graph of states, prompt profiles, and transition gates that drive an agent's behavior over multiple ticks.

## Architecture Overview

Workflows are implemented as a set of ECS components and a dedicated system:

- **`WorkflowDefinitionComponent`**: Holds the compiled workflow graph.
- **`WorkflowRuntimeComponent`**: Tracks the current state and transition history.
- **`WorkflowBindingComponent`**: Binds an agent key to the workflow.
- **`WorkflowStateSystem`**: Evaluates gates and commits transitions (priority -25).
- **`WorkflowPromptPlaceholderProvider`**: Injects the active state's prompt into the system prompt via the `${_workflow_state_prompt}` placeholder.

## Quick Start

The following example defines a simple two-state workflow where an agent starts in a `DRAFT` state and moves to `REVIEW` once a `DoneMarker` component is attached to the entity.

```python
from ecs_agent.core import World
from ecs_agent.workflows import workflow, install_workflow, has
from ecs_agent.systems.workflow_state import WorkflowStateSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.prompts.contracts import SystemPromptConfigSpec, PromptTemplateSource

# 1. Define the workflow
spec = workflow(
    "my-flow",
    initial="DRAFT",
    profiles={
        "agent": {
            "draft_p": "You are in draft mode. Help the user write a plan.",
            "review_p": "You are in review mode. Critique the plan.",
        }
    },
    states={
        "DRAFT": {
            "bind": {"agent": "draft_p"},
            "go": {"REVIEW": has(DoneMarker)},
        },
        "REVIEW": {
            "bind": {"agent": "review_p"},
            "go": {},
        },
    },
)

# 2. Setup World and Entity
world = World()
eid = world.create_entity()

# 3. Install workflow and prompt config
install_workflow(world, eid, spec, agent_key="agent")
world.add_component(eid, SystemPromptConfigSpec(
    template_source=PromptTemplateSource(inline="${_workflow_state_prompt}")
))

# 4. Register systems
world.register_system(WorkflowStateSystem(priority=-25), priority=-25)
world.register_system(SystemPromptRenderSystem(priority=-20), priority=-20)
```

## Gate Primitives

Gates are expressions evaluated against the entity's components to determine if a transition should fire.

| Primitive | Description |
| :--- | :--- |
| `has(Component)` | Matches if the component is present on the entity. |
| `absent(Component)` | Matches if the component is absent from the entity. |
| `field(Component, "attr") == value` | Matches if the component is present and the attribute equals the value. |
| `all_of([gate1, gate2])` | Matches if all child gates match. |
| `any_of([gate1, gate2])` | Matches if any child gate matches. |
| `not_(gate)` | Negates the child gate. |

```python
from ecs_agent.workflows import has, absent, field, all_of, any_of, not_

# Complex gate example
gate = all_of([
    has(StatusComponent),
    field(StatusComponent, "value") == "ready",
    not_(has(ErrorComponent))
])
```

## Shared Prompt Profiles

Prompt profiles are grouped by `agent_key`. Multiple states can bind to the same profile. When a transition occurs between two states that share the same profile, the `${_workflow_state_prompt}` placeholder remains identical. This prevents unnecessary cache invalidation in the `SystemPromptRenderSystem`, ensuring stable prefix caching for LLM providers.

```python
profiles = {
    "main": {
        "standard": "You are a helpful assistant.",
        "expert": "You are a subject matter expert.",
    }
}

states = {
    "IDLE": {
        "bind": {"main": "standard"},
        "go": {"ACTIVE": has(StartFlag)},
    },
    "ACTIVE": {
        "bind": {"main": "standard"}, # Shared profile: no prompt churn on transition
        "go": {"DONE": has(EndFlag)},
    }
}
```

## System Ordering Contract

For workflows to function correctly, systems must be registered in a specific order. The `WorkflowStateSystem` must run after any systems that mutate components used in gates (like `UserPromptNormalizationSystem` script handlers) and before the `SystemPromptRenderSystem`.

Recommended priorities:
1. **`UserPromptNormalizationSystem`** (priority -30): Processes triggers and runs script handlers that might attach "gate" components.
2. **`WorkflowStateSystem`** (priority -25): Evaluates gates and commits transitions.
3. **`SystemPromptRenderSystem`** (priority -20): Resolves `${_workflow_state_prompt}` based on the newly committed state.
4. **`ReasoningSystem`** (priority 0): Performs LLM inference using the rendered prompt.

## TriggerSpec Interaction

Triggers defined in `UserPromptConfigComponent` can use `action="script"` to run Python code when a keyword or event is matched. These scripts can attach components to the entity, which the `WorkflowStateSystem` will observe in the same tick. This allows slash commands to drive immediate state transitions.

```python
async def handle_activate(world, eid, text):
    world.add_component(eid, FlagComponent())
    return None

# User types "/activate" -> FlagComponent attached -> Workflow transitions to ACTIVE -> Prompt updates
```

## Checkpoint and Resume

The workflow system is designed to be serializable, with one important exception: the `WorkflowDefinitionComponent`.

- **`WorkflowRuntimeComponent`**: Contains the `current_state_id` and `transition_history`. This component IS serialized and restored.
- **`WorkflowDefinitionComponent`**: Contains the compiled graph. This component is NOT serialized because it may contain non-serializable callables or large static data.

**Resume Contract**: After loading a world from a checkpoint (e.g., via `Runner.load_checkpoint`), the caller must re-install the workflow definition using `install_workflow` or manually adding the `WorkflowDefinitionComponent`. The runtime state will be preserved, and the agent will resume from the correct state.

## Transition Semantics

The `WorkflowStateSystem` evaluates all transitions from the current state in every tick.

- **0 matches**: No transition occurs. The state remains unchanged.
- **1 match**: The transition is committed. `current_state_id` is updated, the transition ID is appended to `transition_history`, and a `WorkflowLastTransitionComponent` is attached for the current tick.
- **>1 match**: This is considered an ambiguous state. The system attaches an `ErrorComponent` and a `TerminalComponent` with the reason `workflow_ambiguous_transition`, halting execution.
