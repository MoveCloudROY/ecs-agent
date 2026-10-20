# Agent turns and run lifecycle

Use `AgentHandle` to submit input to an existing entity, continue a paused turn,
and consume a `RunResult`. The handle reuses the entity's components, systems,
history, and session recorder.

For a new agent, [build_agent](agent-authoring.md) installs the complete runtime
and returns this same handle. It also provides controlled Prompt/history editing
and system rebinding for checkpoints.

```python
from ecs_agent.core import AgentHandle, RunStatus

agent = AgentHandle(world, entity)
result = await agent.run_turn("Analyze the data", max_steps=20)
if result.status is RunStatus.MAX_TICKS:
    result = await agent.resume(max_steps=20)
if result.success:
    assert result.final_message is not None
    print(result.final_message.content)
```

The entity must have an `LLMComponent`, and its World must already contain the
systems needed for its capabilities. The handle creates flat conversation
history on first input when necessary. It does not install an agent runtime or
close shared providers and session stores.

Native ConversationTreeComponent writing remains outside this interface because
the existing tree reasoning path is read-only. AgentHandle rejects that setup
before accepting input, rather than appending input to a flat history that the
tree would hide. Existing low-level tree APIs remain available.

## Session turn and run identities

| Identity | Meaning |
| --- | --- |
| Session | The existing durable conversation identity, when recording is installed |
| Turn | One submitted user input and its processing, identified by `turn_id` |
| Run | One execution attempt with a budget, identified by `run_id` |

`resume()` retains `turn_id`, creates a new `run_id`, and does not append input.
`run_turn()` starts a new turn. Inputs and generated messages carry `turn_id`,
including session records. Current-turn output remains attributable after
working-history compaction: `AgentTurnComponent.last_message` retains the last
generated assistant message separately from the model-facing context.

New input emits one `UserInputReceivedEvent` with `source="agent_handle"`,
`turn_id`, and the submitting `run_id`. Continuation emits no duplicate input
event. RunStartedEvent also carries the target and turn identity. Existing
observability plugins retain their user-input trace integration without
re-emitting all historical user messages for every managed run.

`RunResult` is the stable outcome of one attempt. `MAX_TICKS`, `CANCELLED`,
`INTERRUPTED`, and errors are not successful final answers. A new turn that has
not generated output has no partial or final message; it never returns the
previous turn's answer. Handles bound to the same entity share `last_result`.

## Budgets and state transitions

`max_steps` counts completed ticks in this invocation. Zero accepts the input
and pauses before executing a tick, `None` means unlimited ticks, and negative
or non-integer values are rejected before input is submitted. A tick can await
model or tool I/O, so the step limit is not a timeout.

| State | Allowed next operation |
| --- | --- |
| No managed turn | `run_turn(message)` |
| Running | Await completion or call `cancel()` |
| Paused with no unresolved execution failure | `resume()` or `stream_resume()` |
| Completed | `run_turn(message)` |
| Failed with closed tool history | Resolve the failure or explicitly `abandon_turn(reason=...)`, then start a new turn |
| Tool completion unknown | Reconcile the tool outcome; input, resume, and abandonment cannot bypass it |
| Legacy checkpoint | Explicitly `adopt_turn()` when its context is safe to continue |

Starting a new turn clears only its predecessor's allowed lifecycle markers.
Managed turns do not rely on `TerminalCleanupSystem`, and `ErrorHandlingSystem`
retains their error information. `abandon_turn()` does not change the previous
RunResult into success. It refuses pending or unanswered tool calls and unknown
tool outcomes.

The low-level `Runner.run()` remains available. Its legacy arguments keep their
absolute-limit meaning: `start_tick=5, max_ticks=8` permits three ticks. Runner
checks pre-existing terminal state before executing systems. Exhausting a
budget returns `MAX_TICKS` without creating an unrelated terminal entity.
Repeated calls continue the cumulative counter in `RunnerStateComponent`.

## Execution scope and custom systems

An AgentHandle run executes the bound entity. Delegated child Worlds have their
own drivers; parent-World stubs do not execute. Only one driver may process a
World at a time. Concurrent handles, runners, and manual `world.process()` calls
are rejected while another driver is active.

Built-in agent systems use `world.query_active(...)`. Custom systems that mutate
agent entities must use the same API, or call `is_entity_active(world, entity)`:

```python
class MySystem:
    async def process(self, world: World) -> None:
        for entity, (conversation,) in world.query_active(ConversationComponent):
            # Process only the current scope and eligible managed turns.
            ...
```

`world.query()` always returns the complete matching World state. It remains
appropriate for inspection and persistence. A custom system that intentionally
uses unrestricted queries is responsible for respecting the execution scope;
the framework does not silently filter every query. World-level infrastructure,
such as message-bus routing, remains available.

## Streaming and cancellation

Use an async context manager so an early exit has a deterministic cleanup point:

```python
from ecs_agent.types import RunCompletedEvent, StreamContentDeltaEvent

async with agent.stream_turn("Explain the result", max_steps=20) as events:
    async for item in events:
        # Every envelope includes entity_id, turn_id, and run_id.
        if isinstance(item.event, StreamContentDeltaEvent):
            print(item.event.delta, end="", flush=True)
        elif isinstance(item.event, RunCompletedEvent):
            result = item.event.result
```

`stream_resume()` uses the same context-manager interface. Both paths share the
normal turn driver and final RunResult. Context exit cancels unfinished work,
removes its event subscriptions, and restores the previous streaming setting.
The scoped adapter uses ordered publication so the final result follows all
deltas; it rejects delayed events attributed to a previous run.

`await agent.cancel()` requests cancellation and returns the retained snapshot.
The task awaiting `run_turn()` or `resume()` still receives `CancelledError`.
External task cancellation has the same propagation contract. Cancelling before
the first tick also releases the World driver.

Cancellation during approval leaves an unstarted call resumable. A started tool
whose result has not been confirmed blocks automatic continuation, including
after a checkpoint. Completed results are removed from the pending queue as
they are committed, so a checkpoint between results cannot replay that prefix.

## Checkpoints and migration

```python
from ecs_agent.core import AgentHandle, Runner

Runner().save_checkpoint(world, path)
restored, tick = Runner.load_checkpoint(path, providers, tool_handlers)
# Reinstall systems and runtime dependencies on restored, as before.
result = await AgentHandle(restored, entity).resume(max_steps=20)
```

Checkpoint schema version 4 persists `AgentTurnComponent`, message turn identity,
and terminal states. Completed turns stay completed. Running turns become paused
on load unless a terminal or unconfirmed tool execution requires a different
state. Saving does not mutate the live World. Futures, driver tasks, locks,
execution scope, and reusable approval decisions are not serialized.

Versions 1 and 2 lack reliable turn boundaries and may have discarded terminal
markers. Their agent entities load with a `legacy` turn state. `adopt_turn()` is
an explicit choice to continue existing context, not a claim that the old run
was unfinished. It refuses unresolved tool calls and existing terminal states.
After safe adoption, call `resume()`. Do not delete unknown tool-start evidence
to force continuation. Session recording still requires explicit rebinding via
the existing session installation API.

Runnable examples: [basic chat](../../examples/chat_agent.py),
[typed tool registration](../../examples/tool_agent.py),
[scoped streaming](../../examples/streaming_system_agent.py), and
[multi-turn input with checkpoint continuation](../../examples/agent_session.py).
The last example accepts `--interactive` and requires no manual input-component
re-arming or terminal cleanup.

The plan-and-task CLI also uses `setup_turn_input()` and AgentHandle for its
interactive loop. Its existing `setup_interactive_input()` adapter remains for
legacy hosts; mixing that component-driven loop with managed turns is unnecessary.
