"""Entity-bound authoring API for new turns, continuation, and cancellation."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from copy import deepcopy
from typing import Literal
from uuid import uuid4

from ecs_agent.components import (
    AgentTurnComponent, ChildStubComponent, ConversationComponent, ConversationTreeComponent,
    ErrorComponent, InterruptionComponent, LLMComponent, PendingToolCallsComponent,
    RunnerStateComponent, SessionRefComponent, TerminalComponent, UserInputComponent,
)
from ecs_agent.core.lifecycle import ActiveAgentRun, finish_turn
from ecs_agent.core.run_result import RunResult, RunStatus, snapshot_run_result
from ecs_agent.core.runner import Runner
from ecs_agent.core.streaming import AgentStreamEvent, stream_run
from ecs_agent.core.world import World
from ecs_agent.history import HistorySnapshot, append_conversation_message, replace_conversation_history
from ecs_agent.logging import get_logger
from ecs_agent.providers._message_contract import tool_span_error
from ecs_agent.prompts.contracts import SystemPromptConfigSpec
from ecs_agent.prompts.editing import PromptInspection, inspect_prompt, update_prompt
from ecs_agent.types import EntityId, Message

logger = get_logger(__name__)


class AgentHandle:
    """Bind authoring operations to an existing ECS entity and its World."""

    def __init__(self, world: World, entity: EntityId) -> None:
        if not world.has_entity(entity) or not world.has_component(entity, LLMComponent):
            raise ValueError("AgentHandle requires an existing entity with LLMComponent")
        if world.has_component(entity, ChildStubComponent):
            raise ValueError("A delegation stub cannot be driven as an agent")
        if world.has_component(entity, ConversationTreeComponent):
            raise ValueError("AgentHandle requires flat conversation history; native tree writes are not supported")
        self.world = world
        self.entity = entity

    @property
    def last_result(self) -> RunResult | None:
        """Most recent run snapshot, shared by handles bound to this entity."""
        return self.world._agent_results.get(self.entity)

    def history_snapshot(self) -> HistorySnapshot:
        """Copy history including nested message content, without exposing mutable state."""
        conversation = self.world.get_component(self.entity, ConversationComponent)
        if conversation is None:
            return HistorySnapshot("", 0, ())
        return HistorySnapshot(conversation.history_id, conversation.revision, tuple(deepcopy(conversation.messages)))

    def replace_history(
        self, messages: list[Message], *, reason: str,
        expected_history_id: str | None = None, expected_revision: int | None = None,
    ) -> None:
        """Commit a complete replacement while idle, optionally guarding against stale edits."""
        self._check_editable()
        snapshot = self.history_snapshot()
        if (expected_history_id is not None and snapshot.history_id != expected_history_id) or (expected_revision is not None and snapshot.revision != expected_revision):
            raise ValueError("History edit is based on a stale history ID or revision")
        if any(not isinstance(message, Message) or message.role not in {"system", "user", "assistant", "tool"} for message in messages):
            raise ValueError("History must contain valid Message instances")
        if error := tool_span_error(messages):
            raise ValueError(f"Replacement history has an invalid tool protocol: {error}")
        replace_conversation_history(self.world, self.entity, messages, reason=reason)
        self._clear_history_views()

    def reset_session(self, *, reason: str) -> None:
        """Atomically switch to an empty session while preserving the old transcript."""
        self._check_editable()
        replace_conversation_history(self.world, self.entity, [], reason=reason, new_conversation=True)
        self._clear_history_views()
        for component_type in (AgentTurnComponent, TerminalComponent, ErrorComponent, InterruptionComponent):
            self.world.remove_component(self.entity, component_type)
        self.world._agent_results.pop(self.entity, None)

    def set_prompt(self, prompt: str | SystemPromptConfigSpec) -> None:
        """Commit a new prompt between turns and invalidate derived and remote state."""
        self._check_editable()
        update_prompt(self.world, self.entity, prompt)

    def inspect_prompt(self) -> PromptInspection:
        """Read the last rendered prompt and source information without running callbacks."""
        return inspect_prompt(self.world, self.entity)

    def adopt_prompt(self, *, source: Literal["config", "system", "llm"] | None = None) -> None:
        """Migrate legacy prompt fields, rejecting ambiguity unless a source is explicit."""
        from ecs_agent.components import RenderedSystemPromptComponent, SystemPromptComponent

        self._check_editable()
        configured = self.world.get_component(self.entity, SystemPromptConfigSpec)
        legacy = self.world.get_component(self.entity, SystemPromptComponent)
        llm = self.world.get_component(self.entity, LLMComponent)
        rendered = self.world.get_component(self.entity, RenderedSystemPromptComponent)
        candidates: dict[str, str | SystemPromptConfigSpec] = {}
        if configured is not None:
            candidates["config"] = configured
        legacy_text = (legacy.template or legacy.content) if legacy is not None else ""
        for name, text in (("system", legacy_text), ("llm", llm.system_prompt if llm is not None else "")):
            if not text:
                continue
            if source != name and configured is not None and (
                text == configured.template_source.inline or (rendered is not None and text == rendered.text)
            ):
                # Canonical templates and their compatibility render copies
                # describe one source, not independent conflicting inputs.
                continue
            candidates[name] = text
        if source is not None:
            if source not in candidates:
                raise ValueError(f"Prompt source '{source}' is unavailable")
            selected = candidates[source]
        elif not candidates:
            selected = ""
        else:
            selected = next(iter(candidates.values()))
            if any(candidate != selected for candidate in candidates.values()):
                raise ValueError(f"Conflicting prompt sources: {sorted(candidates)}; choose source explicitly")
        update_prompt(self.world, self.entity, selected)

    def _check_editable(self) -> None:
        self._check_idle(None)
        turn = self.world.get_component(self.entity, AgentTurnComponent)
        if turn is not None and (turn.status not in {"completed", "abandoned"} or turn.unconfirmed_tool_calls or turn.input_normalization_state == "running"):
            raise ValueError("Finish or safely abandon the current turn before editing agent context")
        self._check_closed_tool_history()

    def _clear_history_views(self) -> None:
        from ecs_agent.components import ContextCacheComponent, PromptContextReservationComponent, RenderedUserPromptComponent

        for component_type in (ContextCacheComponent, PromptContextReservationComponent, RenderedUserPromptComponent):
            self.world.remove_component(self.entity, component_type)

    async def run_turn(self, message: str | Message, *, max_steps: int | None = 100) -> RunResult:
        """Submit one input and execute a new turn within this invocation's budget.

        Args:
            message: A user message or text. An unfinished turn must be resumed.
            max_steps: Number of ticks for this invocation; None is unlimited.
        """
        return await self._start(message, max_steps=max_steps)

    async def resume(self, *, max_steps: int | None = 100) -> RunResult:
        """Continue a safe paused turn without adding input or replaying tools."""
        return await self._start(None, max_steps=max_steps)

    def stream_turn(
        self, message: str | Message, *, max_steps: int | None = 100,
    ) -> AbstractAsyncContextManager[AsyncIterator[AgentStreamEvent]]:
        """Stream a new turn inside ``async with``; exiting cancels unfinished work.

        Events include a final RunCompletedEvent with its RunResult. The context
        manager makes early loop exits deterministic instead of relying on GC.
        """
        return stream_run(self, message, max_steps=max_steps)

    def stream_resume(
        self, *, max_steps: int | None = 100,
    ) -> AbstractAsyncContextManager[AsyncIterator[AgentStreamEvent]]:
        """Stream continuation of the same paused turn inside ``async with``."""
        return stream_run(self, None, max_steps=max_steps)

    async def cancel(self) -> RunResult | None:
        """Stop this entity's active run and return its retained snapshot."""
        active = self.world._agent_run
        if active is None or active.entity != self.entity or active.task is None:
            return self.last_result
        if active.task is asyncio.current_task():
            raise RuntimeError("An agent driver cannot await its own cancellation")
        active.task.cancel()
        try:
            await asyncio.shield(active.task)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        return self.last_result

    def adopt_turn(self) -> str:
        """Explicitly adopt legacy context without uncertain pending tool calls.

        This does not infer whether a legacy checkpoint had completed. The host
        explicitly chooses to continue its current context; no input is added.
        """
        self._check_idle(None)
        previous = self.world.get_component(self.entity, AgentTurnComponent)
        if previous is not None and previous.status != "legacy":
            raise ValueError("Only unmanaged or legacy context can be adopted")
        self._check_closed_tool_history()
        if self.world.has_component(self.entity, TerminalComponent):
            raise ValueError("Explicitly resolve the legacy terminal state before adopting context")
        turn = AgentTurnComponent(turn_id=uuid4().hex, status="paused", reason="legacy_context_adopted")
        self.world.add_component(self.entity, turn)
        return turn.turn_id

    def abandon_turn(self, *, reason: str) -> None:
        """Explicitly end a stopped turn whose tool history is fully resolved.

        This permits a new input after a failed or paused turn without claiming
        that the previous attempt succeeded. Unknown tool outcomes cannot be
        bypassed. Cancel an active run before abandoning its turn.
        """
        self._check_idle(None)
        turn = self.world.get_component(self.entity, AgentTurnComponent)
        if turn is None or not reason.strip():
            raise ValueError("Abandoning a turn requires an existing turn and an explicit reason")
        if turn.input_normalization_state == "running":
            raise ValueError("Input normalization completion is unconfirmed; reconcile the trigger first")
        if turn.unconfirmed_tool_calls:
            raise ValueError("Tool completion is unconfirmed; it must be reconciled first")
        self._check_closed_tool_history()
        turn.status = "abandoned"
        turn.reason = reason
        for component_type in (ErrorComponent, TerminalComponent, InterruptionComponent):
            self.world.remove_component(self.entity, component_type)

    def _check_idle(self, max_steps: int | None) -> None:
        if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0):
            raise ValueError("max_steps must be a non-negative integer or None")
        if self.world._agent_run is not None or self.world._runner_task is not None or self.world._processing_task is not None:
            raise RuntimeError("A Runner is already driving this World")
        if not self.world.has_entity(self.entity):
            raise ValueError("The agent entity no longer exists")
        if self.world.has_component(self.entity, ConversationTreeComponent):
            raise ValueError("AgentHandle requires flat conversation history; native tree writes are not supported")
        if self.world.has_component(self.entity, SessionRefComponent) and (
            self.world.session_service is None or self.entity not in self.world._session_writer_tokens
        ):
            raise ValueError("Rebind session recording and its writer before driving or editing a restored agent")

    def _check_closed_tool_history(self) -> None:
        pending = self.world.get_component(self.entity, PendingToolCallsComponent)
        conversation = self.world.get_component(self.entity, ConversationComponent)
        if pending is not None or (conversation is not None and tool_span_error(conversation.messages)):
            raise ValueError("Unresolved tool calls must be reconciled before starting or adopting a turn")

    def _start(self, message: str | Message | None, *, max_steps: int | None) -> asyncio.Task[RunResult]:
        try:
            self._check_idle(max_steps)
            from ecs_agent.components import UserPromptConfigComponent

            user_prompt = self.world.get_component(self.entity, UserPromptConfigComponent)
            if user_prompt is not None:
                missing_scripts = [trigger.content for trigger in user_prompt.triggers if trigger.action == "script" and trigger.content not in user_prompt.script_handlers]
                if missing_scripts:
                    raise ValueError(f"Rebind script trigger handlers before accepting input: {missing_scripts}")
            previous = self.world.get_component(self.entity, AgentTurnComponent)
            if message is None:
                if previous is None or previous.status != "paused":
                    reason = previous.reason if previous is not None else "no managed turn"
                    raise ValueError(f"Turn cannot be resumed: {reason}")
                if previous.input_normalization_state == "running":
                    raise ValueError("Input normalization completion is unconfirmed; automatic replay is prohibited")
                if previous.unconfirmed_tool_calls:
                    raise ValueError("Tool completion is unconfirmed; automatic replay is prohibited")
                error = self.world.get_component(self.entity, ErrorComponent)
                terminal = self.world.get_component(self.entity, TerminalComponent)
                if error is not None or terminal is not None:
                    raise ValueError("Resolve the turn's error or terminal state before resuming")
                turn = previous
                turn.status = "running"
                self.world.remove_component(self.entity, InterruptionComponent)
            else:
                if previous is not None and previous.status not in {"completed", "abandoned"}:
                    raise ValueError(f"Previous turn is {previous.status}; resume or explicitly resolve it first")
                if previous is not None and previous.unconfirmed_tool_calls:
                    raise ValueError("Tool completion is unconfirmed; a new input cannot bypass it")
                self._check_closed_tool_history()
                if self.world.has_component(self.entity, ErrorComponent):
                    raise ValueError("Resolve the agent error before starting a turn")
                terminal = self.world.get_component(self.entity, TerminalComponent)
                if terminal is not None and (previous is None or terminal.reason != "reasoning_complete"):
                    raise ValueError("Resolve the existing terminal state before starting a turn")
                incoming = Message(role="user", content=message) if isinstance(message, str) else deepcopy(message)
                if incoming.role != "user":
                    raise ValueError("run_turn accepts only user input")
                turn = AgentTurnComponent(turn_id=uuid4().hex, status="running")
                incoming.turn_id = turn.turn_id
                self.world.add_component(self.entity, turn)
                try:
                    append_conversation_message(self.world, self.entity, incoming)
                except Exception:
                    if previous is None:
                        self.world.remove_component(self.entity, AgentTurnComponent)
                    else:
                        self.world.add_component(self.entity, previous)
                    raise
                self.world.remove_component(self.entity, TerminalComponent)
                self.world.remove_component(self.entity, InterruptionComponent)
            user_input = self.world.get_component(self.entity, UserInputComponent)
            if user_input is not None:
                if user_input.future is not None and not user_input.future.done():
                    user_input.future.cancel()
                self.world.remove_component(self.entity, UserInputComponent)
            active = ActiveAgentRun(
                entity=self.entity, runner=Runner(),
                input_text=incoming.content if message is not None else None,
            )
            self.world._agent_run = active
            active.task = asyncio.create_task(self._drive(active, max_steps))
            active.task.add_done_callback(lambda task: self._finished(active, task))
            return active.task
        except (ValueError, RuntimeError) as exc:
            logger.error("agent_turn_rejected", entity_id=int(self.entity), exception=str(exc))
            raise

    async def _drive(self, active: ActiveAgentRun, max_steps: int | None) -> RunResult:
        states = self.world.query(RunnerStateComponent)
        start_tick = states[0][1][0].current_tick if states else 0
        try:
            return await active.runner.run(
                self.world,
                max_ticks=None if max_steps is None else start_tick + max_steps,
                start_tick=start_tick,
                target_entity=self.entity,
                run_id=active.run_id,
            )
        except BaseException as exc:
            if active.runner.last_result is None:
                result = snapshot_run_result(
                    self.world, run_id=active.run_id, ticks=0, target_entity=self.entity,
                    reason="external_cancellation" if isinstance(exc, asyncio.CancelledError) else "exception",
                    error=None if isinstance(exc, asyncio.CancelledError) else str(exc),
                )
                finish_turn(self.world, result)
            raise
        finally:
            if active.runner.last_result is not None:
                finish_turn(self.world, active.runner.last_result)
            if self.world._agent_run is active:
                self.world._agent_run = None

    def _finished(self, active: ActiveAgentRun, task: asyncio.Task[RunResult]) -> None:
        # A task cancelled before its coroutine starts never executes _drive's
        # finally block. Finalize that case as well, without retaining the lock.
        if self.world._agent_run is not active:
            return
        error = None if task.cancelled() else task.exception()
        result = snapshot_run_result(
            self.world, run_id=active.run_id, ticks=0, target_entity=self.entity,
            reason="external_cancellation" if task.cancelled() else "exception",
            error=str(error) if error is not None else None,
        )
        finish_turn(self.world, result)
        self.world._agent_run = None
