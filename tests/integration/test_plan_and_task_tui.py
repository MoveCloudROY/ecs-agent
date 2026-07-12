"""Tests for the plan-and-task TUI view model, event bridge, and Textual app."""

from __future__ import annotations

import asyncio

from ecs_agent.accounting.models import LLMInvocationEvent, UsageRecord
from ecs_agent.components import UserInputComponent
from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.core import World
from ecs_agent.types import (
    DelegationCompletedEvent,
    DelegationStartedEvent,
    EntityId,
    ErrorOccurredEvent,
    PhaseChangedEvent,
    PromptReplacementEvent,
    ReasoningCompleteEvent,
    StreamContentDeltaEvent,
    StreamContentStartEvent,
    StreamEndEvent,
    StreamReasoningDeltaEvent,
    StreamReasoningEndEvent,
    StreamStartEvent,
    SubagentStreamDeltaEvent,
    ToolCall,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    UserInputReceivedEvent,
    UserInputRequestedEvent,
)
from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH
from examples.e2e.plan_and_task.state_models import RuntimeState, TaskRecord
from examples.e2e.plan_and_task.tui.bridge import PlanTaskTuiBridge
from examples.e2e.plan_and_task.tui.view_model import (
    PlanTaskViewModel,
    TranscriptEntry,
    UiChange,
)

AGENT = EntityId(1)
OTHER = EntityId(99)

PHASE_IDS = tuple(PLAN_TASK_PHASE_GRAPH.phases_by_id)


def make_vm() -> PlanTaskViewModel:
    return PlanTaskViewModel(agent_id=AGENT, phase_ids=PHASE_IDS)


def transcript_kinds(vm: PlanTaskViewModel) -> list[str]:
    return [entry.kind for entry in vm.transcript]


def sections(changes: list[UiChange]) -> set[str]:
    return {change.section for change in changes}


class TestViewModelStreaming:
    def test_content_stream_lifecycle_flushes_assistant_entry(self) -> None:
        vm = make_vm()
        vm.apply_event(StreamStartEvent(entity_id=AGENT, timestamp=0.0))
        vm.apply_event(StreamContentDeltaEvent(entity_id=AGENT, delta="Hello "))
        changes = vm.apply_event(StreamContentDeltaEvent(entity_id=AGENT, delta="world"))
        assert vm.live_content == "Hello world"
        assert sections(changes) == {"live"}

        changes = vm.apply_event(StreamEndEvent(entity_id=AGENT, timestamp=1.0))
        assert vm.live_content == ""
        assert not vm.streaming
        entries = [e for c in changes for e in c.entries]
        assert [e.kind for e in entries] == ["assistant"]
        assert entries[0].text == "Hello world"
        assert transcript_kinds(vm) == ["assistant"]

    def test_reasoning_stream_flushes_before_content(self) -> None:
        vm = make_vm()
        vm.apply_event(StreamStartEvent(entity_id=AGENT, timestamp=0.0))
        vm.apply_event(
            StreamReasoningDeltaEvent(entity_id=AGENT, reasoning_delta="thinking...")
        )
        changes = vm.apply_event(StreamReasoningEndEvent(entity_id=AGENT))
        entries = [e for c in changes for e in c.entries]
        assert [e.kind for e in entries] == ["reasoning"]
        assert entries[0].text == "thinking..."

        vm.apply_event(StreamContentDeltaEvent(entity_id=AGENT, delta="answer"))
        vm.apply_event(StreamEndEvent(entity_id=AGENT, timestamp=1.0))
        assert transcript_kinds(vm) == ["reasoning", "assistant"]

    def test_stream_end_flushes_unterminated_reasoning(self) -> None:
        vm = make_vm()
        vm.apply_event(StreamStartEvent(entity_id=AGENT, timestamp=0.0))
        vm.apply_event(
            StreamReasoningDeltaEvent(entity_id=AGENT, reasoning_delta="partial")
        )
        vm.apply_event(StreamEndEvent(entity_id=AGENT, timestamp=1.0))
        assert transcript_kinds(vm) == ["reasoning"]
        assert vm.live_reasoning == ""

    def test_empty_stream_produces_no_transcript_entry(self) -> None:
        vm = make_vm()
        vm.apply_event(StreamStartEvent(entity_id=AGENT, timestamp=0.0))
        vm.apply_event(StreamEndEvent(entity_id=AGENT, timestamp=1.0))
        assert vm.transcript == []

    def test_events_for_other_entities_are_ignored(self) -> None:
        vm = make_vm()
        changes = vm.apply_event(
            StreamContentDeltaEvent(entity_id=OTHER, delta="nope")
        )
        assert changes == []
        assert vm.live_content == ""


class TestViewModelTranscript:
    def test_user_input_received_appends_user_entry(self) -> None:
        vm = make_vm()
        changes = vm.apply_event(
            UserInputReceivedEvent(entity_id=AGENT, prompt="You> ", text="/plan:status")
        )
        assert transcript_kinds(vm) == ["user"]
        assert vm.transcript[0].text == "/plan:status"
        # A user message also starts a busy turn (status section).
        assert sections(changes) == {"transcript", "status"}

    def test_script_prompt_replacement_appends_command_entry(self) -> None:
        vm = make_vm()
        changes = vm.apply_event(
            PromptReplacementEvent(
                entity_id=AGENT,
                prompt_kind="user",
                source_text="/plan:status",
                rendered_text='{"phase": "IDLE"}',
                metadata={"trigger_action": "script"},
            )
        )
        assert transcript_kinds(vm) == ["command"]
        assert vm.transcript[0].text == '{"phase": "IDLE"}'
        assert sections(changes) == {"transcript"}

    def test_non_script_prompt_replacement_is_ignored(self) -> None:
        vm = make_vm()
        changes = vm.apply_event(
            PromptReplacementEvent(
                entity_id=AGENT,
                prompt_kind="user",
                source_text="hello",
                rendered_text="hello rendered",
            )
        )
        assert changes == []
        assert vm.transcript == []

    def test_error_event_appends_error_entry(self) -> None:
        vm = make_vm()
        changes = vm.apply_event(
            ErrorOccurredEvent(entity_id=AGENT, error="boom", system_name="Reasoning")
        )
        assert transcript_kinds(vm) == ["error"]
        assert "boom" in vm.transcript[0].text
        assert any(c.section == "notify" for c in changes)


class TestViewModelTools:
    def test_tool_lifecycle_appends_call_and_result(self) -> None:
        vm = make_vm()
        vm.apply_event(
            ToolExecutionStartedEvent(
                entity_id=AGENT,
                tool_call=ToolCall(
                    id="t1", name="read_file", arguments={"path": "plan/draft.md"}
                ),
            )
        )
        assert transcript_kinds(vm) == ["tool_call"]
        assert "read_file" in vm.transcript[0].text
        assert "plan/draft.md" in vm.transcript[0].text

        vm.apply_event(
            ToolExecutionCompletedEvent(
                entity_id=AGENT,
                tool_call_id="t1",
                result="file contents here",
                success=True,
                tool_name="read_file",
                duration_seconds=0.25,
            )
        )
        assert transcript_kinds(vm) == ["tool_call", "tool_result"]
        assert "read_file" in vm.transcript[1].text

    def test_failed_tool_result_is_marked(self) -> None:
        vm = make_vm()
        vm.apply_event(
            ToolExecutionCompletedEvent(
                entity_id=AGENT,
                tool_call_id="t2",
                result="no such file",
                success=False,
                tool_name="read_file",
            )
        )
        assert vm.transcript[-1].kind == "tool_result"
        assert vm.transcript[-1].meta.get("success") == "false"


class TestViewModelSubagents:
    def test_delegation_lifecycle_tracks_runs(self) -> None:
        vm = make_vm()
        vm.apply_event(
            DelegationStartedEvent(
                entity_id=AGENT,
                subagent_name="advisor",
                task="review the draft plan",
                correlation_id="c-1",
                traceparent="",
            )
        )
        assert len(vm.subagent_runs) == 1
        run = vm.subagent_runs[0]
        assert run.name == "advisor"
        assert run.status == "running"

        vm.apply_event(
            SubagentStreamDeltaEvent(
                session_id="s-1",
                parent_entity_id=AGENT,
                category="advisor",
                child_world_name="advisor-world",
                seq=1,
                timestamp="2026-07-11T00:00:00Z",
                delta="approved because...",
            )
        )
        assert vm.subagent_runs[0].stream_chars > 0

        changes = vm.apply_event(
            DelegationCompletedEvent(
                entity_id=AGENT,
                subagent_name="advisor",
                result="approved",
                success=True,
                correlation_id="c-1",
                traceparent="",
            )
        )
        assert vm.subagent_runs[0].status == "completed"
        assert "subagent" in transcript_kinds(vm)
        assert "subagents" in sections(changes)

    def test_failed_delegation_marks_run_failed(self) -> None:
        vm = make_vm()
        vm.apply_event(
            DelegationStartedEvent(
                entity_id=AGENT,
                subagent_name="qa",
                task="qa review",
                correlation_id="c-2",
                traceparent="",
            )
        )
        vm.apply_event(
            DelegationCompletedEvent(
                entity_id=AGENT,
                subagent_name="qa",
                result="",
                success=False,
                error="timeout",
                correlation_id="c-2",
                traceparent="",
            )
        )
        assert vm.subagent_runs[0].status == "failed"


class TestViewModelPhasesAndUsage:
    def test_phase_change_updates_current_phase(self) -> None:
        vm = make_vm()
        assert vm.current_phase == "IDLE"
        changes = vm.apply_event(
            PhaseChangedEvent(
                entity_id=AGENT,
                graph_id="plan-task",
                from_phase="IDLE",
                to_phase="DRAFT_INTERVIEW",
                reason="plan_start",
                forced=True,
                tick=3,
            )
        )
        assert vm.current_phase == "DRAFT_INTERVIEW"
        assert "phases" in sections(changes)
        assert any(
            e.kind == "system" and "DRAFT_INTERVIEW" in e.text
            for c in changes
            for e in c.entries
        )

    def test_usage_accumulates_across_invocations(self) -> None:
        vm = make_vm()
        vm.apply_event(
            LLMInvocationEvent(
                entity_id=int(AGENT),
                provider_id="openai",
                model="qwen3.5-flash",
                usage=UsageRecord(
                    prompt_tokens=100, completion_tokens=20, total_tokens=120
                ),
            )
        )
        changes = vm.apply_event(
            LLMInvocationEvent(
                entity_id=int(AGENT),
                provider_id="openai",
                model="qwen3.5-flash",
                usage=UsageRecord(
                    prompt_tokens=200,
                    completion_tokens=30,
                    total_tokens=230,
                    cache_read_tokens=150,
                ),
            )
        )
        assert vm.usage.invocations == 2
        assert vm.usage.prompt_tokens == 300
        assert vm.usage.completion_tokens == 50
        assert vm.usage.total_tokens == 350
        assert vm.usage.cache_read_tokens == 150
        assert sections(changes) == {"usage"}

    def test_refresh_runtime_populates_workflow_and_tasks(self) -> None:
        vm = make_vm()
        state = RuntimeState(
            workflow_id="demo-workflow",
            phase="TASK_RUNNING",
            status="active",
            active_plan_file="plan/workflow_plan.md",
            current_task_id="task-002",
            review_verdicts=[],
            active_subagents=[],
            memory_refs=[],
            last_checkpoint=None,
            created_at="2026-07-11T00:00:00Z",
            updated_at="2026-07-11T00:00:00Z",
            tasks=[
                TaskRecord(task_id="task-001", title="First", status="completed"),
                TaskRecord(
                    task_id="task-002",
                    title="Second",
                    status="in_progress",
                    retry_count=1,
                ),
            ],
        )
        change = vm.refresh_runtime(state)
        assert change.section == "tasks"
        assert vm.workflow_id == "demo-workflow"
        assert [t.task_id for t in vm.tasks] == ["task-001", "task-002"]
        assert vm.tasks[1].retry_count == 1
        assert vm.current_task_id == "task-002"


class TestViewModelActivity:
    async def test_turn_activity_lifecycle(self) -> None:
        vm = make_vm()
        assert vm.activity == "idle"
        assert not vm.busy

        changes = vm.apply_event(
            UserInputReceivedEvent(entity_id=AGENT, prompt="You> ", text="hi")
        )
        assert vm.busy
        assert vm.activity == "waiting"
        assert "status" in sections(changes)

        changes = vm.apply_event(StreamStartEvent(entity_id=AGENT, timestamp=0.0))
        assert vm.activity == "thinking"
        assert "status" in sections(changes)

        vm.apply_event(
            StreamReasoningDeltaEvent(entity_id=AGENT, reasoning_delta="hmm")
        )
        assert vm.activity == "thinking"

        changes = vm.apply_event(StreamContentStartEvent(entity_id=AGENT))
        assert vm.activity == "generating"
        assert "status" in sections(changes)

        # Repeated deltas in the same state do not re-emit a status change.
        changes = vm.apply_event(StreamContentDeltaEvent(entity_id=AGENT, delta="x"))
        assert "status" not in sections(changes)

        vm.apply_event(StreamEndEvent(entity_id=AGENT, timestamp=1.0))
        assert vm.activity == "thinking"

        changes = vm.apply_event(
            ToolExecutionStartedEvent(
                entity_id=AGENT,
                tool_call=ToolCall(id="t1", name="read_file", arguments={}),
            )
        )
        assert vm.activity == "tool"
        assert vm.activity_detail == "read_file"
        assert "status" in sections(changes)

        vm.apply_event(
            ToolExecutionCompletedEvent(
                entity_id=AGENT,
                tool_call_id="t1",
                result="ok",
                success=True,
                tool_name="read_file",
            )
        )
        assert vm.activity == "thinking"

        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        changes = vm.apply_event(
            UserInputRequestedEvent(
                entity_id=AGENT, prompt="You> ", input_future=future
            )
        )
        assert not vm.busy
        assert vm.activity == "idle"
        assert "status" in sections(changes)

    def test_subagent_delegation_sets_activity(self) -> None:
        vm = make_vm()
        vm.apply_event(
            UserInputReceivedEvent(entity_id=AGENT, prompt="You> ", text="go")
        )
        vm.apply_event(
            DelegationStartedEvent(
                entity_id=AGENT,
                subagent_name="advisor",
                task="review",
                correlation_id="c-1",
                traceparent="",
            )
        )
        assert (vm.activity, vm.activity_detail) == ("subagent", "advisor")
        vm.apply_event(
            DelegationCompletedEvent(
                entity_id=AGENT,
                subagent_name="advisor",
                result="approved",
                success=True,
                correlation_id="c-1",
                traceparent="",
            )
        )
        assert vm.activity == "thinking"

    def test_turn_token_estimate_grows_with_deltas(self) -> None:
        vm = make_vm()
        vm.apply_event(
            UserInputReceivedEvent(entity_id=AGENT, prompt="You> ", text="hi")
        )
        assert vm.turn_output_tokens == 0
        vm.apply_event(StreamStartEvent(entity_id=AGENT, timestamp=0.0))
        vm.apply_event(
            StreamReasoningDeltaEvent(entity_id=AGENT, reasoning_delta="abcdefgh")
        )
        after_reasoning = vm.turn_output_tokens
        assert after_reasoning >= 2  # 8 ascii chars ≈ 2 tokens
        vm.apply_event(StreamContentDeltaEvent(entity_id=AGENT, delta="中文四字"))
        assert vm.turn_output_tokens >= after_reasoning + 4  # CJK ≈ 1 token/char
        assert vm.turn_tokens_estimated

    def test_invocation_usage_reconciles_estimate(self) -> None:
        vm = make_vm()
        vm.apply_event(
            UserInputReceivedEvent(entity_id=AGENT, prompt="You> ", text="hi")
        )
        vm.apply_event(StreamStartEvent(entity_id=AGENT, timestamp=0.0))
        vm.apply_event(
            StreamContentDeltaEvent(entity_id=AGENT, delta="some streamed text")
        )
        assert vm.turn_tokens_estimated
        vm.apply_event(StreamEndEvent(entity_id=AGENT, timestamp=1.0))
        vm.apply_event(
            LLMInvocationEvent(
                entity_id=int(AGENT),
                provider_id="openai",
                model="m",
                usage=UsageRecord(
                    prompt_tokens=10, completion_tokens=42, total_tokens=52
                ),
            )
        )
        assert vm.turn_output_tokens == 42
        assert not vm.turn_tokens_estimated

        # A second invocation in the same turn accumulates on top.
        vm.apply_event(StreamStartEvent(entity_id=AGENT, timestamp=2.0))
        vm.apply_event(StreamContentDeltaEvent(entity_id=AGENT, delta="moremore"))
        assert vm.turn_output_tokens > 42
        assert vm.turn_tokens_estimated

    def test_new_turn_resets_token_counter(self) -> None:
        vm = make_vm()
        vm.apply_event(
            UserInputReceivedEvent(entity_id=AGENT, prompt="You> ", text="one")
        )
        vm.apply_event(
            LLMInvocationEvent(
                entity_id=int(AGENT),
                provider_id="openai",
                model="m",
                usage=UsageRecord(
                    prompt_tokens=1, completion_tokens=7, total_tokens=8
                ),
            )
        )
        assert vm.turn_output_tokens == 7
        vm.apply_event(
            UserInputReceivedEvent(entity_id=AGENT, prompt="You> ", text="two")
        )
        assert vm.turn_output_tokens == 0


class TestStatusBar:
    async def test_status_bar_reflects_busy_state(self) -> None:
        from textual.widgets import Static

        from examples.e2e.plan_and_task.tui.app import (
            SPINNER_FRAMES,
            PlanTaskTuiApp,
        )

        class StubSink:
            input_pending = False

            def submit_input(self, text: str) -> bool:
                return True

            def request_quit(self) -> None:
                return None

        vm = make_vm()
        app = PlanTaskTuiApp(view_model=vm, sink=StubSink())
        async with app.run_test(size=(100, 30)) as pilot:
            bar = app.query_one("#status-bar")
            assert not bar.display

            for change in vm.apply_event(
                UserInputReceivedEvent(entity_id=AGENT, prompt="You> ", text="hi")
            ):
                app.dispatch_change(change)
            await pilot.pause()
            assert bar.display
            label = str(app.query_one("#status-label", Static).content)
            assert "waiting" in label
            assert any(frame in label for frame in SPINNER_FRAMES)

            for change in vm.apply_event(
                StreamStartEvent(entity_id=AGENT, timestamp=0.0)
            ):
                app.dispatch_change(change)
            vm.apply_event(
                StreamContentDeltaEvent(entity_id=AGENT, delta="hello world data")
            )
            await pilot.pause(0.3)  # let the status timer tick
            label = str(app.query_one("#status-label", Static).content)
            assert "generating" in label
            tokens = str(app.query_one("#status-tokens", Static).content)
            assert "tok" in tokens
            assert "~" in tokens  # estimate marker while streaming

            future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            for change in vm.apply_event(
                UserInputRequestedEvent(
                    entity_id=AGENT, prompt="You> ", input_future=future
                )
            ):
                app.dispatch_change(change)
            await pilot.pause()
            assert not bar.display

    async def test_spinner_frame_advances_over_time(self) -> None:
        from textual.widgets import Static

        from examples.e2e.plan_and_task.tui.app import PlanTaskTuiApp

        class StubSink:
            input_pending = False

            def submit_input(self, text: str) -> bool:
                return True

            def request_quit(self) -> None:
                return None

        vm = make_vm()
        app = PlanTaskTuiApp(view_model=vm, sink=StubSink())
        async with app.run_test(size=(100, 30)) as pilot:
            for change in vm.apply_event(
                UserInputReceivedEvent(entity_id=AGENT, prompt="You> ", text="hi")
            ):
                app.dispatch_change(change)
            await pilot.pause()
            first = str(app.query_one("#status-label", Static).content)
            await pilot.pause(0.5)
            second = str(app.query_one("#status-label", Static).content)
            assert first != second  # spinner frame advanced


class TestBridge:
    async def _build(
        self,
    ) -> tuple[
        World, EntityId, PlanTaskViewModel, PlanTaskTuiBridge, list[UiChange]
    ]:
        world = World()
        agent_id = world.create_entity()
        vm = PlanTaskViewModel(agent_id=agent_id, phase_ids=PHASE_IDS)
        seen: list[UiChange] = []
        runtime_state_ref: list[RuntimeState | None] = [None]
        bridge = PlanTaskTuiBridge(
            world=world,
            agent_id=agent_id,
            view_model=vm,
            runtime_state_ref=runtime_state_ref,
            on_change=seen.append,
        )
        bridge.attach()
        return world, agent_id, vm, bridge, seen

    async def test_events_route_through_view_model(self) -> None:
        world, agent_id, vm, _bridge, seen = await self._build()
        await world.event_bus.publish(
            UserInputReceivedEvent(entity_id=agent_id, prompt="You> ", text="hi")
        )
        assert [e.kind for e in vm.transcript] == ["user"]
        assert any(c.section == "transcript" for c in seen)

    async def test_submit_resolves_pending_future(self) -> None:
        world, agent_id, _vm, bridge, _seen = await self._build()
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        await world.event_bus.publish(
            UserInputRequestedEvent(
                entity_id=agent_id, prompt="You> ", input_future=future
            )
        )
        assert bridge.input_pending
        assert bridge.submit_input("/plan:status")
        assert future.result() == "/plan:status"
        assert not bridge.input_pending

    async def test_submit_without_pending_future_is_rejected(self) -> None:
        _world, _agent_id, _vm, bridge, _seen = await self._build()
        assert not bridge.submit_input("hello")

    async def test_exit_command_sets_terminal_component(self) -> None:
        world, agent_id, _vm, bridge, _seen = await self._build()
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        await world.event_bus.publish(
            UserInputRequestedEvent(
                entity_id=agent_id, prompt="You> ", input_future=future
            )
        )
        assert bridge.submit_input("exit")
        assert future.done()
        assert world.get_component(agent_id, TerminalComponent) is not None

    async def test_request_quit_terminates_even_without_future(self) -> None:
        world, agent_id, _vm, bridge, _seen = await self._build()
        bridge.request_quit()
        assert world.get_component(agent_id, TerminalComponent) is not None

    async def test_reasoning_complete_rearms_user_input(self) -> None:
        world, agent_id, _vm, _bridge, _seen = await self._build()
        world.remove_component(agent_id, UserInputComponent)
        await world.event_bus.publish(
            ReasoningCompleteEvent(entity_id=agent_id, model="m", duration_ms=1.0)
        )
        assert world.get_component(agent_id, UserInputComponent) is not None

    async def test_attach_arms_initial_user_input(self) -> None:
        world, agent_id, _vm, _bridge, _seen = await self._build()
        assert world.get_component(agent_id, UserInputComponent) is not None


class TestEndToEnd:
    async def test_headless_session_runs_command_and_streams(
        self, tmp_path: object
    ) -> None:
        """Full loop: Runner + UserInputSystem + command handler + streaming.

        Uses the exact wiring the ``__main__`` entrypoint uses
        (``create_tui_session``) with a FakeModel, headless Textual app, and a
        real Runner ticking the world.
        """
        from pathlib import Path

        from ecs_agent.core import Runner
        from ecs_agent.providers import FakeModel
        from ecs_agent.types import CompletionResult, Message, Usage
        from examples.e2e.plan_and_task.main import build_plan_task_world
        from examples.e2e.plan_and_task.tui.session import create_tui_session

        assert isinstance(tmp_path, Path)
        model = FakeModel(
            responses=[
                CompletionResult(
                    message=Message(role="assistant", content="Streamed TUI reply."),
                    usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                ),
                CompletionResult(
                    message=Message(role="assistant", content="Goodbye."),
                    usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
                ),
            ]
        )
        world, agent_id, _adapter_ref, runtime_state = await build_plan_task_world(
            model=model, base_dir=tmp_path
        )
        session = create_tui_session(world, agent_id, runtime_state)

        async def wait_for(predicate: object, timeout: float = 8.0) -> None:
            assert callable(predicate)
            deadline = asyncio.get_running_loop().time() + timeout
            while not predicate():
                if asyncio.get_running_loop().time() > deadline:
                    raise AssertionError(f"timed out waiting for {predicate}")
                await asyncio.sleep(0.02)

        vm = session.view_model
        async with session.app.run_test() as pilot:
            runner_task = asyncio.create_task(
                Runner().run(world, max_ticks=None)
            )
            await wait_for(lambda: session.bridge.input_pending)
            assert session.bridge.submit_input("/plan:status")
            await wait_for(
                lambda: any(e.kind == "assistant" for e in vm.transcript)
            )
            await wait_for(lambda: session.bridge.input_pending)
            assert session.bridge.submit_input("exit")
            await asyncio.wait_for(runner_task, timeout=10)
            await pilot.pause()

        kinds = [entry.kind for entry in vm.transcript]
        assert "user" in kinds
        assert "command" in kinds
        assert "assistant" in kinds
        command_entry = next(e for e in vm.transcript if e.kind == "command")
        assert "No active workflow" in command_entry.text
        assistant_entry = next(e for e in vm.transcript if e.kind == "assistant")
        assert assistant_entry.text == "Streamed TUI reply."
        assert vm.live_content == ""
        assert not session.bridge.input_pending


class TestCopyOnSelect:
    class _Sink:
        def submit_input(self, text: str) -> bool:
            return True

        def request_quit(self) -> None:
            return None

    def _make_app(self) -> object:
        from examples.e2e.plan_and_task.tui.app import PlanTaskTuiApp

        return PlanTaskTuiApp(view_model=make_vm(), sink=self._Sink())

    async def test_transcript_get_selection_extracts_written_lines(self) -> None:
        from textual.geometry import Offset as TextualOffset
        from textual.selection import Selection

        from examples.e2e.plan_and_task.tui.app import (
            PlanTaskTuiApp,
            SelectableRichLog,
        )

        app = self._make_app()
        assert isinstance(app, PlanTaskTuiApp)
        async with app.run_test(size=(100, 30)):
            log = app.query_one("#transcript", SelectableRichLog)
            from rich.text import Text

            log.write(Text("alpha beta"))
            log.write(Text("gamma delta"))
            extracted = log.get_selection(
                Selection(TextualOffset(0, 0), TextualOffset(5, 1))
            )
            assert extracted is not None
            text, ending = extracted
            assert text == "alpha beta\ngamma"
            assert ending == "\n"

    async def test_mouse_selection_auto_copies_to_clipboard(self) -> None:
        from textual.geometry import Offset as TextualOffset

        from examples.e2e.plan_and_task.tui.app import PlanTaskTuiApp

        app = self._make_app()
        assert isinstance(app, PlanTaskTuiApp)
        notifications: list[str] = []
        async with app.run_test(size=(100, 30)) as pilot:
            app.notify = lambda message, **kwargs: notifications.append(  # type: ignore[method-assign]
                str(message)
            )
            app.dispatch_change(
                UiChange(
                    section="transcript",
                    entries=[TranscriptEntry(kind="user", text="hello copy me")],
                )
            )
            await pilot.pause()
            await pilot.mouse_down("#transcript", offset=TextualOffset(2, 1))
            await pilot.hover("#transcript", offset=TextualOffset(40, 2))
            await pilot.mouse_up("#transcript", offset=TextualOffset(40, 2))
            await pilot.pause()
        assert "hello copy me" in app.clipboard
        assert len(notifications) == 1

    async def test_partial_single_line_selection_copies_substring(self) -> None:
        from textual.geometry import Offset as TextualOffset

        from examples.e2e.plan_and_task.tui.app import PlanTaskTuiApp

        app = self._make_app()
        assert isinstance(app, PlanTaskTuiApp)
        async with app.run_test(size=(100, 30)) as pilot:
            app.dispatch_change(
                UiChange(
                    section="transcript",
                    entries=[TranscriptEntry(kind="user", text="hello copy me")],
                )
            )
            await pilot.pause()
            # Content line 1 is "❯ hello copy me"; widget offsets include the
            # round border (1) and horizontal padding (1): content (2, 1) is
            # the "h" at widget offset (4, 2).
            await pilot.mouse_down("#transcript", offset=TextualOffset(4, 2))
            await pilot.hover("#transcript", offset=TextualOffset(9, 2))
            await pilot.mouse_up("#transcript", offset=TextualOffset(9, 2))
            await pilot.pause()
        assert app.clipboard == "hello "

    async def test_cjk_partial_selection_extracts_characters(self) -> None:
        from textual.geometry import Offset as TextualOffset

        from examples.e2e.plan_and_task.tui.app import PlanTaskTuiApp

        app = self._make_app()
        assert isinstance(app, PlanTaskTuiApp)
        async with app.run_test(size=(100, 30)) as pilot:
            app.dispatch_change(
                UiChange(
                    section="transcript",
                    entries=[
                        TranscriptEntry(kind="user", text="hello copy me"),
                        TranscriptEntry(
                            kind="assistant", text="中文选择测试 CJK line here"
                        ),
                    ],
                )
            )
            await pilot.pause()
            # The CJK line is content line 3 (blank, user, blank, markdown);
            # double-width characters occupy cells 0..8 for the first four.
            await pilot.mouse_down("#transcript", offset=TextualOffset(2, 4))
            await pilot.hover("#transcript", offset=TextualOffset(10, 4))
            await pilot.mouse_up("#transcript", offset=TextualOffset(10, 4))
            await pilot.pause()
        assert app.clipboard == "中文选择测"

    async def test_drag_into_blank_area_selects_to_content_end(self) -> None:
        from textual.geometry import Offset as TextualOffset

        from examples.e2e.plan_and_task.tui.app import PlanTaskTuiApp

        app = self._make_app()
        assert isinstance(app, PlanTaskTuiApp)
        async with app.run_test(size=(100, 30)) as pilot:
            app.dispatch_change(
                UiChange(
                    section="transcript",
                    entries=[TranscriptEntry(kind="user", text="hello copy me")],
                )
            )
            await pilot.pause()
            await pilot.mouse_down("#transcript", offset=TextualOffset(4, 2))
            await pilot.hover("#transcript", offset=TextualOffset(30, 12))
            await pilot.mouse_up("#transcript", offset=TextualOffset(30, 12))
            await pilot.pause()
        assert app.clipboard == "hello copy me"

    async def test_plain_click_does_not_copy(self) -> None:
        from examples.e2e.plan_and_task.tui.app import PlanTaskTuiApp

        app = self._make_app()
        assert isinstance(app, PlanTaskTuiApp)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.click("#transcript")
            await pilot.pause()
        assert app.clipboard == ""

    async def test_repeated_identical_selection_copies_once(self) -> None:
        from textual.geometry import Offset as TextualOffset

        from examples.e2e.plan_and_task.tui.app import PlanTaskTuiApp

        app = self._make_app()
        assert isinstance(app, PlanTaskTuiApp)
        notifications: list[str] = []
        async with app.run_test(size=(100, 30)) as pilot:
            app.notify = lambda message, **kwargs: notifications.append(  # type: ignore[method-assign]
                str(message)
            )
            app.dispatch_change(
                UiChange(
                    section="transcript",
                    entries=[TranscriptEntry(kind="user", text="hello copy me")],
                )
            )
            await pilot.pause()
            for _ in range(2):
                await pilot.mouse_down("#transcript", offset=TextualOffset(2, 1))
                await pilot.hover("#transcript", offset=TextualOffset(40, 2))
                await pilot.mouse_up("#transcript", offset=TextualOffset(40, 2))
                await pilot.pause()
        assert "hello copy me" in app.clipboard
        assert len(notifications) == 1


class TestTextualApp:
    async def test_app_mounts_and_submits_input(self) -> None:
        from examples.e2e.plan_and_task.tui.app import CommandInput, PlanTaskTuiApp

        submitted: list[str] = []

        class StubSink:
            def submit_input(self, text: str) -> bool:
                submitted.append(text)
                return True

            def request_quit(self) -> None:
                submitted.append("<quit>")

        vm = make_vm()
        app = PlanTaskTuiApp(view_model=vm, sink=StubSink())
        async with app.run_test() as pilot:
            field = app.query_one(CommandInput)
            field.value = "/plan:status"
            await pilot.pause()
            field.post_message(field.Submitted(field, field.value))
            await pilot.pause()
        assert submitted == ["/plan:status"]

    async def test_app_renders_transcript_change(self) -> None:
        from examples.e2e.plan_and_task.tui.app import PlanTaskTuiApp

        class StubSink:
            def submit_input(self, text: str) -> bool:
                return True

            def request_quit(self) -> None:
                return None

        vm = make_vm()
        app = PlanTaskTuiApp(view_model=vm, sink=StubSink())
        async with app.run_test() as pilot:
            changes = vm.apply_event(
                UserInputReceivedEvent(
                    entity_id=AGENT, prompt="You> ", text="hello world"
                )
            )
            for change in changes:
                app.dispatch_change(change)
            await pilot.pause()
            entry = TranscriptEntry(kind="system", text="direct entry")
            app.dispatch_change(UiChange(section="transcript", entries=[entry]))
            await pilot.pause()


class TestMultilineInput:
    """The command input composes multi-line messages: Ctrl+J inserts a
    newline, Enter submits the whole (possibly multi-line) buffer."""

    def _make_app(self, submitted: list[str]) -> object:
        from examples.e2e.plan_and_task.tui.app import PlanTaskTuiApp

        class StubSink:
            input_pending = True

            def submit_input(self, text: str) -> bool:
                submitted.append(text)
                return True

            def request_quit(self) -> None:
                return None

        return PlanTaskTuiApp(view_model=make_vm(), sink=StubSink())

    async def test_ctrl_j_inserts_newline_without_submitting(self) -> None:
        from examples.e2e.plan_and_task.tui.app import CommandInput

        submitted: list[str] = []
        app = self._make_app(submitted)
        assert isinstance(app, object)
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one(CommandInput)  # type: ignore[attr-defined]
            field.focus()
            await pilot.press("a")
            await pilot.press("ctrl+j")
            await pilot.press("b")
            await pilot.pause()
            assert field.value == "a\nb"
            assert submitted == []  # Ctrl+J must not submit

    async def test_enter_submits_multiline_and_clears(self) -> None:
        from examples.e2e.plan_and_task.tui.app import CommandInput

        submitted: list[str] = []
        app = self._make_app(submitted)
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one(CommandInput)  # type: ignore[attr-defined]
            field.value = "first line"
            field.cursor_position = len(field.value)
            await pilot.press("ctrl+j")
            await pilot.press("s", "e", "c", "o", "n", "d")
            await pilot.pause()
            assert field.value == "first line\nsecond"
            await pilot.press("enter")
            await pilot.pause()
            assert submitted == ["first line\nsecond"]
            assert field.value == ""

    async def test_value_roundtrips_multiline_text(self) -> None:
        from examples.e2e.plan_and_task.tui.app import CommandInput

        app = self._make_app([])
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one(CommandInput)  # type: ignore[attr-defined]
            field.value = "x\ny\nz"
            await pilot.pause()
            assert field.value == "x\ny\nz"

    async def test_value_setter_places_cursor_at_end(self) -> None:
        from examples.e2e.plan_and_task.tui.app import CommandInput

        app = self._make_app([])
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one(CommandInput)  # type: ignore[attr-defined]
            field.value = "/plan"
            await pilot.pause()
            # Cursor sits at the end, so typing appends rather than prepends.
            await pilot.press("!")
            await pilot.pause()
            assert field.value == "/plan!"
            assert field.cursor_position == len(field.value)

    async def test_cursor_position_roundtrips_across_lines(self) -> None:
        from examples.e2e.plan_and_task.tui.app import CommandInput

        app = self._make_app([])
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one(CommandInput)  # type: ignore[attr-defined]
            field.value = "ab\ncd"
            # Index 4 is the "c": "ab\n" spans indices 0..2, so 3 lands on the
            # second line's start and 4 is one column in.
            field.cursor_position = 4
            await pilot.pause()
            assert field.cursor_position == 4


class TestCommandCompletion:
    COMMANDS = (
        ("/plan:finalize", "finalize the reviewed plan"),
        ("/plan:start", "<description> — start a new planning workflow"),
        ("/plan:status", "show current plan status"),
        ("/task:abort", "abort task execution"),
        ("/task:start", "[workflow_id] — initialize the task queue and run"),
    )

    def _make_app(self, submitted: list[str]) -> object:
        from examples.e2e.plan_and_task.tui.app import PlanTaskTuiApp

        class StubSink:
            input_pending = True

            def submit_input(self, text: str) -> bool:
                submitted.append(text)
                return True

            def request_quit(self) -> None:
                return None

        return PlanTaskTuiApp(
            view_model=make_vm(), sink=StubSink(), commands=self.COMMANDS
        )

    async def test_slash_shows_completion_list_with_hints(self) -> None:
        from textual.widgets import OptionList

        from examples.e2e.plan_and_task.tui.app import CommandInput

        app = self._make_app([])
        async with app.run_test(size=(100, 30)) as pilot:
            lst = app.query_one("#command-list", OptionList)
            assert not lst.display

            app.query_one(CommandInput).value = "/"
            await pilot.pause()
            assert lst.display
            assert lst.option_count == len(self.COMMANDS)
            first = str(lst.get_option_at_index(0).prompt)
            assert "/plan:finalize" in first
            assert "finalize the reviewed plan" in first

    async def test_typing_filters_completion_list(self) -> None:
        from textual.widgets import OptionList

        from examples.e2e.plan_and_task.tui.app import CommandInput

        app = self._make_app([])
        async with app.run_test(size=(100, 30)) as pilot:
            app.query_one(CommandInput).value = "/plan:s"
            await pilot.pause()
            lst = app.query_one("#command-list", OptionList)
            assert lst.display
            prompts = [
                str(lst.get_option_at_index(i).prompt)
                for i in range(lst.option_count)
            ]
            assert len(prompts) == 2
            assert any("/plan:start" in p for p in prompts)
            assert any("/plan:status" in p for p in prompts)

    async def test_plain_text_keeps_list_hidden(self) -> None:
        from textual.widgets import OptionList

        from examples.e2e.plan_and_task.tui.app import CommandInput

        app = self._make_app([])
        async with app.run_test(size=(100, 30)) as pilot:
            app.query_one(CommandInput).value = "describe the goal"
            await pilot.pause()
            assert not app.query_one("#command-list", OptionList).display

    async def test_tab_completes_highlighted_command(self) -> None:
        from examples.e2e.plan_and_task.tui.app import CommandInput

        app = self._make_app([])
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one(CommandInput)
            field.value = "/task:a"
            await pilot.pause()
            await pilot.press("tab")
            await pilot.pause()
            assert field.value == "/task:abort "
            assert field.cursor_position == len(field.value)

    async def test_enter_completes_partial_instead_of_submitting(self) -> None:
        from textual.widgets import OptionList

        from examples.e2e.plan_and_task.tui.app import CommandInput

        submitted: list[str] = []
        app = self._make_app(submitted)
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one(CommandInput)
            field.value = "/plan:f"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert field.value == "/plan:finalize "
            assert submitted == []
            # the completed value is no command prefix, so the list closes
            assert not app.query_one("#command-list", OptionList).display

    async def test_enter_submits_exact_command(self) -> None:
        from examples.e2e.plan_and_task.tui.app import CommandInput

        submitted: list[str] = []
        app = self._make_app(submitted)
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one(CommandInput)
            field.value = "/plan:status"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert submitted == ["/plan:status"]
            assert field.value == ""

    async def test_escape_hides_completion_list(self) -> None:
        from textual.widgets import OptionList

        from examples.e2e.plan_and_task.tui.app import CommandInput

        app = self._make_app([])
        async with app.run_test(size=(100, 30)) as pilot:
            app.query_one(CommandInput).value = "/"
            await pilot.pause()
            lst = app.query_one("#command-list", OptionList)
            assert lst.display
            await pilot.press("escape")
            await pilot.pause()
            assert not lst.display

    async def test_arrow_down_moves_highlight(self) -> None:
        from examples.e2e.plan_and_task.tui.app import CommandInput

        app = self._make_app([])
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one(CommandInput)
            field.value = "/plan:s"
            await pilot.pause()
            await pilot.press("down")
            await pilot.press("tab")
            await pilot.pause()
            assert field.value == "/plan:status "

    async def test_clicking_option_completes_command(self) -> None:
        from textual.widgets import OptionList

        from examples.e2e.plan_and_task.tui.app import CommandInput

        app = self._make_app([])
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one(CommandInput)
            field.value = "/task:a"
            await pilot.pause()
            lst = app.query_one("#command-list", OptionList)
            lst.action_select()  # same path as a mouse click on the option
            await pilot.pause()
            assert field.value == "/task:abort "
            assert app.focused is field

    async def test_world_triggers_carry_command_hints(self, tmp_path: object) -> None:
        """The real world wires a non-empty hint for every slash command."""
        from pathlib import Path

        from ecs_agent.providers import FakeModel
        from examples.e2e.plan_and_task.main import build_plan_task_world
        from examples.e2e.plan_and_task.tui.session import _slash_commands

        assert isinstance(tmp_path, Path)
        world, agent_id, _, _ = await build_plan_task_world(
            model=FakeModel(responses=[]), base_dir=tmp_path
        )
        commands = _slash_commands(world, agent_id)
        assert commands, "expected slash commands from the world's TriggerSpecs"
        for pattern, hint in commands:
            assert pattern.startswith("/")
            assert hint, f"command {pattern} is missing a completion hint"
