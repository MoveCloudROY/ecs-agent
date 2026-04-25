"""Tests for Runner checkpoint save/load and resume functionality."""

import asyncio
import json
from pathlib import Path

import pytest

from ecs_agent.components import ToolRegistryComponent
from ecs_agent.components.definitions import (
    ConversationComponent,
    LLMComponent,
    RunnerStateComponent,
    SubagentNotificationQueueComponent,
    SubagentRegistryComponent,
    SubagentSessionTableComponent,
    TerminalComponent,
)
from ecs_agent.core.runner import Runner
from ecs_agent.core.world import World
from ecs_agent.serialization import NON_SERIALIZABLE_PLACEHOLDER, WorldSerializer
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.types import EntityId, Message, SubagentConfig, SubagentSessionRecord


class DummyProvider:
    """Test model for serialization."""

    model_id: str = "test"

    async def complete(self, messages, tools=None, stream=False, response_format=None):
        _ = (messages, tools, stream, response_format)
        raise NotImplementedError


class CounterSystem:
    """Test system that counts how many times it runs."""

    def __init__(self, priority: int = 0) -> None:
        self.priority = priority
        self.run_count = 0

    async def process(self, world: World) -> None:
        self.run_count += 1


def _build_session_record(
    parent_entity_id: EntityId,
    session_id: str,
    *,
    category: str,
    prompt: str,
    created_at: str,
    updated_at: str,
    status: str,
    started_at: str | None = None,
    finished_at: str | None = None,
    result_excerpt: str | None = None,
    error: str | None = None,
) -> SubagentSessionRecord:
    return SubagentSessionRecord(
        session_id=session_id,
        category=category,
        prompt=prompt,
        parent_entity_id=parent_entity_id,
        created_at=created_at,
        updated_at=updated_at,
        status=status,
        background=True,
        started_at=started_at,
        finished_at=finished_at,
        result_excerpt=result_excerpt,
        error=error,
    )


class TerminateAtTickSystem:
    """Test system that adds TerminalComponent after N ticks."""

    def __init__(self, terminate_at_tick: int, priority: int = 0) -> None:
        self.priority = priority
        self.terminate_at_tick = terminate_at_tick
        self.tick_count = 0

    async def process(self, world: World) -> None:
        self.tick_count += 1
        if self.tick_count >= self.terminate_at_tick:
            entity_id = world.create_entity()
            world.add_component(entity_id, TerminalComponent(reason="test_termination"))


class TestRunnerResume:
    """Test Runner checkpoint save/load and resume behavior."""

    @pytest.fixture
    def tmp_checkpoint_path(self, tmp_path: Path) -> Path:
        """Create a temporary checkpoint file path."""
        return tmp_path / "checkpoint.json"

    @pytest.fixture
    def world(self) -> World:
        """Create a fresh World instance."""
        return World()

    @pytest.fixture
    def runner(self) -> Runner:
        """Create Runner instance."""
        return Runner()

    @pytest.mark.asyncio
    async def test_save_checkpoint_saves_world_state_to_json(
        self, world: World, runner: Runner, tmp_checkpoint_path: Path
    ) -> None:
        """Test that save_checkpoint saves world state to JSON file via WorldSerializer."""
        model = DummyProvider()
        entity = world.create_entity()
        world.add_component(entity, LLMComponent(model=model))
        world.add_component(
            entity, ConversationComponent(messages=[Message(role="user", content="hi")])
        )

        runner.save_checkpoint(world, tmp_checkpoint_path)

        assert tmp_checkpoint_path.exists()
        import json

        data = json.loads(tmp_checkpoint_path.read_text(encoding="utf-8"))
        assert "entities" in data
        assert "next_entity_id" in data
        assert "runner_state" in data

    @pytest.mark.asyncio
    async def test_load_checkpoint_restores_world_from_file(
        self, world: World, runner: Runner, tmp_checkpoint_path: Path
    ) -> None:
        """Test that load_checkpoint restores world from file."""
        model = DummyProvider()
        entity = world.create_entity()
        world.add_component(entity, LLMComponent(model=model))
        world.add_component(
            entity, ConversationComponent(messages=[Message(role="user", content="hi")])
        )

        runner.save_checkpoint(world, tmp_checkpoint_path)

        loaded_world, loaded_tick = Runner.load_checkpoint(
            tmp_checkpoint_path, providers={"test": model}, tool_handlers={}
        )

        assert loaded_world.has_component(EntityId(1), LLMComponent)
        assert loaded_world.has_component(EntityId(1), ConversationComponent)
        conv = loaded_world.get_component(EntityId(1), ConversationComponent)
        assert conv is not None
        assert len(conv.messages) == 1
        assert conv.messages[0].content == "hi"

    @pytest.mark.asyncio
    async def test_runner_state_component_tracks_current_tick(
        self, world: World, runner: Runner
    ) -> None:
        """Test that RunnerStateComponent tracks current_tick during execution."""
        counter = CounterSystem()
        world.register_system(counter, priority=0)

        await runner.run(world, max_ticks=5)

        # RunnerStateComponent should be attached to an entity
        runner_state_entities = list(world.query(RunnerStateComponent))
        assert len(runner_state_entities) == 1
        _, (runner_state,) = runner_state_entities[0]
        assert runner_state.current_tick == 5

    @pytest.mark.asyncio
    async def test_resume_continues_from_saved_tick_count(
        self, world: World, runner: Runner, tmp_checkpoint_path: Path
    ) -> None:
        """Test that resume continues from saved tick count (doesn't restart from 0)."""
        model = DummyProvider()
        counter = CounterSystem()
        entity = world.create_entity()
        world.add_component(entity, LLMComponent(model=model))
        world.register_system(counter, priority=0)

        # Run 5 ticks
        await runner.run(world, max_ticks=5)
        assert counter.run_count == 5

        # Save checkpoint
        runner.save_checkpoint(world, tmp_checkpoint_path)

        # Load checkpoint and resume for 3 more ticks
        loaded_world, start_tick = Runner.load_checkpoint(
            tmp_checkpoint_path, providers={"test": model}, tool_handlers={}
        )
        assert start_tick == 5

        # Create new runner and counter for loaded world
        new_runner = Runner()
        new_counter = CounterSystem()
        loaded_world.register_system(new_counter, priority=0)

        await new_runner.run(loaded_world, max_ticks=8, start_tick=start_tick)

        # Counter should run only 3 times (8 - 5 = 3)
        assert new_counter.run_count == 3

        # RunnerStateComponent should show total of 8 ticks
        runner_state_entities = list(loaded_world.query(RunnerStateComponent))
        assert len(runner_state_entities) == 1
        _, (runner_state,) = runner_state_entities[0]
        assert runner_state.current_tick == 8

    @pytest.mark.asyncio
    async def test_resume_respects_remaining_max_ticks(
        self, world: World, runner: Runner, tmp_checkpoint_path: Path
    ) -> None:
        """Test that resume respects remaining max_ticks (total - already_run)."""
        model = DummyProvider()
        counter = CounterSystem()
        entity = world.create_entity()
        world.add_component(entity, LLMComponent(model=model))
        world.register_system(counter, priority=0)

        # Run 7 ticks
        await runner.run(world, max_ticks=7)
        assert counter.run_count == 7

        # Save checkpoint
        runner.save_checkpoint(world, tmp_checkpoint_path)

        # Load checkpoint and resume with max_ticks=10 (should run only 3 more)
        loaded_world, start_tick = Runner.load_checkpoint(
            tmp_checkpoint_path, providers={"test": model}, tool_handlers={}
        )

        new_runner = Runner()
        new_counter = CounterSystem()
        loaded_world.register_system(new_counter, priority=0)

        await new_runner.run(loaded_world, max_ticks=10, start_tick=start_tick)

        # Counter should run only 3 times (10 - 7 = 3)
        assert new_counter.run_count == 3

    @pytest.mark.asyncio
    async def test_save_load_roundtrip_preserves_conversation_messages(
        self, world: World, runner: Runner, tmp_checkpoint_path: Path
    ) -> None:
        """Test that save/load round-trip preserves conversation messages."""
        model = DummyProvider()
        entity = world.create_entity()
        world.add_component(entity, LLMComponent(model=model))
        world.add_component(
            entity,
            ConversationComponent(
                messages=[
                    Message(role="user", content="hello"),
                    Message(role="assistant", content="hi there"),
                    Message(role="user", content="how are you?"),
                ]
            ),
        )

        runner.save_checkpoint(world, tmp_checkpoint_path)

        loaded_world, _ = Runner.load_checkpoint(
            tmp_checkpoint_path, providers={"test": model}, tool_handlers={}
        )

        conv = loaded_world.get_component(EntityId(1), ConversationComponent)
        assert conv is not None
        assert len(conv.messages) == 3
        assert conv.messages[0].content == "hello"
        assert conv.messages[1].content == "hi there"
        assert conv.messages[2].content == "how are you?"

    @pytest.mark.asyncio
    async def test_load_from_nonexistent_path_raises_file_not_found_error(
        self, tmp_path: Path
    ) -> None:
        """Test that load from non-existent path raises FileNotFoundError."""
        nonexistent_path = tmp_path / "does_not_exist.json"

        with pytest.raises(FileNotFoundError):
            Runner.load_checkpoint(nonexistent_path, providers={}, tool_handlers={})

    @pytest.mark.asyncio
    async def test_resume_restore_reconciles_queued_and_running_subagent_sessions(
        self,
        world: World,
        runner: Runner,
        tmp_checkpoint_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import ecs_agent.systems.subagent_runtime as runtime_module

        monkeypatch.setattr(runtime_module, "_GLOBAL_SCHEDULER", None)

        model = DummyProvider()
        parent = world.create_entity()
        world.add_component(parent, LLMComponent(model=model))
        world.add_component(
            parent,
            ConversationComponent(
                messages=[Message(role="user", content="resume subagents")]
            ),
        )
        world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
        world.add_component(
            parent,
            SubagentRegistryComponent(
                subagents={
                    "queued-agent": SubagentConfig(
                        name="queued-agent", model=model
                    ),
                    "running-agent": SubagentConfig(
                        name="running-agent", model=model
                    ),
                    "done-agent": SubagentConfig(
                        name="done-agent", model=model
                    ),
                }
            ),
        )
        world.add_component(
            parent,
            SubagentSessionTableComponent(
                sessions={
                    "queued-b": _build_session_record(
                        parent,
                        "queued-b",
                        category="queued-agent",
                        prompt="second queued",
                        created_at="2026-04-05T10:01:00Z",
                        updated_at="2026-04-05T10:01:00Z",
                        status="queued",
                    ),
                    "queued-a": _build_session_record(
                        parent,
                        "queued-a",
                        category="queued-agent",
                        prompt="first queued",
                        created_at="2026-04-05T10:00:00Z",
                        updated_at="2026-04-05T10:00:00Z",
                        status="queued",
                    ),
                    "running-z": _build_session_record(
                        parent,
                        "running-z",
                        category="running-agent",
                        prompt="running when checkpointed",
                        created_at="2026-04-05T09:58:00Z",
                        updated_at="2026-04-05T10:02:00Z",
                        status="running",
                        started_at="2026-04-05T09:59:00Z",
                    ),
                    "done-c": _build_session_record(
                        parent,
                        "done-c",
                        category="done-agent",
                        prompt="already finished",
                        created_at="2026-04-05T09:50:00Z",
                        updated_at="2026-04-05T09:55:00Z",
                        status="succeeded",
                        started_at="2026-04-05T09:50:30Z",
                        finished_at="2026-04-05T09:55:00Z",
                        result_excerpt="completed before restore",
                    ),
                }
            ),
        )

        checkpoint_data = WorldSerializer.to_dict(world)
        subagents = checkpoint_data["entities"][str(int(parent))][
            "SubagentRegistryComponent"
        ]["subagents"]
        for subagent_data in subagents.values():
            subagent_data["provider"] = NON_SERIALIZABLE_PLACEHOLDER

        checkpoint_data["runner_state"] = {"current_tick": 0}
        tmp_checkpoint_path.write_text(json.dumps(checkpoint_data), encoding="utf-8")

        loaded_world, _ = Runner.load_checkpoint(
            tmp_checkpoint_path,
            providers={"default": model, "test": model},
            tool_handlers={},
        )

        system = SubagentSystem(max_background_concurrency=1)
        system.install_subagent_control_tools(loaded_world, parent)

        start_events = {
            "queued-a": asyncio.Event(),
            "queued-b": asyncio.Event(),
        }
        release_first = asyncio.Event()
        started_order: list[str] = []

        async def fake_execute_core(
            world_arg: World,
            parent_entity_id: EntityId,
            subagent_name: str,
            task: str,
            correlation_id: str,
            traceparent: str,
            config_snapshot: SubagentConfig,
        ) -> tuple[str, bool, str | None]:
            del (
                world_arg,
                parent_entity_id,
                correlation_id,
                traceparent,
                config_snapshot,
            )
            if task == "first queued":
                started_order.append("queued-a")
                start_events["queued-a"].set()
                await release_first.wait()
                return ("queued-a-result", True, None)

            started_order.append("queued-b")
            start_events["queued-b"].set()
            return (f"{subagent_name}:{task}", True, None)

        system._execute_subagent_core = fake_execute_core  # type: ignore[method-assign]

        await system.process(loaded_world)
        await asyncio.wait_for(start_events["queued-a"].wait(), timeout=1.0)
        await asyncio.sleep(0)

        tools = loaded_world.get_component(parent, ToolRegistryComponent)
        assert tools is not None

        status_handler = tools.handlers["subagent_status"]
        result_handler = tools.handlers["subagent_result"]

        queued_status = json.loads(await status_handler(session_id="queued-b"))
        assert queued_status["lifecycle_status"] == "queued"
        assert queued_status["queue_position"] == 0

        running_result = json.loads(
            await result_handler(session_id="running-z", timeout=None)
        )
        assert running_result["status"] == "terminal"
        assert running_result["lifecycle_status"] == "failed"
        assert running_result["error"] == "restored_without_live_task_handle"

        terminal_result = json.loads(
            await result_handler(session_id="done-c", timeout=None)
        )
        assert terminal_result["status"] == "success"
        assert terminal_result["lifecycle_status"] == "succeeded"

        scheduler = runtime_module._GLOBAL_SCHEDULER
        assert scheduler is not None
        assert [item.session_id for item in scheduler.pending_queue] == ["queued-b"]
        assert started_order == ["queued-a"]

        await system.process(loaded_world)
        await asyncio.sleep(0)

        assert [item.session_id for item in scheduler.pending_queue] == ["queued-b"]
        assert started_order == ["queued-a"]

        release_first.set()
        first_task = await system._runtime_manager.get_task("queued-a")
        assert first_task is not None
        await first_task
        await asyncio.wait_for(start_events["queued-b"].wait(), timeout=1.0)

        queued_result = json.loads(
            await asyncio.wait_for(
                result_handler(session_id="queued-b", timeout=None), timeout=1.0
            )
        )
        assert queued_result["status"] == "success"
        assert queued_result["lifecycle_status"] == "succeeded"

        restored_table = loaded_world.get_component(
            parent, SubagentSessionTableComponent
        )
        assert restored_table is not None
        assert restored_table.sessions["running-z"].status == "failed"
        assert restored_table.sessions["running-z"].finished_at is not None
        assert restored_table.sessions["done-c"].status == "succeeded"
        assert (
            restored_table.sessions["done-c"].result_excerpt
            == "completed before restore"
        )
        assert (
            restored_table.sessions["queued-b"].result_excerpt
            == "queued-agent:second queued"
        )

    @pytest.mark.asyncio
    async def test_restored_running_session_enqueues_one_failure_notification(
        self,
        world: World,
        runner: Runner,
        tmp_checkpoint_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import ecs_agent.systems.subagent_runtime as runtime_module

        monkeypatch.setattr(runtime_module, "_GLOBAL_SCHEDULER", None)

        model = DummyProvider()
        parent = world.create_entity()
        world.add_component(parent, LLMComponent(model=model))
        world.add_component(
            parent,
            ConversationComponent(messages=[Message(role="user", content="resume")]),
        )
        world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))
        world.add_component(
            parent,
            SubagentRegistryComponent(
                subagents={
                    "running-agent": SubagentConfig(
                        name="running-agent", model=model
                    )
                }
            ),
        )
        world.add_component(
            parent,
            SubagentSessionTableComponent(
                sessions={
                    "running-z": _build_session_record(
                        parent,
                        "running-z",
                        category="running-agent",
                        prompt="running when checkpointed",
                        created_at="2026-04-05T09:58:00Z",
                        updated_at="2026-04-05T10:02:00Z",
                        status="running",
                        started_at="2026-04-05T09:59:00Z",
                    )
                }
            ),
        )

        checkpoint_data = WorldSerializer.to_dict(world)
        subagents = checkpoint_data["entities"][str(int(parent))][
            "SubagentRegistryComponent"
        ]["subagents"]
        for subagent_data in subagents.values():
            subagent_data["provider"] = NON_SERIALIZABLE_PLACEHOLDER

        checkpoint_data["runner_state"] = {"current_tick": 0}
        tmp_checkpoint_path.write_text(json.dumps(checkpoint_data), encoding="utf-8")

        loaded_world, _ = Runner.load_checkpoint(
            tmp_checkpoint_path,
            providers={"default": model, "test": model},
            tool_handlers={},
        )

        system = SubagentSystem(max_background_concurrency=1)
        system.install_subagent_control_tools(loaded_world, parent)

        await system.process(loaded_world)

        table = loaded_world.get_component(parent, SubagentSessionTableComponent)
        assert table is not None
        assert table.sessions["running-z"].status == "failed"
        assert table.sessions["running-z"].error == "restored_without_live_task_handle"

        queue = loaded_world.get_component(parent, SubagentNotificationQueueComponent)
        assert queue is not None
        assert [item.notification_id for item in queue.notifications] == [
            "running-z:failed"
        ]
        assert queue.notifications[0].error == "restored_without_live_task_handle"
        assert queue.notifications[0].delivered_at is None

        await system.process(loaded_world)

        assert [item.notification_id for item in queue.notifications] == [
            "running-z:failed"
        ]
