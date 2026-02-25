"""Tests for Runner checkpoint save/load and resume functionality."""

from pathlib import Path

import pytest

from ecs_agent.components.definitions import (
    ConversationComponent,
    LLMComponent,
    RunnerStateComponent,
    TerminalComponent,
)
from ecs_agent.core.runner import Runner
from ecs_agent.core.world import World
from ecs_agent.types import EntityId, Message


class DummyProvider:
    """Test provider for serialization."""

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
        provider = DummyProvider()
        entity = world.create_entity()
        world.add_component(entity, LLMComponent(provider=provider, model="test"))
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
        provider = DummyProvider()
        entity = world.create_entity()
        world.add_component(entity, LLMComponent(provider=provider, model="test"))
        world.add_component(
            entity, ConversationComponent(messages=[Message(role="user", content="hi")])
        )

        runner.save_checkpoint(world, tmp_checkpoint_path)

        loaded_world, loaded_tick = Runner.load_checkpoint(
            tmp_checkpoint_path, providers={"test": provider}, tool_handlers={}
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
        provider = DummyProvider()
        counter = CounterSystem()
        entity = world.create_entity()
        world.add_component(entity, LLMComponent(provider=provider, model="test"))
        world.register_system(counter, priority=0)

        # Run 5 ticks
        await runner.run(world, max_ticks=5)
        assert counter.run_count == 5

        # Save checkpoint
        runner.save_checkpoint(world, tmp_checkpoint_path)

        # Load checkpoint and resume for 3 more ticks
        loaded_world, start_tick = Runner.load_checkpoint(
            tmp_checkpoint_path, providers={"test": provider}, tool_handlers={}
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
        provider = DummyProvider()
        counter = CounterSystem()
        entity = world.create_entity()
        world.add_component(entity, LLMComponent(provider=provider, model="test"))
        world.register_system(counter, priority=0)

        # Run 7 ticks
        await runner.run(world, max_ticks=7)
        assert counter.run_count == 7

        # Save checkpoint
        runner.save_checkpoint(world, tmp_checkpoint_path)

        # Load checkpoint and resume with max_ticks=10 (should run only 3 more)
        loaded_world, start_tick = Runner.load_checkpoint(
            tmp_checkpoint_path, providers={"test": provider}, tool_handlers={}
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
        provider = DummyProvider()
        entity = world.create_entity()
        world.add_component(entity, LLMComponent(provider=provider, model="test"))
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
            tmp_checkpoint_path, providers={"test": provider}, tool_handlers={}
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
