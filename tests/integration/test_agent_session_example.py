"""Public multi-turn and checkpoint example works without internal re-arming."""

from examples.agent_session import main


async def test_agent_session_checkpoint_demo(monkeypatch, capsys) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    await main([])
    output = capsys.readouterr().out
    assert output.count("Assistant: Demo reply:") == 2
    assert "Paused turn:" in output


async def test_agent_session_interactive_loop(monkeypatch, capsys) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    inputs = iter(["first", "second", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    await main(["--interactive"])
    output = capsys.readouterr().out
    assert "Assistant: Demo reply: first" in output
    assert "Assistant: Demo reply: second" in output


async def test_plan_task_cli_turn_reader_needs_no_component_rearming(monkeypatch) -> None:
    from examples.e2e.plan_and_task.runtime import setup_turn_input
    from ecs_agent.components import ConversationComponent, LLMComponent, UserInputComponent
    from ecs_agent.core import AgentHandle, World
    from ecs_agent.providers import FakeModel
    from ecs_agent.systems.reasoning import ReasoningSystem
    from ecs_agent.types import CompletionResult, Message

    inputs = iter(["first", "", "second", "", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    world = World()
    entity = world.create_entity()
    model = FakeModel([
        CompletionResult(message=Message(role="assistant", content="one")),
        CompletionResult(message=Message(role="assistant", content="two")),
    ])
    world.add_component(entity, LLMComponent(model=model))
    world.register_system(ReasoningSystem(), priority=0)
    reader = await setup_turn_input(world, entity)
    handle = AgentHandle(world, entity)
    for expected in ("first", "second"):
        assert not world.has_component(entity, UserInputComponent)
        text = await reader()
        assert text == expected
        assert (await handle.run_turn(text)).success
    assert await reader() is None
    history = world.get_component(entity, ConversationComponent)
    assert [message.content for message in history.messages] == ["first", "one", "second", "two"]
    assert model._index == 2
