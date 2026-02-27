import asyncio
import shutil

import pytest

from ecs_agent.components import SandboxConfigComponent, ToolRegistryComponent
from ecs_agent.core import World
from ecs_agent.skills import SkillManager
from ecs_agent.tools.bwrap_sandbox import bwrap_execute, wrap_sandbox_handler
from ecs_agent.types import EntityId, ToolSchema


class _FakeProcess:
    def __init__(
        self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_bwrap_execute_falls_back_to_shell_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_shell(
        command: str,
        stdout: int,
        stderr: int,
    ) -> _FakeProcess:
        _ = stdout
        _ = stderr
        calls.append(command)
        return _FakeProcess(stdout=b"ok\n")

    monkeypatch.setattr("ecs_agent.tools.bwrap_sandbox._BWRAP_AVAILABLE", False)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_shell)

    result = await bwrap_execute("echo ok")

    assert result == "ok"
    assert calls == ["echo ok"]


@pytest.mark.asyncio
async def test_bwrap_execute_uses_bwrap_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    async def fake_exec(*args: object, stdout: int, stderr: int) -> _FakeProcess:
        _ = stdout
        _ = stderr
        calls.append(args)
        return _FakeProcess(stdout=b"wrapped\n")

    monkeypatch.setattr("ecs_agent.tools.bwrap_sandbox._BWRAP_AVAILABLE", True)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    result = await bwrap_execute("echo wrapped")

    assert result == "wrapped"
    assert len(calls) == 1
    assert calls[0][0] == "bwrap"
    assert calls[0][-3:] == ("sh", "-c", "echo wrapped")


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not installed")
@pytest.mark.asyncio
async def test_bwrap_execute_real_command_when_bwrap_installed() -> None:
    result = await bwrap_execute("echo hello")
    assert result == "hello"


@pytest.mark.asyncio
async def test_wrap_sandbox_handler_returns_original_when_schema_not_compatible() -> (
    None
):
    async def handler(command: str) -> str:
        return command

    schema = ToolSchema(name="bash", description="run shell", parameters={})
    config = SandboxConfigComponent(sandbox_mode="bwrap")

    wrapped = wrap_sandbox_handler(handler, schema, config)
    assert wrapped is handler


@pytest.mark.asyncio
async def test_wrap_sandbox_handler_returns_original_for_asyncio_mode() -> None:
    async def handler(command: str) -> str:
        return command

    schema = ToolSchema(
        name="bash",
        description="run shell",
        parameters={},
        sandbox_compatible=True,
    )
    config = SandboxConfigComponent(sandbox_mode="asyncio")

    wrapped = wrap_sandbox_handler(handler, schema, config)
    assert wrapped is handler


@pytest.mark.asyncio
async def test_wrap_sandbox_handler_intercepts_command_when_bwrap_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    async def handler(command: str) -> str:
        return f"handler:{command}"

    async def fake_bwrap(command: str, timeout: float = 30.0) -> str:
        _ = timeout
        seen.append(command)
        return "bwrap-result"

    monkeypatch.setattr("ecs_agent.tools.bwrap_sandbox.bwrap_execute", fake_bwrap)

    schema = ToolSchema(
        name="bash",
        description="run shell",
        parameters={},
        sandbox_compatible=True,
    )
    config = SandboxConfigComponent(sandbox_mode="bwrap")
    wrapped = wrap_sandbox_handler(handler, schema, config)

    result = await wrapped(command="echo hi")

    assert result == "bwrap-result"
    assert seen == ["echo hi"]


@pytest.mark.asyncio
async def test_wrap_sandbox_handler_keeps_non_command_arguments() -> None:
    async def handler(value: str) -> str:
        return f"original:{value}"

    schema = ToolSchema(
        name="utility",
        description="not shell",
        parameters={},
        sandbox_compatible=True,
    )
    config = SandboxConfigComponent(sandbox_mode="bwrap")
    wrapped = wrap_sandbox_handler(handler, schema, config)

    result = await wrapped(value="x")
    assert result == "original:x"


class _Skill:
    name = "sandbox-skill"
    description = "sandbox skill"

    def tools(self) -> dict[str, tuple[ToolSchema, object]]:
        async def bash_tool(command: str) -> str:
            return f"raw:{command}"

        return {
            "bash": (
                ToolSchema(
                    name="bash",
                    description="shell",
                    parameters={},
                    sandbox_compatible=True,
                ),
                bash_tool,
            )
        }

    def system_prompt(self) -> str:
        return ""

    def install(self, world: World, entity_id: EntityId) -> None:
        _ = world
        _ = entity_id

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        _ = world
        _ = entity_id


@pytest.mark.asyncio
async def test_skill_manager_wraps_sandbox_compatible_tools_when_bwrap_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, SandboxConfigComponent(sandbox_mode="bwrap"))
    manager = SkillManager()

    async def fake_bwrap(command: str, timeout: float = 30.0) -> str:
        _ = timeout
        return f"wrapped:{command}"

    monkeypatch.setattr("ecs_agent.tools.bwrap_sandbox.bwrap_execute", fake_bwrap)

    manager.install(world, entity_id, _Skill())

    registry = world.get_component(entity_id, ToolRegistryComponent)
    assert registry is not None
    result = await registry.handlers["bash"](command="echo hi")
    assert result == "wrapped:echo hi"
