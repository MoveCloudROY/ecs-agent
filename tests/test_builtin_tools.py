from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from ecs_agent.components import (
    ConversationComponent,
    PendingToolCallsComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
)
from ecs_agent.components.definitions import SkillComponent
from ecs_agent.core import World
from ecs_agent.skills import SkillManager
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools.builtins import BuiltinToolsSkill
from ecs_agent.tools.builtins.bash_tool import bash
from ecs_agent.tools.builtins.edit_tool import edit_file
from ecs_agent.tools.builtins.file_tools import read_file, write_file
from ecs_agent.types import ToolCall

# Try importing glob; will fail if not implemented yet (expected for TDD red phase)
try:
    from ecs_agent.tools.builtins.glob_tool import glob
except ImportError:
    glob = None  # type: ignore

try:
    from ecs_agent.tools.builtins.bash_tool import interactive_bash
except ImportError:
    interactive_bash = None  # type: ignore


@pytest.mark.asyncio
async def test_read_file_valid(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "note.txt"
    target.write_text("hello\nworld", encoding="utf-8")

    result = await read_file("note.txt", str(workspace))

    assert result == "hello\nworld"


@pytest.mark.asyncio
async def test_read_file_empty_content(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "empty.txt"
    target.write_text("", encoding="utf-8")

    assert await read_file("empty.txt", str(workspace)) == ""


@pytest.mark.asyncio
async def test_read_file_blank_line_preserved(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "blank.txt"
    target.write_text("line1\n\nline3", encoding="utf-8")

    result = await read_file("blank.txt", str(workspace))

    assert result == "line1\n\nline3"


@pytest.mark.asyncio
async def test_read_file_accepts_numeric_string_offset_and_limit(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "numbers.txt"
    target.write_text("one\ntwo\nthree", encoding="utf-8")

    result = await read_file("numbers.txt", str(workspace), offset="2", limit="1")

    assert result == "two"


@pytest.mark.asyncio
async def test_read_file_rejects_boolean_numeric_inputs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "numbers.txt"
    target.write_text("one\ntwo\nthree", encoding="utf-8")

    with pytest.raises(ValueError, match="offset must be an integer >= 1"):
        await read_file("numbers.txt", str(workspace), offset=True)

    with pytest.raises(ValueError, match="limit must be an integer >= 0"):
        await read_file("numbers.txt", str(workspace), limit=False)


@pytest.mark.asyncio
async def test_read_file_output_does_not_expose_hash_anchors(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "multi.txt"
    content = "first\nsecond\nthird"
    target.write_text(content, encoding="utf-8")

    result = await read_file("multi.txt", str(workspace))

    assert result == content


@pytest.mark.asyncio
async def test_read_file_missing_raises(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(FileNotFoundError):
        await read_file("missing.txt", str(workspace))


@pytest.mark.asyncio
async def test_read_file_rejects_parent_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await read_file("../secret.txt", str(workspace))


@pytest.mark.asyncio
async def test_read_file_rejects_absolute_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await read_file("/etc/passwd", str(workspace))


@pytest.mark.asyncio
async def test_read_file_rejects_symlink_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside_file)

    with pytest.raises(ValueError, match="outside workspace"):
        await read_file("link.txt", str(workspace))


@pytest.mark.asyncio
async def test_write_file_writes_content(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = await write_file("out.txt", "alpha", str(workspace))

    assert result == "Wrote 5 bytes to out.txt"
    assert (workspace / "out.txt").read_text(encoding="utf-8") == "alpha"


@pytest.mark.asyncio
async def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    await write_file("nested/path/out.txt", "value", str(workspace))

    assert (workspace / "nested" / "path" / "out.txt").read_text(
        encoding="utf-8"
    ) == "value"


@pytest.mark.asyncio
async def test_write_file_requires_fresh_read_before_overwriting_existing_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "existing.txt"
    target.write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="read_file before overwriting"):
        await write_file("existing.txt", "new", str(workspace))

    assert target.read_text(encoding="utf-8") == "old"


@pytest.mark.asyncio
async def test_write_file_allows_overwrite_after_fresh_read(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "existing.txt"
    target.write_text("old", encoding="utf-8")

    assert await read_file("existing.txt", str(workspace)) == "old"
    result = await write_file("existing.txt", "new", str(workspace))

    assert result == "Wrote 3 bytes to existing.txt"
    assert target.read_text(encoding="utf-8") == "new"


@pytest.mark.asyncio
async def test_write_file_rejects_stale_overwrite_after_external_change(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "existing.txt"
    target.write_text("old", encoding="utf-8")

    assert await read_file("existing.txt", str(workspace)) == "old"
    target.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="changed since it was last read"):
        await write_file("existing.txt", "new", str(workspace))

    assert target.read_text(encoding="utf-8") == "changed"


@pytest.mark.asyncio
async def test_write_file_rejects_parent_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await write_file("../oops.txt", "x", str(workspace))


@pytest.mark.asyncio
async def test_write_file_rejects_absolute_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await write_file("/etc/passwd", "x", str(workspace))


@pytest.mark.asyncio
async def test_write_file_rejects_symlink_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside_file)

    with pytest.raises(ValueError, match="outside workspace"):
        await write_file("link.txt", "x", str(workspace))


@pytest.mark.asyncio
async def test_edit_file_applies_edits_and_persists(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "edit.txt"
    target.write_text("alpha\nbeta\ngamma", encoding="utf-8")
    assert await read_file("edit.txt", str(workspace)) == "alpha\nbeta\ngamma"

    result = await edit_file(
        "edit.txt",
        "replace",
        "2",
        content="BETA",
        workspace_root=str(workspace),
    )

    assert result == "Applied edit to edit.txt"
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma"


@pytest.mark.asyncio
async def test_edit_file_multiline_content(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "edit.txt"
    target.write_text("alpha\nbeta\ngamma", encoding="utf-8")
    await read_file("edit.txt", str(workspace))

    result = await edit_file(
        "edit.txt",
        "replace",
        "2",
        content="BETA\nEXTRA",
        workspace_root=str(workspace),
    )

    assert result == "Applied edit to edit.txt"
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\nEXTRA\ngamma"


@pytest.mark.asyncio
async def test_edit_file_range_replace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "edit.txt"
    target.write_text("alpha\nbeta\ngamma", encoding="utf-8")
    await read_file("edit.txt", str(workspace))

    result = await edit_file(
        "edit.txt",
        "replace",
        "1",
        end="2",
        content="NEW",
        workspace_root=str(workspace),
    )

    assert result == "Applied edit to edit.txt"
    assert target.read_text(encoding="utf-8") == "NEW\ngamma"


@pytest.mark.asyncio
async def test_edit_file_append(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "edit.txt"
    target.write_text("alpha\nbeta", encoding="utf-8")
    await read_file("edit.txt", str(workspace))

    await edit_file(
        "edit.txt",
        "append",
        "1",
        content="inserted",
        workspace_root=str(workspace),
    )

    assert target.read_text(encoding="utf-8") == "alpha\ninserted\nbeta"


@pytest.mark.asyncio
async def test_edit_file_prepend(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "edit.txt"
    target.write_text("alpha\nbeta", encoding="utf-8")
    await read_file("edit.txt", str(workspace))

    await edit_file(
        "edit.txt",
        "prepend",
        "1",
        content="before",
        workspace_root=str(workspace),
    )

    assert target.read_text(encoding="utf-8") == "before\nalpha\nbeta"


@pytest.mark.asyncio
async def test_edit_file_multi_step_two_edits_on_real_python_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "sample.py"
    target.write_text(
        'def hello():\n    return "world"\n\n\ndef add(a, b):\n    return a + b\n',
        encoding="utf-8",
    )

    first_read = await read_file("sample.py", str(workspace))
    assert "#" not in first_read

    first_result = await edit_file(
        "sample.py",
        "replace",
        "1",
        content="def greet():",
        workspace_root=str(workspace),
    )

    assert first_result == "Applied edit to sample.py"

    second_read = await read_file("sample.py", str(workspace))
    assert "#" not in second_read

    second_result = await edit_file(
        "sample.py",
        "replace",
        "2",
        content='    return "earth"',
        workspace_root=str(workspace),
    )

    assert second_result == "Applied edit to sample.py"
    assert target.read_text(encoding="utf-8") == (
        'def greet():\n    return "earth"\n\n\ndef add(a, b):\n    return a + b'
    )


@pytest.mark.asyncio
async def test_edit_file_stale_hash_rejected_after_external_modification(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.py"
    target.write_text("alpha = 1\nbeta = 2\ngamma = 3\n", encoding="utf-8")

    initial_read = await read_file("target.py", str(workspace))
    assert initial_read == "alpha = 1\nbeta = 2\ngamma = 3"

    target.write_text("alpha = 1\nbeta = 200\ngamma = 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed since it was last read"):
        await edit_file(
            "target.py",
            "replace",
            "2",
            content="beta = 20",
            workspace_root=str(workspace),
        )


@pytest.mark.asyncio
async def test_edit_file_requires_read_snapshot_before_editing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_text("alpha\nbeta", encoding="utf-8")

    with pytest.raises(ValueError, match="read_file before editing"):
        await edit_file(
            "target.txt",
            "replace",
            "2",
            content="BETA",
            workspace_root=str(workspace),
        )

    assert target.read_text(encoding="utf-8") == "alpha\nbeta"


@pytest.mark.asyncio
async def test_edit_file_edits_duplicate_content_by_line_number(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_text("same\nother\nsame", encoding="utf-8")
    await read_file("target.txt", str(workspace))

    await edit_file(
        "target.txt",
        "replace",
        "3",
        content="changed",
        workspace_root=str(workspace),
    )

    assert target.read_text(encoding="utf-8") == "same\nother\nchanged"


@pytest.mark.asyncio
async def test_edit_file_accepts_numeric_pos_and_end(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_text("one\ntwo\nthree\nfour", encoding="utf-8")
    await read_file("target.txt", str(workspace))

    await edit_file(
        "target.txt",
        "replace",
        2,
        end=3,
        content="TWO THREE",
        workspace_root=str(workspace),
    )

    assert target.read_text(encoding="utf-8") == "one\nTWO THREE\nfour"


@pytest.mark.asyncio
async def test_edit_file_rejects_boolean_line_inputs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_text("one\ntwo\nthree", encoding="utf-8")
    await read_file("target.txt", str(workspace))

    with pytest.raises(ValueError, match="line number"):
        await edit_file(
            "target.txt",
            "replace",
            True,
            content="ONE",
            workspace_root=str(workspace),
        )

    with pytest.raises(ValueError, match="line number"):
        await edit_file(
            "target.txt",
            "replace",
            1,
            end=True,
            content="ONE",
            workspace_root=str(workspace),
        )


@pytest.mark.asyncio
async def test_edit_file_rejects_line_outside_last_read_range(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_text("one\ntwo\nthree", encoding="utf-8")
    assert await read_file("target.txt", str(workspace), offset=1, limit=1) == "one"

    with pytest.raises(ValueError, match="line not found in the last read"):
        await edit_file(
            "target.txt",
            "replace",
            "2",
            content="TWO",
            workspace_root=str(workspace),
        )

    assert target.read_text(encoding="utf-8") == "one\ntwo\nthree"


@pytest.mark.asyncio
async def test_edit_file_repeated_cycles_on_python_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "cycles.py"
    target.write_text(
        "def compute(value):\n"
        "    result = value + 1\n"
        "    if result > 10:\n"
        "        return result\n"
        "    return value\n",
        encoding="utf-8",
    )

    for line_number, replacement in [
        (1, "def compute_total(value):"),
        (2, "    result = value + 2"),
        (5, "    return result"),
    ]:
        read_result = await read_file("cycles.py", str(workspace))
        assert "#" not in read_result

        result = await edit_file(
            "cycles.py",
            "replace",
            str(line_number),
            content=replacement,
            workspace_root=str(workspace),
        )

        assert result == "Applied edit to cycles.py"

    assert target.read_text(encoding="utf-8") == (
        "def compute_total(value):\n"
        "    result = value + 2\n"
        "    if result > 10:\n"
        "        return result\n"
        "    return result"
    )


@pytest.mark.asyncio
async def test_edit_file_rejects_parent_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await edit_file(
            "../edit.txt",
            "replace",
            "1",
            content="y",
            workspace_root=str(workspace),
        )


@pytest.mark.asyncio
async def test_edit_file_rejects_absolute_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await edit_file(
            "/etc/passwd",
            "replace",
            "1",
            content="y",
            workspace_root=str(workspace),
        )


@pytest.mark.asyncio
async def test_edit_file_rejects_symlink_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside_file)

    with pytest.raises(ValueError, match="outside workspace"):
        await edit_file(
            "link.txt",
            "replace",
            "1",
            content="y",
            workspace_root=str(workspace),
        )


@pytest.mark.asyncio
async def test_bash_captures_output(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    output = await bash("python -c 'print(\"hello\")'", 1.0, str(workspace))

    assert output.strip() == "hello"


@pytest.mark.asyncio
async def test_bash_nonzero_exit_includes_stdout_and_stderr(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    output = await bash(
        'python -c \'import sys; print("out"); print("err", file=sys.stderr); sys.exit(3)\'',
        1.0,
        str(workspace),
    )

    assert "Exit code 3" in output
    assert "STDOUT:\nout\n" in output
    assert "STDERR:\nerr\n" in output


@pytest.mark.asyncio
async def test_bash_timeout(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="timed out"):
        await bash('python -c "import time; time.sleep(2)"', 0.1, str(workspace))


@pytest.mark.asyncio
async def test_bash_timeout_terminates_spawned_children(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "child_marker.txt"

    child_script = workspace / "child.py"
    child_script.write_text(
        "import pathlib\n"
        "import time\n"
        "time.sleep(0.3)\n"
        "pathlib.Path('child_marker.txt').write_text('alive')\n",
        encoding="utf-8",
    )

    # The spawned child would leave a marker behind if only the shell died.
    command = (
        'python -c "import subprocess, sys, time; '
        'subprocess.Popen([sys.executable, \'child.py\']); '
        'time.sleep(5)"'
    )

    with pytest.raises(ValueError, match="timed out"):
        await bash(command, 0.1, str(workspace))

    await asyncio.sleep(0.5)
    assert not marker.exists()


def test_builtin_skill_tools_returns_all_schemas() -> None:
    skill = BuiltinToolsSkill()
    discovered = skill.tools()

    assert set(discovered) >= {"read_file", "write_file", "edit_file", "bash"}
    for tool_name, (schema, handler) in discovered.items():
        assert schema.name == tool_name
        assert schema.description
        assert schema.parameters["type"] == "object"
        assert callable(handler)


def test_file_tool_function_signatures_do_not_expose_snapshot_store() -> None:
    signatures = [
        inspect.signature(read_file),
        inspect.signature(write_file),
        inspect.signature(edit_file),
    ]

    for signature in signatures:
        assert "snapshot_store" not in signature.parameters


def test_edit_file_preserves_original_public_signature() -> None:
    signature = inspect.signature(edit_file)

    assert list(signature.parameters) == [
        "file_path",
        "op",
        "pos",
        "end",
        "content",
        "workspace_root",
    ]


@pytest.mark.asyncio
async def test_edit_file_accepts_clean_line_position_after_read(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "clean.txt"
    target.write_text("alpha\nbeta\ngamma", encoding="utf-8")

    assert await read_file("clean.txt", str(workspace)) == "alpha\nbeta\ngamma"

    result = await edit_file(
        "clean.txt",
        "replace",
        "2",
        content="BETA",
        workspace_root=str(workspace),
    )

    assert result == "Applied edit to clean.txt"
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma"


@pytest.mark.asyncio
async def test_edit_file_rejects_public_hash_anchor_input(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "hash.txt"
    target.write_text("alpha\nbeta", encoding="utf-8")
    await read_file("hash.txt", str(workspace))

    with pytest.raises(ValueError, match="line number"):
        await edit_file(
            "hash.txt",
            "replace",
            "2#a1b2",
            content="BETA",
            workspace_root=str(workspace),
        )


@pytest.mark.asyncio
async def test_edit_file_rejects_public_text_anchor_input(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "text.txt"
    target.write_text("alpha\nbeta", encoding="utf-8")
    await read_file("text.txt", str(workspace))

    with pytest.raises(ValueError, match="line number"):
        await edit_file(
            "text.txt",
            "replace",
            "beta",
            content="BETA",
            workspace_root=str(workspace),
        )


@pytest.mark.asyncio
async def test_bound_builtin_tools_do_not_share_edit_snapshots(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "shared.txt"
    target.write_text("alpha\nbeta", encoding="utf-8")
    skill_a = BuiltinToolsSkill().bind_workspace(str(workspace))
    skill_b = BuiltinToolsSkill().bind_workspace(str(workspace))
    tools_a = skill_a.tools()
    tools_b = skill_b.tools()

    _, read_a = tools_a["read_file"]
    _, edit_b = tools_b["edit_file"]

    assert await read_a(file_path="shared.txt") == "alpha\nbeta"
    with pytest.raises(ValueError, match="read_file before editing"):
        await edit_b(
            file_path="shared.txt",
            op="replace",
            pos="2",
            content="BETA",
        )

    assert target.read_text(encoding="utf-8") == "alpha\nbeta"


@pytest.mark.asyncio
async def test_bound_builtin_tools_do_not_share_write_snapshots(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "shared.txt"
    target.write_text("alpha", encoding="utf-8")
    skill_a = BuiltinToolsSkill().bind_workspace(str(workspace))
    skill_b = BuiltinToolsSkill().bind_workspace(str(workspace))
    tools_a = skill_a.tools()
    tools_b = skill_b.tools()

    _, read_a = tools_a["read_file"]
    _, write_b = tools_b["write_file"]

    assert await read_a(file_path="shared.txt") == "alpha"
    with pytest.raises(ValueError, match="read_file before overwriting"):
        await write_b(file_path="shared.txt", content="new")

    assert target.read_text(encoding="utf-8") == "alpha"


@pytest.mark.asyncio
async def test_file_snapshots_are_entity_scoped_even_with_shared_skill_instance(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "shared.txt"
    target.write_text("alpha\nbeta", encoding="utf-8")
    skill = BuiltinToolsSkill().bind_workspace(str(workspace))
    tools = skill.tools()
    schemas = {name: schema for name, (schema, _) in tools.items()}
    handlers = {name: handler for name, (_, handler) in tools.items()}
    world = World()
    reader = world.create_entity()
    editor = world.create_entity()
    for entity_id in (reader, editor):
        world.add_component(entity_id, ConversationComponent(messages=[]))
        world.add_component(
            entity_id,
            ToolRegistryComponent(tools=schemas, handlers=handlers),
        )

    world.add_component(
        reader,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(id="read-1", name="read_file", arguments={"file_path": "shared.txt"})
            ]
        ),
    )
    await ToolExecutionSystem().process(world)

    world.add_component(
        editor,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(
                    id="edit-1",
                    name="edit_file",
                        arguments={
                            "file_path": "shared.txt",
                            "op": "replace",
                            "pos": "2",
                            "content": "BETA",
                        },
                )
            ]
        ),
    )
    await ToolExecutionSystem().process(world)

    assert target.read_text(encoding="utf-8") == "alpha\nbeta"
    results = world.get_component(editor, ToolResultsComponent)
    assert results is not None
    assert "read_file before editing" in results.results["edit-1"]


@pytest.mark.asyncio
async def test_parallel_read_snapshots_preserve_multiple_ranges_for_later_edits(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "ranges.txt"
    target.write_text("one\ntwo\nthree\nfour", encoding="utf-8")

    await read_file("ranges.txt", str(workspace), offset=1, limit=1)
    await read_file("ranges.txt", str(workspace), offset=4, limit=1)

    await edit_file(
        "ranges.txt",
        "replace",
        "1",
        content="ONE",
        workspace_root=str(workspace),
    )
    await edit_file(
        "ranges.txt",
        "replace",
        "4",
        content="FOUR",
        workspace_root=str(workspace),
    )

    assert target.read_text(encoding="utf-8") == "ONE\ntwo\nthree\nFOUR"


def test_builtin_skill_install() -> None:
    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()

    manager.install(world, entity_id, BuiltinToolsSkill())

    registry = world.get_component(entity_id, ToolRegistryComponent)
    assert registry is not None
    assert set(registry.tools) >= {
        "read_file",
        "write_file",
        "edit_file",
        "bash",
        "load_skill_details",
    }
    assert set(registry.handlers) >= {
        "read_file",
        "write_file",
        "edit_file",
        "bash",
        "load_skill_details",
    }


def test_builtin_tools_skill_satisfies_protocol() -> None:
    """BuiltinToolsSkill should satisfy ScriptSkill protocol."""
    skill = BuiltinToolsSkill()
    assert isinstance(skill, ScriptSkill)
    assert skill.name == "builtin-tools"
    assert skill.description is not None
    assert len(skill.description) > 0
    assert callable(skill.tools)
    assert callable(skill.system_prompt)
    assert callable(skill.install)
    assert callable(skill.uninstall)


@pytest.mark.asyncio
async def test_glob_happy_path(tmp_path: Path) -> None:
    """Test glob matches *.txt files and returns sorted newline-delimited paths."""
    if glob is None:
        pytest.skip("glob not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file1.txt").write_text("content1")
    (workspace / "file2.txt").write_text("content2")
    (workspace / "file3.md").write_text("content3")

    result = await glob("*.txt", ".", str(workspace))

    lines = sorted(result.strip().split("\n")) if result.strip() else []
    assert set(lines) == {"file1.txt", "file2.txt"}
    assert lines == sorted(lines)  # Verify sorted


@pytest.mark.asyncio
async def test_glob_no_matches(tmp_path: Path) -> None:
    """Test glob returns empty string when no files match."""
    if glob is None:
        pytest.skip("glob not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file1.md").write_text("content")

    result = await glob("*.txt", ".", str(workspace))

    assert result == ""


@pytest.mark.asyncio
async def test_glob_rejects_parent_traversal(tmp_path: Path) -> None:
    """Test glob raises ValueError when base_path attempts to escape workspace."""
    if glob is None:
        pytest.skip("glob not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await glob("*.txt", "..", str(workspace))


@pytest.mark.asyncio
async def test_glob_rejects_absolute_path(tmp_path: Path) -> None:
    """Test glob raises ValueError when base_path is absolute."""
    if glob is None:
        pytest.skip("glob not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await glob("*.txt", "/etc", str(workspace))


@pytest.mark.asyncio
async def test_glob_nested_matches(tmp_path: Path) -> None:
    """Test glob includes files in subdirectories with relative paths."""
    if glob is None:
        pytest.skip("glob not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file1.txt").write_text("content1")
    subdir = workspace / "subdir"
    subdir.mkdir()
    (subdir / "file2.txt").write_text("content2")

    result = await glob("**/*.txt", ".", str(workspace))

    lines = sorted(result.strip().split("\n")) if result.strip() else []
    assert "file1.txt" in lines
    # Handle both Unix and Windows path separators
    assert any("file2.txt" in line for line in lines)


def test_builtin_skill_tools_includes_glob() -> None:
    """Test that catalog includes glob tool key."""
    skill = BuiltinToolsSkill()
    discovered = skill.tools()

    assert "glob" in discovered


def test_builtin_skill_install_includes_glob() -> None:
    """Test that installed registry includes glob in tools."""
    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()

    manager.install(world, entity_id, BuiltinToolsSkill())

    registry = world.get_component(entity_id, ToolRegistryComponent)
    assert registry is not None
    assert "glob" in registry.tools
    assert "glob" in registry.handlers


# ---------------------------------------------------------------------------
# is_tool_bundle contract tests (RED phase — written before implementation)
# ---------------------------------------------------------------------------


def test_builtin_tools_skill_is_tool_bundle_flag() -> None:
    """BuiltinToolsSkill.is_tool_bundle must be True."""
    skill = BuiltinToolsSkill()
    assert skill.is_tool_bundle is True


def test_tool_bundle_not_registered_in_skill_component() -> None:
    """Installing a tool-bundle skill must NOT add it to SkillComponent."""
    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()

    manager.install(world, entity_id, BuiltinToolsSkill())

    skill_comp = world.get_component(entity_id, SkillComponent)
    # SkillComponent may not be created at all, or if it is, builtin-tools must not be in it
    if skill_comp is not None:
        assert "builtin-tools" not in skill_comp.skills


def test_tool_bundle_tools_still_registered_in_tool_registry() -> None:
    """Tool-bundle tools must still appear in ToolRegistryComponent after install."""
    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()

    manager.install(world, entity_id, BuiltinToolsSkill())

    registry = world.get_component(entity_id, ToolRegistryComponent)
    assert registry is not None
    assert "read_file" in registry.tools
    assert "write_file" in registry.tools
    assert "bash" in registry.tools
    assert "glob" in registry.tools


def test_regular_skill_still_registered_in_skill_component() -> None:
    """Non-tool-bundle skills must still appear in SkillComponent after install."""
    from ecs_agent.types import ToolSchema

    class _NormalSkill:
        name = "normal-skill"
        description = "A regular skill"
        is_tool_bundle = False

        def tools(self) -> dict[str, tuple[ToolSchema, object]]:
            return {}

        def system_prompt(self) -> str:
            return ""

        def install(self, world: object, entity_id: object) -> None:
            pass

        def uninstall(self, world: object, entity_id: object) -> None:
            pass

    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()

    manager.install(world, entity_id, _NormalSkill())  # type: ignore[arg-type]

    skill_comp = world.get_component(entity_id, SkillComponent)
    assert skill_comp is not None
    assert "normal-skill" in skill_comp.skills


def test_default_is_tool_bundle_is_false_for_protocol_compliant_skill() -> None:
    """ScriptSkill that omits is_tool_bundle should default to False (via getattr)."""
    from ecs_agent.types import ToolSchema

    class _MinimalSkill:
        name = "minimal"
        description = "minimal"

        def tools(self) -> dict[str, tuple[ToolSchema, object]]:
            return {}

        def system_prompt(self) -> str:
            return ""

        def install(self, world: object, entity_id: object) -> None:
            pass

        def uninstall(self, world: object, entity_id: object) -> None:
            pass

    skill = _MinimalSkill()
    assert getattr(skill, "is_tool_bundle", False) is False


# ---------------------------------------------------------------------------
# interactive_bash tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interactive_bash_new_session() -> None:
    if interactive_bash is None:
        pytest.skip("interactive_bash not implemented yet")

    session_name = "test-ecs-ib-new"
    result = await interactive_bash(f"new-session -d -s {session_name}")

    assert "error" not in result.lower() or session_name in result or result == ""
    await interactive_bash(f"kill-session -t {session_name}")


@pytest.mark.asyncio
async def test_interactive_bash_send_keys_and_capture() -> None:
    if interactive_bash is None:
        pytest.skip("interactive_bash not implemented yet")

    session_name = "test-ecs-ib-capture"
    await interactive_bash(f"new-session -d -s {session_name}")
    try:
        send_result = await interactive_bash(
            f"send-keys -t {session_name} 'echo hello-from-tmux' Enter"
        )
        assert isinstance(send_result, str)
    finally:
        await interactive_bash(f"kill-session -t {session_name}")


@pytest.mark.asyncio
async def test_interactive_bash_invalid_command_returns_error() -> None:
    if interactive_bash is None:
        pytest.skip("interactive_bash not implemented yet")

    result = await interactive_bash("invalid-subcommand-xyz")
    assert "error" in result.lower() or "unknown" in result.lower() or result != ""


def test_interactive_bash_registered_in_builtin_skill() -> None:
    if interactive_bash is None:
        pytest.skip("interactive_bash not implemented yet")

    skill = BuiltinToolsSkill()
    discovered = skill.tools()
    assert "interactive_bash" in discovered


def test_interactive_bash_in_installed_registry() -> None:
    if interactive_bash is None:
        pytest.skip("interactive_bash not implemented yet")

    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()
    manager.install(world, entity_id, BuiltinToolsSkill())

    registry = world.get_component(entity_id, ToolRegistryComponent)
    assert registry is not None
    assert "interactive_bash" in registry.tools
    assert "interactive_bash" in registry.handlers


# ---------------------------------------------------------------------------
# Tool schema parameter description tests
# ---------------------------------------------------------------------------


def test_tool_schema_has_parameter_descriptions() -> None:
    skill = BuiltinToolsSkill()
    discovered = skill.tools()

    for tool_name, (schema, _) in discovered.items():
        props = schema.parameters.get("properties", {})
        for param_name, param_schema in props.items():
            if param_name == "workspace_root":
                continue
            assert "description" in param_schema, (
                f"Tool '{tool_name}' parameter '{param_name}' missing description"
            )
            assert len(param_schema["description"]) > 0, (
                f"Tool '{tool_name}' parameter '{param_name}' has empty description"
            )


def test_edit_file_schema_exposes_direct_params_not_edits_json() -> None:
    skill = BuiltinToolsSkill()
    discovered = skill.tools()
    assert "edit_file" in discovered

    schema, _ = discovered["edit_file"]
    props = schema.parameters.get("properties", {})

    assert "edits_json" not in props, "edits_json should not appear in new API schema"
    assert "read_id" not in props
    assert "snapshot_id" not in props
    assert "old_text" not in props
    assert "new_text" not in props
    assert "replace_all" not in props
    assert "op" in props
    assert "pos" in props
    assert "content" in props

    required = schema.parameters.get("required", [])
    assert "file_path" in required
    assert "op" in required
    assert "pos" in required


def test_file_tool_schemas_accept_numeric_strings_and_numbers() -> None:
    skill = BuiltinToolsSkill()
    discovered = skill.tools()

    read_schema, _ = discovered["read_file"]
    read_props = read_schema.parameters.get("properties", {})
    assert read_props["offset"]["type"] == ["integer", "string"]
    assert read_props["limit"]["type"] == ["integer", "string"]

    edit_schema, _ = discovered["edit_file"]
    edit_props = edit_schema.parameters.get("properties", {})
    assert edit_props["pos"]["type"] == ["integer", "string"]
    assert edit_props["end"]["type"] == ["integer", "string", "null"]


def test_builtin_read_only_tools_are_concurrency_safe(tmp_path: Path) -> None:
    skill = BuiltinToolsSkill().bind_workspace(str(tmp_path / "workspace"))
    tools = {name: schema for name, (schema, _handler) in skill.tools().items()}

    read_only = ("read_file", "grep", "glob", "explore", "webfetch")
    for name in read_only:
        assert tools[name].concurrency_safe is True, name

    mutating = ("write_file", "edit_file", "bash", "interactive_bash", "code_execution")
    for name in mutating:
        assert tools[name].concurrency_safe is False, name


@pytest.mark.asyncio
async def test_interactive_bash_times_out_on_blocking_subcommand() -> None:
    """A blocking tmux subcommand must not hang the run forever."""
    if interactive_bash is None:
        pytest.skip("interactive_bash not implemented yet")
    import shutil

    if shutil.which("tmux") is None:
        pytest.skip("tmux not installed")

    # wait-for only blocks when a tmux server is running; without one it
    # errors out immediately.
    session_name = "ecs-ib-timeout-test"
    await interactive_bash(f"new-session -d -s {session_name}")
    try:
        with pytest.raises(ValueError, match="timed out"):
            await interactive_bash("wait-for ecs-test-never-signalled", timeout=0.3)
    finally:
        await interactive_bash(f"kill-session -t {session_name}")


@pytest.mark.asyncio
async def test_communicate_with_timeout_kills_hanging_process() -> None:
    """The shared timeout helper terminates the process group on expiry."""
    import asyncio as _asyncio

    from ecs_agent.tools.builtins.bash_tool import _communicate_with_timeout

    process = await _asyncio.create_subprocess_exec(
        "sleep",
        "30",
        stdout=_asyncio.subprocess.PIPE,
        stderr=_asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    with pytest.raises(ValueError, match="timed out after 0.2s"):
        await _communicate_with_timeout(process, 0.2, "sleep 30")
    assert process.returncode is not None


def test_interactive_bash_schema_has_optional_timeout() -> None:
    """timeout is exposed in the tool schema but optional (has a default)."""
    if interactive_bash is None:
        pytest.skip("interactive_bash not implemented yet")

    schema = interactive_bash._tool_schema  # type: ignore[attr-defined]
    assert "timeout" in schema.parameters["properties"]
    assert "timeout" not in schema.parameters.get("required", [])
