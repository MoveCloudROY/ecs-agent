"""TDD tests for new built-in tools: grep, read_file range, explore, webfetch, code_execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# grep tool tests
# ---------------------------------------------------------------------------

try:
    from ecs_agent.tools.builtins.grep_tool import grep
except ImportError:
    grep = None  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_grep_finds_matching_lines(tmp_path: Path) -> None:
    if grep is None:
        pytest.skip("grep not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "code.py").write_text(
        "def hello():\n    return 'world'\n\ndef foo():\n    pass\n",
        encoding="utf-8",
    )

    result = await grep("def ", "code.py", str(workspace))

    assert "1: def hello():" in result
    assert "4: def foo():" in result
    assert "return" not in result


@pytest.mark.asyncio
async def test_grep_no_matches_returns_empty(tmp_path: Path) -> None:
    if grep is None:
        pytest.skip("grep not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sample.txt").write_text("hello\nworld\n", encoding="utf-8")

    result = await grep("xyz_nonexistent", "sample.txt", str(workspace))

    assert result == ""


@pytest.mark.asyncio
async def test_grep_regex_pattern(tmp_path: Path) -> None:
    if grep is None:
        pytest.skip("grep not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "data.txt").write_text(
        "error: bad input\ninfo: all good\nwarning: check this\nerror: disk full\n",
        encoding="utf-8",
    )

    result = await grep(r"^error:", "data.txt", str(workspace))

    lines = result.strip().split("\n")
    assert len(lines) == 2
    assert "1: error: bad input" in result
    assert "4: error: disk full" in result
    assert "info" not in result


@pytest.mark.asyncio
async def test_grep_rejects_path_traversal(tmp_path: Path) -> None:
    if grep is None:
        pytest.skip("grep not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await grep("pattern", "../secret.txt", str(workspace))


@pytest.mark.asyncio
async def test_grep_rejects_absolute_path(tmp_path: Path) -> None:
    if grep is None:
        pytest.skip("grep not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await grep("pattern", "/etc/passwd", str(workspace))


@pytest.mark.asyncio
async def test_grep_missing_file_raises(tmp_path: Path) -> None:
    if grep is None:
        pytest.skip("grep not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(FileNotFoundError):
        await grep("pattern", "nonexistent.txt", str(workspace))


@pytest.mark.asyncio
async def test_grep_case_sensitive_by_default(tmp_path: Path) -> None:
    if grep is None:
        pytest.skip("grep not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "mixed.txt").write_text("Hello\nhello\nHELLO\n", encoding="utf-8")

    result = await grep("hello", "mixed.txt", str(workspace))

    assert "2: hello" in result
    assert "1: Hello" not in result
    assert "3: HELLO" not in result


@pytest.mark.asyncio
async def test_grep_python_fallback_when_rg_unavailable(tmp_path: Path) -> None:
    """When _RG_BIN is None the Python re fallback must produce identical output."""
    if grep is None:
        pytest.skip("grep not implemented yet")
    import ecs_agent.tools.builtins.grep_tool as grep_mod

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "code.py").write_text(
        "def alpha():\n    pass\n\ndef beta():\n    pass\n", encoding="utf-8"
    )

    with patch.object(grep_mod, "_RG_BIN", None):
        result = await grep("def ", "code.py", str(workspace))

    assert "1: def alpha():" in result
    assert "4: def beta():" in result


@pytest.mark.asyncio
async def test_grep_rg_failure_falls_back_to_python(tmp_path: Path) -> None:
    """If rg subprocess raises, the Python fallback is used transparently."""
    if grep is None:
        pytest.skip("grep not implemented yet")
    import ecs_agent.tools.builtins.grep_tool as grep_mod

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "data.txt").write_text("foo\nbar\nfoo bar\n", encoding="utf-8")

    async def _failing_rg(*_a: object, **_kw: object) -> str:
        raise RuntimeError("rg crashed")

    with patch.object(grep_mod, "_RG_BIN", "/usr/bin/rg"), patch.object(
        grep_mod, "_grep_rg", _failing_rg
    ):
        result = await grep("foo", "data.txt", str(workspace))

    assert "1: foo" in result
    assert "3: foo bar" in result


# ---------------------------------------------------------------------------
# read_file range tests
# ---------------------------------------------------------------------------

try:
    from ecs_agent.tools.builtins.file_tools import read_file
    from ecs_agent.tools.builtins.edit_tool import compute_line_hash
except ImportError:
    read_file = None  # type: ignore[assignment]
    compute_line_hash = None  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_read_file_with_offset_and_limit(tmp_path: Path) -> None:
    if read_file is None:
        pytest.skip("read_file not available")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "big.txt").write_text(
        "line1\nline2\nline3\nline4\nline5\n", encoding="utf-8"
    )

    result = await read_file("big.txt", str(workspace), offset=2, limit=3)

    assert result == "line2\nline3\nline4"
    assert "#" not in result


@pytest.mark.asyncio
async def test_read_file_offset_default_is_1(tmp_path: Path) -> None:
    if read_file is None:
        pytest.skip("read_file not available")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    result_default = await read_file("file.txt", str(workspace), limit=2)
    result_explicit = await read_file("file.txt", str(workspace), offset=1, limit=2)

    assert result_default == result_explicit


@pytest.mark.asyncio
async def test_read_file_limit_0_reads_all(tmp_path: Path) -> None:
    if read_file is None:
        pytest.skip("read_file not available")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file.txt").write_text("a\nb\nc\n", encoding="utf-8")

    result = await read_file("file.txt", str(workspace), limit=0)

    lines = result.strip().split("\n")
    assert len(lines) == 3


@pytest.mark.asyncio
async def test_read_file_offset_beyond_eof_returns_empty(tmp_path: Path) -> None:
    if read_file is None:
        pytest.skip("read_file not available")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file.txt").write_text("one\ntwo\n", encoding="utf-8")

    result = await read_file("file.txt", str(workspace), offset=100, limit=10)

    assert result == ""


@pytest.mark.asyncio
async def test_read_file_schema_has_offset_and_limit_params(tmp_path: Path) -> None:
    """The read_file tool schema must expose offset and limit parameters."""
    from ecs_agent.tools.builtins import BuiltinToolsSkill

    skill = BuiltinToolsSkill()
    tools = skill.tools()
    assert "read_file" in tools
    schema, _ = tools["read_file"]
    props = schema.parameters.get("properties", {})
    assert "offset" in props, "read_file schema must have 'offset' parameter"
    assert "limit" in props, "read_file schema must have 'limit' parameter"


# ---------------------------------------------------------------------------
# explore tool tests
# ---------------------------------------------------------------------------

try:
    from ecs_agent.tools.builtins.explore_tool import explore
except ImportError:
    explore = None  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_explore_lists_directory_contents(tmp_path: Path) -> None:
    if explore is None:
        pytest.skip("explore not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file1.py").write_text("x = 1")
    (workspace / "file2.txt").write_text("hello")
    subdir = workspace / "subdir"
    subdir.mkdir()
    (subdir / "inner.py").write_text("y = 2")

    result = await explore(".", 2, str(workspace))

    assert "file1.py" in result
    assert "file2.txt" in result
    assert "subdir" in result


@pytest.mark.asyncio
async def test_explore_respects_max_depth(tmp_path: Path) -> None:
    if explore is None:
        pytest.skip("explore not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lvl1 = workspace / "lvl1"
    lvl1.mkdir()
    lvl2 = lvl1 / "lvl2"
    lvl2.mkdir()
    lvl3 = lvl2 / "lvl3"
    lvl3.mkdir()
    (lvl3 / "deep.txt").write_text("deep")

    result = await explore(".", 2, str(workspace))

    assert "lvl1" in result
    assert "lvl2" in result
    assert "deep.txt" not in result


@pytest.mark.asyncio
async def test_explore_rejects_path_traversal(tmp_path: Path) -> None:
    if explore is None:
        pytest.skip("explore not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await explore("..", 2, str(workspace))


@pytest.mark.asyncio
async def test_explore_rejects_absolute_path(tmp_path: Path) -> None:
    if explore is None:
        pytest.skip("explore not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside workspace"):
        await explore("/etc", 2, str(workspace))


@pytest.mark.asyncio
async def test_explore_empty_dir(tmp_path: Path) -> None:
    if explore is None:
        pytest.skip("explore not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = await explore(".", 2, str(workspace))

    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_explore_nonexistent_path_raises(tmp_path: Path) -> None:
    if explore is None:
        pytest.skip("explore not implemented yet")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(FileNotFoundError):
        await explore("nonexistent_dir", 2, str(workspace))


# ---------------------------------------------------------------------------
# webfetch tool tests
# ---------------------------------------------------------------------------

try:
    from ecs_agent.tools.builtins.webfetch_tool import webfetch
except ImportError:
    webfetch = None  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_webfetch_returns_response_text() -> None:
    if webfetch is None:
        pytest.skip("webfetch not implemented yet")

    mock_response = MagicMock()
    mock_response.text = "<html>Hello World</html>"
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "ecs_agent.tools.builtins.webfetch_tool.httpx.AsyncClient",
        return_value=mock_client,
    ):
        result = await webfetch("https://example.com")

    assert result == "<html>Hello World</html>"


@pytest.mark.asyncio
async def test_webfetch_raises_on_http_error() -> None:
    if webfetch is None:
        pytest.skip("webfetch not implemented yet")

    import httpx

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )
    )

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "ecs_agent.tools.builtins.webfetch_tool.httpx.AsyncClient",
        return_value=mock_client,
    ):
        with pytest.raises(Exception):
            await webfetch("https://example.com/notfound")


@pytest.mark.asyncio
async def test_webfetch_passes_timeout_to_client() -> None:
    if webfetch is None:
        pytest.skip("webfetch not implemented yet")

    mock_response = MagicMock()
    mock_response.text = "ok"
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "ecs_agent.tools.builtins.webfetch_tool.httpx.AsyncClient",
        return_value=mock_client,
    ) as MockClient:
        await webfetch("https://example.com", timeout=5.0)
        MockClient.assert_called_once()
        call_kwargs = MockClient.call_args[1]
        assert call_kwargs.get("timeout") == 5.0


# ---------------------------------------------------------------------------
# code_execution tool tests
# ---------------------------------------------------------------------------

try:
    from ecs_agent.tools.builtins.code_execution_tool import code_execution
except ImportError:
    code_execution = None  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_code_execution_python_hello_world() -> None:
    if code_execution is None:
        pytest.skip("code_execution not implemented yet")

    result = await code_execution('print("hello world")', "python")

    assert "hello world" in result


@pytest.mark.asyncio
async def test_code_execution_python_arithmetic() -> None:
    if code_execution is None:
        pytest.skip("code_execution not implemented yet")

    result = await code_execution("print(2 + 3)", "python")

    assert "5" in result


@pytest.mark.asyncio
async def test_code_execution_python_multiline() -> None:
    if code_execution is None:
        pytest.skip("code_execution not implemented yet")

    code = "x = 10\ny = 20\nprint(x + y)"
    result = await code_execution(code, "python")

    assert "30" in result


@pytest.mark.asyncio
async def test_code_execution_python_error_includes_traceback() -> None:
    if code_execution is None:
        pytest.skip("code_execution not implemented yet")

    result = await code_execution("raise ValueError('test error')", "python")

    assert "ValueError" in result or "test error" in result
    assert "Exit code" in result or "error" in result.lower()


@pytest.mark.asyncio
async def test_code_execution_unsupported_language_raises() -> None:
    if code_execution is None:
        pytest.skip("code_execution not implemented yet")

    with pytest.raises(ValueError, match="Unsupported language"):
        await code_execution("console.log('hi')", "javascript")


@pytest.mark.asyncio
async def test_code_execution_timeout_raises() -> None:
    if code_execution is None:
        pytest.skip("code_execution not implemented yet")

    with pytest.raises(ValueError, match="timed out"):
        await code_execution("import time; time.sleep(10)", "python", timeout=0.1)


# ---------------------------------------------------------------------------
# BuiltinToolsSkill registry tests for new tools
# ---------------------------------------------------------------------------


def test_builtin_skill_includes_new_tools() -> None:
    from ecs_agent.tools.builtins import BuiltinToolsSkill

    skill = BuiltinToolsSkill()
    discovered = skill.tools()

    expected = {"grep", "explore", "webfetch", "code_execution"}
    missing = expected - set(discovered)
    assert not missing, f"Missing tools in BuiltinToolsSkill: {missing}"


def test_new_tools_have_valid_schemas() -> None:
    from ecs_agent.tools.builtins import BuiltinToolsSkill

    skill = BuiltinToolsSkill()
    discovered = skill.tools()

    for tool_name in ("grep", "explore", "webfetch", "code_execution"):
        if tool_name not in discovered:
            continue
        schema, handler = discovered[tool_name]
        assert schema.name == tool_name
        assert schema.description
        assert schema.parameters["type"] == "object"
        assert callable(handler)
