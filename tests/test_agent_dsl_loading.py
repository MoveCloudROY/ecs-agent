"""Tests for JSON DSL loader."""

import json
from pathlib import Path

import pytest

from ecs_agent.dsl import AgentSpec, load_json_agents, resolve_agent_specs


def test_load_json_agents_happy_path_single_agent(tmp_path: Path) -> None:
    """Test loading a single agent from JSON."""
    json_file = tmp_path / "agents.json"
    json_file.write_text(
        json.dumps(
            {
                "researcher": {
                    "mode": "primary",
                    "model": "gpt-4",
                    "prompt": "You are a research assistant.",
                }
            }
        )
    )

    specs = load_json_agents(json_file)

    assert len(specs) == 1
    assert specs[0].name == "researcher"
    assert specs[0].mode == "primary"
    assert specs[0].model == "gpt-4"
    assert specs[0].prompt == "You are a research assistant."
    assert specs[0].tools == {}
    assert specs[0].metadata == {}


def test_load_json_agents_happy_path_multiple_agents(tmp_path: Path) -> None:
    """Test loading multiple agents from JSON."""
    json_file = tmp_path / "agents.json"
    json_file.write_text(
        json.dumps(
            {
                "main": {
                    "mode": "primary",
                    "model": "gpt-4",
                    "prompt": "Main agent",
                    "tools": {"search": True, "calculator": False},
                    "metadata": {"priority": "high"},
                },
                "helper": {
                    "mode": "subagent",
                    "model": "gpt-3.5",
                    "prompt": "Helper agent",
                },
            }
        )
    )

    specs = load_json_agents(json_file)

    assert len(specs) == 2

    # Order is dict iteration order (Python 3.7+ preserves insertion order)
    main = specs[0]
    assert main.name == "main"
    assert main.mode == "primary"
    assert main.tools == {"search": True, "calculator": False}
    assert main.metadata == {"priority": "high"}

    helper = specs[1]
    assert helper.name == "helper"
    assert helper.mode == "subagent"
    assert helper.model == "gpt-3.5"


def test_load_json_agents_empty_dict(tmp_path: Path) -> None:
    """Test loading empty JSON dict returns empty list."""
    json_file = tmp_path / "agents.json"
    json_file.write_text("{}")

    specs = load_json_agents(json_file)

    assert specs == []


def test_load_json_agents_file_not_found() -> None:
    """Test that missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Agent JSON file not found"):
        load_json_agents("/nonexistent/path/agents.json")


def test_load_json_agents_malformed_json(tmp_path: Path) -> None:
    """Test that malformed JSON raises JSONDecodeError."""
    json_file = tmp_path / "agents.json"
    json_file.write_text("{invalid json syntax")

    with pytest.raises(json.JSONDecodeError):
        load_json_agents(json_file)


def test_load_json_agents_invalid_root_type(tmp_path: Path) -> None:
    """Test that non-dict root raises ValueError."""
    json_file = tmp_path / "agents.json"
    json_file.write_text(json.dumps(["not", "a", "dict"]))

    with pytest.raises(ValueError, match="JSON root must be dict"):
        load_json_agents(json_file)


def test_load_json_agents_invalid_config_type(tmp_path: Path) -> None:
    """Test that non-dict config raises ValueError."""
    json_file = tmp_path / "agents.json"
    json_file.write_text(json.dumps({"agent1": "not a dict"}))

    with pytest.raises(ValueError, match="Agent 'agent1' config must be dict"):
        load_json_agents(json_file)


def test_load_json_agents_missing_required_field(tmp_path: Path) -> None:
    """Test that missing required field triggers validation error with source context."""
    json_file = tmp_path / "agents.json"
    json_file.write_text(
        json.dumps(
            {
                "incomplete": {
                    "mode": "primary",
                    "model": "gpt-4",
                    # Missing "prompt"
                }
            }
        )
    )

    with pytest.raises(
        ValueError, match="Missing required field.*prompt.*agents.json:incomplete"
    ):
        load_json_agents(json_file)


def test_load_json_agents_unknown_field_triggers_validation_error(
    tmp_path: Path,
) -> None:
    """Test that unknown field triggers validation error with source context."""
    json_file = tmp_path / "agents.json"
    json_file.write_text(
        json.dumps(
            {
                "badagent": {
                    "mode": "primary",
                    "model": "gpt-4",
                    "prompt": "test",
                    "unknown_field": "value",
                }
            }
        )
    )

    with pytest.raises(
        ValueError, match="Unknown field.*unknown_field.*agents.json:badagent"
    ):
        load_json_agents(json_file)


def test_load_json_agents_invalid_mode_triggers_validation_error(
    tmp_path: Path,
) -> None:
    """Test that invalid mode triggers validation error with source context."""
    json_file = tmp_path / "agents.json"
    json_file.write_text(
        json.dumps(
            {
                "badmode": {
                    "mode": "invalid",
                    "model": "gpt-4",
                    "prompt": "test",
                }
            }
        )
    )

    with pytest.raises(ValueError, match="Invalid mode 'invalid'.*agents.json:badmode"):
        load_json_agents(json_file)


def test_load_json_agents_invalid_tools_type_triggers_validation_error(
    tmp_path: Path,
) -> None:
    """Test that invalid tools type triggers validation error."""
    json_file = tmp_path / "agents.json"
    json_file.write_text(
        json.dumps(
            {
                "badtools": {
                    "mode": "primary",
                    "model": "gpt-4",
                    "prompt": "test",
                    "tools": ["not", "a", "dict"],
                }
            }
        )
    )

    with pytest.raises(
        TypeError, match="Field 'tools' must be dict.*agents.json:badtools"
    ):
        load_json_agents(json_file)


def test_load_json_agents_path_as_string(tmp_path: Path) -> None:
    """Test that path can be passed as string."""
    json_file = tmp_path / "agents.json"
    json_file.write_text(
        json.dumps(
            {
                "agent1": {
                    "mode": "primary",
                    "model": "gpt-4",
                    "prompt": "test",
                }
            }
        )
    )

    specs = load_json_agents(str(json_file))

    assert len(specs) == 1
    assert specs[0].name == "agent1"


def test_load_json_agents_preserves_all_fields(tmp_path: Path) -> None:
    """Test that all fields are preserved through normalization."""
    json_file = tmp_path / "agents.json"
    json_file.write_text(
        json.dumps(
            {
                "full_spec": {
                    "mode": "subagent",
                    "model": "claude-3",
                    "prompt": "Full spec test",
                    "tools": {"tool1": True, "tool2": False},
                    "metadata": {
                        "version": "1.0",
                        "tags": ["test", "full"],
                        "nested": {"key": "value"},
                    },
                }
            }
        )
    )

    specs = load_json_agents(json_file)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "full_spec"
    assert spec.mode == "subagent"
    assert spec.model == "claude-3"
    assert spec.prompt == "Full spec test"
    assert spec.tools == {"tool1": True, "tool2": False}
    assert spec.metadata == {
        "version": "1.0",
        "tags": ["test", "full"],
        "nested": {"key": "value"},
    }


def test_load_json_agents_name_from_dict_key_overrides_explicit_name(
    tmp_path: Path,
) -> None:
    """Test that agent name from dict key takes precedence over explicit 'name' field."""
    json_file = tmp_path / "agents.json"
    json_file.write_text(
        json.dumps(
            {
                "agent_key": {
                    "mode": "primary",
                    "model": "gpt-4",
                    "prompt": "test",
                    "name": "explicit_name_ignored",  # This will be overridden
                }
            }
        )
    )

    specs = load_json_agents(json_file)

    assert len(specs) == 1
    # Name from dict key wins
    assert specs[0].name == "agent_key"


def test_load_json_agents_returns_agentspec_instances(tmp_path: Path) -> None:
    """Test that returned objects are AgentSpec instances."""
    json_file = tmp_path / "agents.json"
    json_file.write_text(
        json.dumps(
            {
                "test": {
                    "mode": "primary",
                    "model": "gpt-4",
                    "prompt": "test",
                }
            }
        )
    )

    specs = load_json_agents(json_file)

    assert all(isinstance(spec, AgentSpec) for spec in specs)


class TestDiscoverAgentSources:
    """Test deterministic source discovery."""

    def test_discover_empty_directory(self) -> None:
        """Empty directory returns empty list."""
        import tempfile
        from ecs_agent.dsl.discovery import discover_agent_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            result = discover_agent_sources(tmpdir)
            assert result == []

    def test_discover_json_sources(self) -> None:
        """Discovers JSON files in sorted order."""
        import tempfile
        from ecs_agent.dsl.discovery import discover_agent_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create files (intentionally out of order)
            (tmppath / "zebra.json").touch()
            (tmppath / "apple.json").touch()
            (tmppath / "banana.json").touch()

            result = discover_agent_sources(tmpdir)

            # Should be sorted lexicographically
            assert len(result) == 3
            assert result[0].name == "apple.json"
            assert result[1].name == "banana.json"
            assert result[2].name == "zebra.json"

    def test_discover_markdown_sources(self) -> None:
        """Discovers Markdown files in sorted order."""
        import tempfile
        from ecs_agent.dsl.discovery import discover_agent_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create files (intentionally out of order)
            (tmppath / "zebra.md").touch()
            (tmppath / "apple.md").touch()
            (tmppath / "banana.md").touch()

            result = discover_agent_sources(tmpdir)

            # Should be sorted lexicographically
            assert len(result) == 3
            assert result[0].name == "apple.md"
            assert result[1].name == "banana.md"
            assert result[2].name == "zebra.md"

    def test_discover_mixed_json_and_markdown(self) -> None:
        """Discovers both JSON and Markdown files, sorted together."""
        import tempfile
        from ecs_agent.dsl.discovery import discover_agent_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create mixed files (intentionally out of order)
            (tmppath / "zebra.json").touch()
            (tmppath / "apple.md").touch()
            (tmppath / "banana.json").touch()
            (tmppath / "delta.md").touch()

            result = discover_agent_sources(tmpdir)

            # Should all be included and sorted lexicographically
            assert len(result) == 4
            names = [p.name for p in result]
            assert names == ["apple.md", "banana.json", "delta.md", "zebra.json"]

    def test_discover_ignores_other_extensions(self) -> None:
        """Discovers only .json and .md files, ignores other extensions."""
        import tempfile
        from ecs_agent.dsl.discovery import discover_agent_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "agent.json").touch()
            (tmppath / "config.md").touch()
            (tmppath / "readme.txt").touch()
            (tmppath / "data.yaml").touch()

            result = discover_agent_sources(tmpdir)

            # Should only include JSON and Markdown
            assert len(result) == 2
            names = [p.name for p in result]
            assert "agent.json" in names
            assert "config.md" in names
            assert "readme.txt" not in names
            assert "data.yaml" not in names

    def test_discover_deterministic_order_repeated_calls(self) -> None:
        """Repeated discovery calls return identical order."""
        import tempfile
        from ecs_agent.dsl.discovery import discover_agent_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create multiple files
            files = ["agent1.json", "agent2.md", "agent3.json", "config.md"]
            for fname in files:
                (tmppath / fname).touch()

            # Multiple calls should return same order
            first_call = discover_agent_sources(tmpdir)
            second_call = discover_agent_sources(tmpdir)
            third_call = discover_agent_sources(tmpdir)

            assert first_call == second_call
            assert second_call == third_call
            # Verify specific order (lexicographic)
            names = [p.name for p in first_call]
            assert names == ["agent1.json", "agent2.md", "agent3.json", "config.md"]

    def test_discover_nonexistent_directory_raises(self) -> None:
        """Raises FileNotFoundError for nonexistent directory."""
        from ecs_agent.dsl.discovery import discover_agent_sources

        with pytest.raises(FileNotFoundError):
            discover_agent_sources("/nonexistent/path/to/agents")

    def test_discover_file_not_directory_raises(self) -> None:
        """Raises FileNotFoundError when path is a file, not directory."""
        import tempfile
        from ecs_agent.dsl.discovery import discover_agent_sources

        with tempfile.NamedTemporaryFile() as tmp:
            with pytest.raises(FileNotFoundError):
                discover_agent_sources(tmp.name)

    def test_discover_with_path_object(self) -> None:
        """Works with Path objects as input."""
        import tempfile
        from ecs_agent.dsl.discovery import discover_agent_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "agent.json").touch()
            (tmppath / "config.md").touch()

            # Pass Path object instead of string
            result = discover_agent_sources(tmppath)

            assert len(result) == 2
            names = {p.name for p in result}
            assert names == {"agent.json", "config.md"}

    def test_discover_with_string_path(self) -> None:
        """Works with string paths as input."""
        import tempfile
        from ecs_agent.dsl.discovery import discover_agent_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "agent.json").touch()

            # Pass string instead of Path object
            result = discover_agent_sources(str(tmppath))

            assert len(result) == 1
            assert result[0].name == "agent.json"

    def test_discover_last_one_wins_ordering(self) -> None:
        """Ordering ensures last-one-wins semantics for name collisions.

        When multiple sources define the same agent, the one with the
        lexicographically last filename wins. This test verifies the
        ordering enables that behavior.
        """
        import tempfile
        from ecs_agent.dsl.discovery import discover_agent_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create multiple sources (ordering matters)
            (tmppath / "agents_v1.json").touch()
            (tmppath / "agents_v2.json").touch()
            (tmppath / "agents_override.md").touch()

            result = discover_agent_sources(tmpdir)

            # Last in order should be agents_v2.json (lexicographically last)
            assert result[-1].name == "agents_v2.json"
            # This ordering enables last-one-wins for collision resolution

    def test_discover_returns_path_objects(self) -> None:
        """Returns list of Path objects, not strings."""
        import tempfile
        from ecs_agent.dsl.discovery import discover_agent_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "agent.json").touch()

            result = discover_agent_sources(tmpdir)

            assert len(result) == 1
            assert isinstance(result[0], Path)

    def test_discover_handles_special_characters_in_filenames(self) -> None:
        """Handles filenames with special characters."""
        import tempfile
        from ecs_agent.dsl.discovery import discover_agent_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create files with special characters (that are valid in filenames)
            (tmppath / "agent-v1.json").touch()
            (tmppath / "agent_v2.json").touch()
            (tmppath / "agent.v3.md").touch()

            result = discover_agent_sources(tmpdir)

            # Should all be discovered and sorted
            assert len(result) == 3
            names = [p.name for p in result]
            # Verify lexicographic sort with special chars
            assert names == sorted(names)




# Conflict resolution tests
def test_resolve_agent_specs_last_one_wins_single_duplicate(
) -> None:
    """Test last-one-wins for duplicate agent names."""
    specs = [
        AgentSpec(
            mode="primary",
            model="gpt-3.5",
            prompt="First version",
            name="researcher",
        ),
        AgentSpec(
            mode="primary",
            model="gpt-4",
            prompt="Second version (winner)",
            name="researcher",
        ),
    ]

    resolved = resolve_agent_specs(specs)

    assert len(resolved) == 1
    assert resolved["researcher"].model == "gpt-4"
    assert resolved["researcher"].prompt == "Second version (winner)"


def test_resolve_agent_specs_last_one_wins_multiple_duplicates(
) -> None:
    """Test last-one-wins with multiple duplicate names."""
    specs = [
        AgentSpec(mode="primary", model="m1", prompt="v1", name="agent_a"),
        AgentSpec(mode="primary", model="m2", prompt="v2", name="agent_b"),
        AgentSpec(mode="primary", model="m3", prompt="v3", name="agent_a"),
        AgentSpec(mode="primary", model="m4", prompt="v4", name="agent_b"),
        AgentSpec(mode="primary", model="m5", prompt="v5 (final)", name="agent_a"),
    ]

    resolved = resolve_agent_specs(specs)

    assert len(resolved) == 2
    assert resolved["agent_a"].model == "m5"
    assert resolved["agent_a"].prompt == "v5 (final)"
    assert resolved["agent_b"].model == "m4"
    assert resolved["agent_b"].prompt == "v4"


def test_resolve_agent_specs_empty_name_raises_error() -> None:
    """Test that empty agent name raises ValueError."""
    specs = [
        AgentSpec(mode="primary", model="gpt-4", prompt="test", name=""),
    ]

    with pytest.raises(ValueError, match="has no name \\(ambiguous identity\\)"):
        resolve_agent_specs(specs)


def test_resolve_agent_specs_single_agent_no_duplicates() -> None:
    """Test single agent returns correctly."""
    specs = [
        AgentSpec(mode="primary", model="gpt-4", prompt="test", name="solo"),
    ]

    resolved = resolve_agent_specs(specs)

    assert len(resolved) == 1
    assert resolved["solo"].model == "gpt-4"


def test_resolve_agent_specs_all_unique_names() -> None:
    """Test that all unique names are preserved."""
    specs = [
        AgentSpec(mode="primary", model="m1", prompt="p1", name="agent1"),
        AgentSpec(mode="subagent", model="m2", prompt="p2", name="agent2"),
        AgentSpec(mode="primary", model="m3", prompt="p3", name="agent3"),
    ]

    resolved = resolve_agent_specs(specs)

    assert len(resolved) == 3
    assert set(resolved.keys()) == {"agent1", "agent2", "agent3"}
    assert resolved["agent1"].model == "m1"
    assert resolved["agent2"].model == "m2"
    assert resolved["agent3"].model == "m3"


def test_resolve_agent_specs_empty_list() -> None:
    """Test empty spec list returns empty dict."""
    resolved = resolve_agent_specs([])

    assert resolved == {}
