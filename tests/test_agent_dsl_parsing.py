"""Tests for Agent DSL schema validation."""

import pytest
import tempfile
from pathlib import Path

import yaml
import json

from ecs_agent.dsl.markdown_loader import load_markdown_agent
from ecs_agent.dsl.json_loader import load_json_agents
from ecs_agent.dsl.schema import AgentSpec, validate_agent_spec


class TestAgentSpecValidation:
    """Test suite for AgentSpec validation."""

    def test_valid_minimal_primary_agent(self) -> None:
        """Valid spec with only required fields for primary agent."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "You are a helpful assistant.",
        }
        spec = validate_agent_spec(data, source_name="test_agent")
        assert spec.mode == "primary"
        assert spec.model == "gpt-4"
        assert spec.prompt == "You are a helpful assistant."
        assert spec.tools == {}
        assert spec.metadata == {}
        assert spec.name == ""

    def test_valid_minimal_subagent(self) -> None:
        """Valid spec with only required fields for subagent."""
        data = {
            "mode": "subagent",
            "model": "claude-3-opus",
            "prompt": "You are a code reviewer.",
        }
        spec = validate_agent_spec(data)
        assert spec.mode == "subagent"
        assert spec.model == "claude-3-opus"
        assert spec.prompt == "You are a code reviewer."

    def test_valid_full_spec_with_all_fields(self) -> None:
        """Valid spec with all optional fields populated."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "You are a helpful assistant.",
            "tools": {"read_file": True, "write_file": False},
            "metadata": {"team": "engineering", "version": "1.0"},
            "name": "code_assistant",
        }
        spec = validate_agent_spec(data, source_name="full_agent.json")
        assert spec.mode == "primary"
        assert spec.model == "gpt-4"
        assert spec.tools == {"read_file": True, "write_file": False}
        assert spec.metadata == {"team": "engineering", "version": "1.0"}
        assert spec.name == "code_assistant"

    def test_valid_empty_tools_and_metadata(self) -> None:
        """Valid spec with explicit empty tools and metadata."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "You are a helpful assistant.",
            "tools": {},
            "metadata": {},
        }
        spec = validate_agent_spec(data)
        assert spec.tools == {}
        assert spec.metadata == {}

    @pytest.mark.parametrize(
        "missing_field,data",
        [
            ("mode", {"model": "gpt-4", "prompt": "test"}),
            ("model", {"mode": "primary", "prompt": "test"}),
            ("prompt", {"mode": "primary", "model": "gpt-4"}),
            (
                "mode,model",
                {"prompt": "test"},
            ),
        ],
    )
    def test_missing_required_fields(self, missing_field: str, data: dict) -> None:
        """Missing required fields raise ValueError with context."""
        with pytest.raises(ValueError, match="Missing required field"):
            validate_agent_spec(data, source_name="test_agent")

    def test_unknown_field_fails_fast(self) -> None:
        """Unknown fields raise ValueError with field name."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "test",
            "unknown_field": "value",
        }
        with pytest.raises(ValueError, match="Unknown field.*unknown_field"):
            validate_agent_spec(data, source_name="bad_agent.json")

    def test_multiple_unknown_fields(self) -> None:
        """Multiple unknown fields are all reported."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "test",
            "bad_field_1": "value1",
            "bad_field_2": "value2",
        }
        with pytest.raises(ValueError, match="Unknown field.*bad_field"):
            validate_agent_spec(data)

    def test_invalid_mode_value(self) -> None:
        """Invalid mode literal raises ValueError."""
        data = {
            "mode": "invalid_mode",
            "model": "gpt-4",
            "prompt": "test",
        }
        with pytest.raises(
            ValueError, match="Invalid mode.*must be 'primary' or 'subagent'"
        ):
            validate_agent_spec(data, source_name="test_agent")

    def test_tools_not_dict_raises_type_error(self) -> None:
        """Tools field must be a dict."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "test",
            "tools": ["read_file", "write_file"],  # list instead of dict
        }
        with pytest.raises(TypeError, match="Field 'tools' must be dict"):
            validate_agent_spec(data)

    def test_tools_value_not_bool_raises_type_error(self) -> None:
        """Tool enabled value must be bool."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "test",
            "tools": {"read_file": "yes"},  # string instead of bool
        }
        with pytest.raises(TypeError, match="Tool 'read_file' value must be bool"):
            validate_agent_spec(data, source_name="test_agent")

    def test_metadata_not_dict_raises_type_error(self) -> None:
        """Metadata field must be a dict."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "test",
            "metadata": "invalid",  # string instead of dict
        }
        with pytest.raises(TypeError, match="Field 'metadata' must be dict"):
            validate_agent_spec(data)

    def test_name_not_str_raises_type_error(self) -> None:
        """Name field must be a string."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "test",
            "name": 123,  # int instead of str
        }
        with pytest.raises(TypeError, match="Field 'name' must be str"):
            validate_agent_spec(data)

    def test_source_name_appears_in_error_messages(self) -> None:
        """Source name is included in validation error messages."""
        data = {"mode": "primary", "model": "gpt-4"}  # missing prompt
        with pytest.raises(ValueError, match="in 'agents/primary.json'"):
            validate_agent_spec(data, source_name="agents/primary.json")

    def test_dataclass_slots_enabled(self) -> None:
        """AgentSpec uses __slots__ for memory efficiency."""
        spec = AgentSpec(mode="primary", model="gpt-4", prompt="test")
        assert hasattr(spec, "__slots__")

    def test_dataclass_immutability_not_enforced(self) -> None:
        """AgentSpec is mutable (frozen=False)."""
        spec = AgentSpec(mode="primary", model="gpt-4", prompt="test")
        spec.name = "updated_name"
        assert spec.name == "updated_name"

    def test_tools_with_multiple_entries(self) -> None:
        """Tools with multiple bool values validate correctly."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "test",
            "tools": {
                "read_file": True,
                "write_file": False,
                "execute_bash": True,
                "web_search": False,
            },
        }
        spec = validate_agent_spec(data)
        assert len(spec.tools) == 4
        assert spec.tools["read_file"] is True
        assert spec.tools["write_file"] is False

    def test_metadata_with_complex_values(self) -> None:
        """Metadata can contain nested dicts and lists."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "test",
            "metadata": {
                "team": "engineering",
                "tags": ["production", "critical"],
                "config": {"timeout": 30, "retries": 3},
            },
        }
        spec = validate_agent_spec(data)
        assert spec.metadata["team"] == "engineering"
        assert spec.metadata["tags"] == ["production", "critical"]
        assert spec.metadata["config"]["timeout"] == 30


class TestMarkdownAgentLoader:
    """Test suite for Markdown agent loader."""

    def test_markdown_agent_parses_frontmatter_and_body(self) -> None:
        """load_markdown_agent extracts YAML frontmatter and markdown body."""
        content = """---
mode: primary
model: gpt-4
tools:
  read_file: true
  write_file: false
metadata:
  team: engineering
---
You are a helpful assistant.
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "code_assistant.md"
            agent_path.write_text(content)

            spec = load_markdown_agent(agent_path)

            assert spec.mode == "primary"
            assert spec.model == "gpt-4"
            assert spec.prompt == "You are a helpful assistant."
            assert spec.tools == {"read_file": True, "write_file": False}
            assert spec.metadata == {"team": "engineering"}
            # Filename is authoritative for name
            assert spec.name == "code_assistant"

    def test_markdown_filename_overrides_frontmatter_name(self) -> None:
        """Filename is authoritative agent name, frontmatter name is ignored."""
        content = """---
mode: primary
model: gpt-4
prompt: System prompt from frontmatter
name: frontmatter_name
---
Body prompt (should be ignored since frontmatter has prompt)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "filename_wins.md"
            agent_path.write_text(content)

            spec = load_markdown_agent(agent_path)

            # Filename ALWAYS wins
            assert spec.name == "filename_wins"
            # Frontmatter prompt is used
            assert spec.prompt == "Body prompt (should be ignored since frontmatter has prompt)"

    def test_markdown_invalid_yaml_raises_error(self) -> None:
        """Invalid YAML frontmatter raises yaml.YAMLError with file path."""
        content = """---
mode: primary
model: gpt-4
prompt: test
tools:
  - this is invalid yaml structure
    read_file: true
---
Body content
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "bad_yaml.md"
            agent_path.write_text(content)

            with pytest.raises(yaml.YAMLError, match="Invalid YAML frontmatter"):
                load_markdown_agent(agent_path)

    def test_markdown_no_frontmatter_raises_validation_error(self) -> None:
        """Markdown with no frontmatter raises validation error for missing fields."""
        content = """You are a helpful assistant."""

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "no_frontmatter.md"
            agent_path.write_text(content)

            # Should fail validation because mode and model are missing
            with pytest.raises(ValueError, match="Missing required field"):
                load_markdown_agent(agent_path)

    def test_markdown_malformed_frontmatter_delimiter(self) -> None:
        """Malformed frontmatter (only opening ---) treats content as body."""
        content = """---
This is not valid frontmatter
Just plain text"""

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "malformed.md"
            agent_path.write_text(content)

            # Should fail validation because mode and model are missing
            with pytest.raises(ValueError, match="Missing required field"):
                load_markdown_agent(agent_path)

    def test_markdown_empty_body_is_valid(self) -> None:
        """Empty markdown body is valid (prompt can be empty string)."""
        content = """---
mode: primary
model: gpt-4
prompt: System prompt
---
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "empty_body.md"
            agent_path.write_text(content)

            spec = load_markdown_agent(agent_path)

            assert spec.mode == "primary"
            assert spec.model == "gpt-4"
            # Body overwrites frontmatter prompt (body is empty)
            assert spec.prompt == ""
            assert spec.name == "empty_body"

    def test_markdown_file_not_found(self) -> None:
        """FileNotFoundError raised for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Markdown agent file not found"):
            load_markdown_agent("/nonexistent/path/agent.md")

    def test_markdown_body_overwrites_frontmatter_prompt(self) -> None:
        """Markdown body always becomes the prompt (frontmatter prompt is ignored)."""
        content = """---
mode: subagent
model: claude-3-opus
prompt: This prompt from frontmatter should be overwritten
---
# Actual Agent Prompt

This is the real system prompt from the markdown body."""

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "body_wins.md"
            agent_path.write_text(content)

            spec = load_markdown_agent(agent_path)

            assert spec.mode == "subagent"
            # Body ALWAYS wins for prompt
            assert spec.prompt == "# Actual Agent Prompt\n\nThis is the real system prompt from the markdown body."
            assert spec.name == "body_wins"


class TestUnicodeAndSpecialCharacters:
    """Test suite for unicode and special character handling in agent names."""

    def test_agent_name_with_chinese_characters(self) -> None:
        """Agent name with Chinese characters validates correctly."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "test",
            "name": "智能助手",
        }
        spec = validate_agent_spec(data)
        assert spec.name == "智能助手"

    def test_agent_name_with_japanese_emoji(self) -> None:
        """Agent name with Japanese and emoji validates correctly."""
        data = {
            "mode": "subagent",
            "model": "claude-3-opus",
            "prompt": "test",
            "name": "コード助手🤖",
        }
        spec = validate_agent_spec(data)
        assert spec.name == "コード助手🤖"

    def test_agent_name_with_special_punctuation(self) -> None:
        """Agent name with special punctuation and spaces validates correctly."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "test",
            "name": "code-assistant_v2.0 (beta)",
        }
        spec = validate_agent_spec(data)
        assert spec.name == "code-assistant_v2.0 (beta)"

    def test_agent_name_empty_string(self) -> None:
        """Empty string name is valid (default value)."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "test",
            "name": "",
        }
        spec = validate_agent_spec(data)
        assert spec.name == ""

    def test_agent_name_whitespace_only(self) -> None:
        """Whitespace-only name is valid."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "test",
            "name": "   ",
        }
        spec = validate_agent_spec(data)
        assert spec.name == "   "

    def test_prompt_with_unicode_and_special_chars(self) -> None:
        """Prompt with mixed unicode and special characters validates correctly."""
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "You are a helpful assistant. 你好！🎯 Use quotes \"like this\" and newlines.\n\nMultiple paragraphs.",
        }
        spec = validate_agent_spec(data)
        assert "你好" in spec.prompt
        assert "🎯" in spec.prompt
        assert "\"" in spec.prompt


class TestLargePayloadAndPerformance:
    """Test suite for large file/payload scenarios and memory safety."""

    def test_prompt_with_very_long_content(self) -> None:
        """Very long prompt (10K+ chars) validates correctly."""
        long_prompt = "System instruction: " + "A" * 10000
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": long_prompt,
        }
        spec = validate_agent_spec(data)
        assert len(spec.prompt) > 10000
        assert spec.prompt.startswith("System instruction: A")

    def test_tools_with_large_number_of_entries(self) -> None:
        """Tools dict with 100+ entries validates correctly."""
        large_tools = {f"tool_{i}": i % 2 == 0 for i in range(150)}
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "test",
            "tools": large_tools,
        }
        spec = validate_agent_spec(data)
        assert len(spec.tools) == 150
        assert spec.tools["tool_0"] is True
        assert spec.tools["tool_1"] is False

    def test_metadata_with_deeply_nested_structure(self) -> None:
        """Deeply nested metadata dict validates correctly."""
        nested_metadata = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {
                                "data": "deep_value",
                                "list": [1, 2, 3],
                            }
                        }
                    }
                }
            },
            "flat_key": "flat_value",
        }
        data = {
            "mode": "primary",
            "model": "gpt-4",
            "prompt": "test",
            "metadata": nested_metadata,
        }
        spec = validate_agent_spec(data)
        assert spec.metadata["level1"]["level2"]["level3"]["level4"]["level5"]["data"] == "deep_value"
        assert spec.metadata["flat_key"] == "flat_value"


class TestYAMLFrontmatterEdgeCases:
    """Test suite for additional YAML frontmatter edge cases in Markdown loader."""

    def test_markdown_empty_frontmatter_block(self) -> None:
        """Empty frontmatter block (---\n---) is parsed as empty dict."""
        content = """---
---
You are a helpful assistant."""

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "empty_frontmatter.md"
            agent_path.write_text(content)

            # Should fail validation because mode and model are missing
            with pytest.raises(ValueError, match="Missing required field"):
                load_markdown_agent(agent_path)

    def test_markdown_frontmatter_without_closing_delimiter(self) -> None:
        """Frontmatter without closing delimiter treats entire content as body."""
        content = """---
mode: primary
model: gpt-4
This is missing closing delimiter"""

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "no_closing.md"
            agent_path.write_text(content)

            # Should fail validation (no frontmatter parsed, so no mode/model)
            with pytest.raises(ValueError, match="Missing required field"):
                load_markdown_agent(agent_path)

    def test_markdown_frontmatter_with_tabs_raises_yaml_error(self) -> None:
        """YAML frontmatter with literal tabs (invalid YAML) raises YAMLError."""
        content = """---
mode:\tprimary
model:\tgpt-4
---
You are a helpful assistant."""

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "tabs.md"
            agent_path.write_text(content)

            # YAML parser rejects literal tabs in values
            with pytest.raises(yaml.YAMLError, match="Invalid YAML frontmatter"):
                load_markdown_agent(agent_path)

    def test_markdown_frontmatter_with_nested_yaml_lists(self) -> None:
        """Nested YAML lists in metadata validate correctly."""
        content = """---
mode: primary
model: gpt-4
metadata:
  teams:
    - engineering
    - research
  configs:
    - name: prod
      timeout: 30
    - name: dev
      timeout: 10
---
You are a helpful assistant."""

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "nested_lists.md"
            agent_path.write_text(content)

            spec = load_markdown_agent(agent_path)

            assert spec.metadata["teams"] == ["engineering", "research"]
            assert len(spec.metadata["configs"]) == 2
            assert spec.metadata["configs"][0]["name"] == "prod"

    def test_markdown_frontmatter_with_yaml_anchors_and_aliases(self) -> None:
        """YAML anchors and aliases in frontmatter parse correctly."""
        content = """---
mode: primary
model: gpt-4
metadata:
  defaults: &defaults
    timeout: 30
    retries: 3
  prod:
    <<: *defaults
    env: production
---
You are a helpful assistant."""

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "yaml_anchors.md"
            agent_path.write_text(content)

            spec = load_markdown_agent(agent_path)

            assert spec.metadata["prod"]["timeout"] == 30
            assert spec.metadata["prod"]["retries"] == 3
            assert spec.metadata["prod"]["env"] == "production"

    def test_markdown_frontmatter_with_multiline_strings(self) -> None:
        """YAML multiline strings (| and >) in frontmatter parse correctly."""
        content = """---
mode: primary
model: gpt-4
metadata:
  description: |
    This is a multiline
    description that preserves
    newlines.
  summary: >
    This is a folded
    multiline string that
    becomes a single line.
---
You are a helpful assistant."""

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "multiline.md"
            agent_path.write_text(content)

            spec = load_markdown_agent(agent_path)

            assert "\n" in spec.metadata["description"]
            assert "multiline" in spec.metadata["description"]
            assert isinstance(spec.metadata["summary"], str)


class TestJSONMalformedInputs:
    """Test suite for JSON malformed input variations with json_loader."""

    def test_json_with_trailing_commas(self) -> None:
        """JSON with trailing commas raises JSONDecodeError."""
        content = """
{
  "agent1": {
    "mode": "primary",
    "model": "gpt-4",
    "prompt": "test",
  },
}
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "trailing_comma.json"
            json_path.write_text(content)

            with pytest.raises(json.JSONDecodeError):
                load_json_agents(json_path)

    def test_json_with_single_quotes(self) -> None:
        """JSON with single quotes instead of double raises JSONDecodeError."""
        content = """
{
  'agent1': {
    'mode': 'primary',
    'model': 'gpt-4',
    'prompt': 'test'
  }
}
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "single_quotes.json"
            json_path.write_text(content)

            with pytest.raises(json.JSONDecodeError):
                load_json_agents(json_path)

    def test_json_with_comments(self) -> None:
        """JSON with comments (not valid JSON) raises JSONDecodeError."""
        content = """
{
  // This is a comment
  "agent1": {
    "mode": "primary",
    "model": "gpt-4",
    "prompt": "test"
  }
}
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "with_comments.json"
            json_path.write_text(content)

            with pytest.raises(json.JSONDecodeError):
                load_json_agents(json_path)

    def test_json_with_unescaped_newlines(self) -> None:
        """JSON with unescaped newlines in strings raises JSONDecodeError."""
        # Create content with actual unescaped newline (invalid JSON)
        content = '{\n  "agent1": {\n    "mode": "primary",\n    "model": "gpt-4",\n    "prompt": "line1\nline2"\n  }\n}'

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "unescaped_newlines.json"
            json_path.write_text(content)

            with pytest.raises(json.JSONDecodeError):
                load_json_agents(json_path)

    def test_json_root_not_dict_raises_validation_error(self) -> None:
        """JSON root as array instead of dict raises ValueError."""
        content = """
[
  {
    "mode": "primary",
    "model": "gpt-4",
    "prompt": "test"
  }
]
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "array_root.json"
            json_path.write_text(content)

            with pytest.raises(ValueError, match="JSON root must be dict"):
                load_json_agents(json_path)

    def test_json_agent_config_not_dict(self) -> None:
        """Agent config value as string instead of dict raises ValueError."""
        content = """
{
  "agent1": "invalid_string_value"
}
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config_not_dict.json"
            json_path.write_text(content)

            with pytest.raises(ValueError, match="Agent 'agent1' config must be dict"):
                load_json_agents(json_path)
