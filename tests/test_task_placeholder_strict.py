"""Strict placeholder renderer tests using bounded grammar (string.Template).

Tests verify that the renderer:
- Supports $identifier and ${identifier} syntax
- Supports $$ escape for literal $
- Raises explicit error on missing placeholders
- Returns deterministic output from frozen snapshot dict
- No eval, no code execution, no Jinja/f-string expansion
"""

from __future__ import annotations

import pytest

from ecs_agent.placeholder import StrictPlaceholderRenderer


class TestStrictSubstitutionHappyPath:
    """Tests for successful placeholder substitution."""

    def test_strict_substitute_simple_dollar_identifier(self) -> None:
        """Test $identifier syntax with simple variable name."""
        renderer = StrictPlaceholderRenderer()
        template = "Hello $name, welcome!"
        snapshot = {"name": "Alice"}
        result = renderer.substitute(template, snapshot)
        assert result == "Hello Alice, welcome!"

    def test_strict_substitute_braced_identifier(self) -> None:
        """Test ${identifier} syntax for explicit boundaries."""
        renderer = StrictPlaceholderRenderer()
        template = "The ${color} fox jumps."
        snapshot = {"color": "brown"}
        result = renderer.substitute(template, snapshot)
        assert result == "The brown fox jumps."

    def test_strict_substitute_multiple_placeholders(self) -> None:
        """Test multiple distinct placeholders in single template."""
        renderer = StrictPlaceholderRenderer()
        template = "$user completed task $task_id with status $status"
        snapshot = {"user": "bob", "task_id": "42", "status": "success"}
        result = renderer.substitute(template, snapshot)
        assert result == "bob completed task 42 with status success"

    def test_strict_substitute_repeated_placeholder(self) -> None:
        """Test same placeholder used multiple times."""
        renderer = StrictPlaceholderRenderer()
        template = "Start $value middle $value end"
        snapshot = {"value": "X"}
        result = renderer.substitute(template, snapshot)
        assert result == "Start X middle X end"

    def test_strict_substitute_dollar_escape(self) -> None:
        """Test $$ escape for literal dollar sign."""
        renderer = StrictPlaceholderRenderer()
        template = "Price is $$100 and variable $amount"
        snapshot = {"amount": "50"}
        result = renderer.substitute(template, snapshot)
        assert result == "Price is $100 and variable 50"

    def test_strict_substitute_empty_template(self) -> None:
        """Test empty template string."""
        renderer = StrictPlaceholderRenderer()
        template = ""
        snapshot = {"key": "value"}
        result = renderer.substitute(template, snapshot)
        assert result == ""

    def test_strict_substitute_no_placeholders(self) -> None:
        """Test template with no placeholders."""
        renderer = StrictPlaceholderRenderer()
        template = "Just plain text, no variables"
        snapshot = {"unused": "value"}
        result = renderer.substitute(template, snapshot)
        assert result == "Just plain text, no variables"

    def test_strict_substitute_numeric_values(self) -> None:
        """Test that numeric values are converted to strings."""
        renderer = StrictPlaceholderRenderer()
        template = "Count: $count, Price: $price"
        snapshot = {"count": 42, "price": 3.14}
        result = renderer.substitute(template, snapshot)
        assert result == "Count: 42, Price: 3.14"

    def test_strict_substitute_boolean_values(self) -> None:
        """Test that boolean values are converted to strings."""
        renderer = StrictPlaceholderRenderer()
        template = "Enabled: $flag, Ready: $ready"
        snapshot = {"flag": True, "ready": False}
        result = renderer.substitute(template, snapshot)
        assert result == "Enabled: True, Ready: False"

    def test_strict_substitute_empty_string_value(self) -> None:
        """Test placeholder that resolves to empty string."""
        renderer = StrictPlaceholderRenderer()
        template = "Start$empty_var End"
        snapshot = {"empty_var": ""}
        result = renderer.substitute(template, snapshot)
        assert result == "Start End"

    def test_strict_substitute_whitespace_values(self) -> None:
        """Test placeholder that resolves to whitespace."""
        renderer = StrictPlaceholderRenderer()
        template = "Start$space|End"
        snapshot = {"space": "   "}
        result = renderer.substitute(template, snapshot)
        assert result == "Start   |End"

    def test_strict_substitute_special_chars_in_value(self) -> None:
        """Test that special characters in values are not interpreted."""
        renderer = StrictPlaceholderRenderer()
        template = "Value: $data"
        snapshot = {"data": "$notavar {{jinja}} {expr}"}
        result = renderer.substitute(template, snapshot)
        assert result == "Value: $notavar {{jinja}} {expr}"

    def test_strict_substitute_mixed_syntax(self) -> None:
        """Test mixing $ and ${} syntax in same template."""
        renderer = StrictPlaceholderRenderer()
        template = "Name: $name, Age: ${age}, City: $city"
        snapshot = {"name": "Carol", "age": "25", "city": "NYC"}
        result = renderer.substitute(template, snapshot)
        assert result == "Name: Carol, Age: 25, City: NYC"

    def test_strict_substitute_underscore_identifier(self) -> None:
        """Test identifier with underscores."""
        renderer = StrictPlaceholderRenderer()
        template = "Task: ${task_id}, User: $user_name"
        snapshot = {"task_id": "task-123", "user_name": "john_doe"}
        result = renderer.substitute(template, snapshot)
        assert result == "Task: task-123, User: john_doe"

    def test_strict_substitute_frozen_dict(self) -> None:
        """Test that snapshot dict is treated as frozen (no mutation)."""
        renderer = StrictPlaceholderRenderer()
        snapshot = {"key": "value"}
        template = "Value: $key"
        result = renderer.substitute(template, snapshot)
        assert result == "Value: value"
        # Verify snapshot was not mutated
        assert snapshot == {"key": "value"}

    def test_strict_substitute_unused_snapshot_keys(self) -> None:
        """Test that extra keys in snapshot are silently ignored."""
        renderer = StrictPlaceholderRenderer()
        template = "Used: $key1"
        snapshot = {"key1": "A", "key2": "B", "key3": "C"}
        result = renderer.substitute(template, snapshot)
        assert result == "Used: A"


class TestStrictSubstitutionErrorHandling:
    """Tests for error conditions and missing placeholders."""

    def test_missing_key_raises_explicit_error(self) -> None:
        """Test that missing placeholder key raises KeyError."""
        renderer = StrictPlaceholderRenderer()
        template = "Hello $name"
        snapshot: dict[str, str] = {}
        with pytest.raises(KeyError, match="name"):
            renderer.substitute(template, snapshot)

    def test_missing_key_in_braced_form(self) -> None:
        """Test missing key with ${} syntax raises error."""
        renderer = StrictPlaceholderRenderer()
        template = "Color: ${color}"
        snapshot: dict[str, str] = {}
        with pytest.raises(KeyError, match="color"):
            renderer.substitute(template, snapshot)

    def test_missing_key_with_other_keys_present(self) -> None:
        """Test that having some keys doesn't resolve missing ones."""
        renderer = StrictPlaceholderRenderer()
        template = "Name: $name, Age: $age"
        snapshot = {"name": "Dave"}
        with pytest.raises(KeyError, match="age"):
            renderer.substitute(template, snapshot)

    def test_multiple_placeholders_fail_on_first_missing(self) -> None:
        """Test error raised on first missing key encountered."""
        renderer = StrictPlaceholderRenderer()
        template = "$a $b $c"
        snapshot = {"a": "1", "b": "2"}  # missing c
        with pytest.raises(KeyError, match="c"):
            renderer.substitute(template, snapshot)

    def test_empty_snapshot_missing_all_placeholders(self) -> None:
        """Test that empty snapshot with placeholders raises error."""
        renderer = StrictPlaceholderRenderer()
        template = "Value: $key"
        snapshot: dict[str, str] = {}
        with pytest.raises(KeyError, match="key"):
            renderer.substitute(template, snapshot)


class TestStrictPlaceholderSyntax:
    """Tests for valid and invalid placeholder syntax."""

    def test_identifier_alphanumeric_underscore(self) -> None:
        """Test valid identifier with alphanumeric and underscores."""
        renderer = StrictPlaceholderRenderer()
        template = "$task_id_123"
        snapshot = {"task_id_123": "valid"}
        result = renderer.substitute(template, snapshot)
        assert result == "valid"

    def test_leading_underscore_identifier(self) -> None:
        """Test valid identifier starting with underscore."""
        renderer = StrictPlaceholderRenderer()
        template = "${_internal}"
        snapshot = {"_internal": "private"}
        result = renderer.substitute(template, snapshot)
        assert result == "private"

    def test_single_letter_identifier(self) -> None:
        """Test single-letter identifier."""
        renderer = StrictPlaceholderRenderer()
        template = "Value: $x"
        snapshot = {"x": "single"}
        result = renderer.substitute(template, snapshot)
        assert result == "Value: single"

    def test_numbers_in_identifier(self) -> None:
        """Test identifier with numbers (but not starting with number)."""
        renderer = StrictPlaceholderRenderer()
        template = "Step $step_2_output"
        snapshot = {"step_2_output": "result"}
        result = renderer.substitute(template, snapshot)
        assert result == "Step result"


class TestStrictPlaceholderSecurityAndDeterminism:
    """Tests for security (no eval) and deterministic behavior."""

    def test_no_code_execution_in_placeholder(self) -> None:
        """Test that placeholder values are not executed as code."""
        renderer = StrictPlaceholderRenderer()
        template = "Result: $code"
        snapshot = {"code": "1 + 1"}
        result = renderer.substitute(template, snapshot)
        # Value is literal string, not evaluated to 2
        assert result == "Result: 1 + 1"

    def test_no_jinja_expansion(self) -> None:
        """Test that Jinja syntax is not expanded."""
        renderer = StrictPlaceholderRenderer()
        template = "Value: $var"
        snapshot = {"var": "{{ import_some_module() }}"}
        result = renderer.substitute(template, snapshot)
        assert result == "Value: {{ import_some_module() }}"

    def test_no_format_string_expansion(self) -> None:
        """Test that f-string syntax is not expanded."""
        renderer = StrictPlaceholderRenderer()
        template = "Value: $var"
        snapshot = {"var": "{1 + 1}"}
        result = renderer.substitute(template, snapshot)
        assert result == "Value: {1 + 1}"

    def test_no_shell_escapes(self) -> None:
        """Test that shell metacharacters are not interpreted."""
        renderer = StrictPlaceholderRenderer()
        template = "Command: $cmd"
        snapshot = {"cmd": "$(rm -rf /)"}
        result = renderer.substitute(template, snapshot)
        assert result == "Command: $(rm -rf /)"

    def test_deterministic_output_same_inputs(self) -> None:
        """Test that same inputs produce same output (deterministic)."""
        renderer = StrictPlaceholderRenderer()
        template = "User $user has $count items"
        snapshot = {"user": "alice", "count": "5"}

        result1 = renderer.substitute(template, snapshot)
        result2 = renderer.substitute(template, snapshot)

        assert result1 == result2
        assert result1 == "User alice has 5 items"


class TestStrictPlaceholderRendererInitialization:
    """Tests for renderer initialization and state."""

    def test_renderer_instantiation(self) -> None:
        """Test basic renderer instantiation."""
        renderer = StrictPlaceholderRenderer()
        assert renderer is not None

    def test_renderer_multiple_instances_independent(self) -> None:
        """Test that multiple renderer instances are independent."""
        renderer1 = StrictPlaceholderRenderer()
        renderer2 = StrictPlaceholderRenderer()

        template = "Value: $x"
        snapshot1 = {"x": "first"}
        snapshot2 = {"x": "second"}

        result1 = renderer1.substitute(template, snapshot1)
        result2 = renderer2.substitute(template, snapshot2)

        assert result1 == "Value: first"
        assert result2 == "Value: second"

    def test_renderer_stateless_across_calls(self) -> None:
        """Test that renderer state doesn't leak between calls."""
        renderer = StrictPlaceholderRenderer()

        result1 = renderer.substitute("$a", {"a": "1"})
        result2 = renderer.substitute("$b", {"b": "2"})
        result3 = renderer.substitute("$c", {"c": "3"})

        assert result1 == "1"
        assert result2 == "2"
        assert result3 == "3"
