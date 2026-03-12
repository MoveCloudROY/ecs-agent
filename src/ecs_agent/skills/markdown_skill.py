"""Markdown-based skill parser.

Loads skills from SKILL.md files with YAML frontmatter.
"""

import asyncio
import logging
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ecs_agent.core.world import World
from ecs_agent.components.definitions import (
    SystemPromptComponent,
    ToolRegistryComponent,
    SkillComponent,
)
from ecs_agent.logging import get_logger
from ecs_agent.skills.protocol import ToolHandler
from ecs_agent.types import EntityId, ToolSchema

logger = get_logger(__name__)
_stdlib_logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"[a-z0-9-]{1,64}")

# Pre-compiled substitution patterns (process $ARGUMENTS[N] before $ARGUMENTS)
_ARGS_INDEXED_RE = re.compile(r"\$ARGUMENTS\[(\d+)\]")
_ARGS_RE = re.compile(r"\$ARGUMENTS(?!\[)")
_DOLLAR_N_RE = re.compile(r"\$([1-9])(?!\w)")
_SESSION_ID_RE = re.compile(r"\$\{CLAUDE_SESSION_ID\}")
_SKILL_DIR_RE = re.compile(r"\$\{CLAUDE_SKILL_DIR\}")
_DYNAMIC_INJECTION_RE = re.compile(r"!`[^`]*`")


def render_skill_content(template: str, arguments: str, skill_dir: Path) -> str:
    """Render a skill template with Claude-compatible string substitutions.

    Substitution forms (all required):
        $ARGUMENTS        → entire arguments string
        $ARGUMENTS[N]     → Nth word (0-indexed), empty string if out of bounds
        $N (N is 1–9)    → word at index N-1, empty string if out of bounds
        ${CLAUDE_SESSION_ID} → literal '<session-id>'
        ${CLAUDE_SKILL_DIR}  → str(skill_dir)
        ${UNKNOWN_VAR}    → left as-is

    Uses regex substitution only. No eval(), no str.format(), no shell.

    Args:
        template: Template string with substitution variables
        arguments: Whitespace-separated arguments string
        skill_dir: Skill base directory (used for ${CLAUDE_SKILL_DIR})

    Returns:
        Template with all substitution variables resolved
    """
    words = arguments.split() if arguments.strip() else []

    def _replace_indexed(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        return words[idx] if idx < len(words) else ""

    def _replace_dollar_n(match: re.Match[str]) -> str:
        idx = int(match.group(1)) - 1  # $1 → index 0
        return words[idx] if idx < len(words) else ""

    # Order matters: $ARGUMENTS[N] must be processed before bare $ARGUMENTS
    result = _ARGS_INDEXED_RE.sub(_replace_indexed, template)
    result = _ARGS_RE.sub(arguments, result)
    result = _DOLLAR_N_RE.sub(_replace_dollar_n, result)
    result = _SESSION_ID_RE.sub("<session-id>", result)
    result = _SKILL_DIR_RE.sub(str(skill_dir), result)
    return result


class Skill:
    """Skill loaded from a SKILL.md file with YAML frontmatter.

    SKILL.md format:
    ---
    name: skill-name
    description: Skill description
    ---
    # Markdown body (system prompt)

    Optional scripts/ directory with .py files for tool handlers.
    """

    def __init__(self, skill_path: Path, sandbox_timeout: float = 30.0) -> None:
        """Initialize Skill by parsing SKILL.md.

        Args:
            skill_path: Path to SKILL.md file
            sandbox_timeout: Timeout for subprocess tool execution
        """
        self._skill_path = skill_path
        self._sandbox_timeout = sandbox_timeout
        self._parse_skill_file()

    def _parse_skill_file(self) -> None:
        """Parse YAML frontmatter and markdown body using strict line-delimited approach.

        Reads the file in binary mode to allow the body to contain arbitrary bytes.
        Only the frontmatter section (before the closing ---) needs to be valid UTF-8.
        The body is stored as raw bytes and decoded lazily in system_prompt().
        """
        raw = self._skill_path.read_bytes()
        # Try to decode up to potential frontmatter boundary in UTF-8
        # We need to find b"---" line boundaries in raw bytes
        # Split on newlines at byte level
        separator = b"\n"
        lines_bytes = raw.split(separator)

        # Strip trailing \r from each line (handle CRLF)
        stripped_lines = [line.rstrip(b"\r") for line in lines_bytes]

        # Find first non-empty line — must be exactly b"---" to have frontmatter
        first_non_empty_idx: int | None = None
        for i, line in enumerate(stripped_lines):
            if line.strip():
                first_non_empty_idx = i
                break

        frontmatter_text = ""
        # body_bytes will be decoded lazily
        self._body_bytes: bytes = raw.strip()

        if (
            first_non_empty_idx is not None
            and stripped_lines[first_non_empty_idx] == b"---"
        ):
            # Look for closing "---"
            closing_idx: int | None = None
            for i in range(first_non_empty_idx + 1, len(stripped_lines)):
                if stripped_lines[i] == b"---":
                    closing_idx = i
                    break

            if closing_idx is not None:
                # Frontmatter bytes are safe UTF-8 YAML
                fm_lines = stripped_lines[first_non_empty_idx + 1 : closing_idx]
                try:
                    frontmatter_text = b"\n".join(fm_lines).decode("utf-8")
                except UnicodeDecodeError:
                    frontmatter_text = b"\n".join(fm_lines).decode(
                        "utf-8", errors="replace"
                    )
                # Body bytes: everything after closing ---
                body_lines = stripped_lines[closing_idx + 1 :]
                self._body_bytes = separator.join(body_lines).strip()

        # Attempt YAML parse
        metadata: dict[str, Any] = {}
        if frontmatter_text:
            try:
                parsed = yaml.safe_load(frontmatter_text)
                if isinstance(parsed, dict):
                    metadata = parsed
            except yaml.YAMLError as exc:
                logger.warning(
                    "markdown_skill_invalid_yaml",
                    skill_path=str(self._skill_path),
                    exception=str(exc),
                )
                _stdlib_logger.warning(
                    "markdown_skill_invalid_yaml: %s", str(self._skill_path)
                )
                self.valid = False
                self._name = ""
                self._description = ""
                return

        # Check required fields
        raw_name = metadata.get("name")
        raw_description = metadata.get("description")

        if not frontmatter_text or raw_name is None or raw_description is None:
            if frontmatter_text and (raw_name is None or raw_description is None):
                # Has frontmatter but missing required fields
                logger.warning(
                    "markdown_skill_missing_required_field",
                    skill_path=str(self._skill_path),
                    missing_name=(raw_name is None),
                    missing_description=(raw_description is None),
                )
                _stdlib_logger.warning(
                    "markdown_skill_missing_required_field: required fields missing in %s",
                    str(self._skill_path),
                )
                self.valid = False
                self._name = str(raw_name) if raw_name is not None else ""
                self._description = (
                    str(raw_description) if raw_description is not None else ""
                )
                return
            # No frontmatter — reject per new spec (frontmatter required)
            logger.warning(
                "markdown_skill_no_frontmatter",
                skill_path=str(self._skill_path),
            )
            _stdlib_logger.warning(
                "markdown_skill_no_frontmatter: no YAML frontmatter in %s",
                str(self._skill_path),
            )
            self.valid = False
            self._name = ""
            self._description = ""
            return

        name = str(raw_name)
        description = str(raw_description)

        # Validate name format
        if not _NAME_RE.fullmatch(name):
            logger.warning(
                "markdown_skill_invalid_name",
                skill_path=str(self._skill_path),
                name=name,
            )
            _stdlib_logger.warning(
                "markdown_skill_invalid_name: invalid name format %r in %s",
                name,
                str(self._skill_path),
            )
            self.valid = False
            self._name = name
            self._description = description
            return

        self.valid = True
        self._name = name
        self._description = description
        self._user_invocable: bool = bool(metadata.get("user-invocable", True))
        self._disable_model_invocation: bool = bool(
            metadata.get("disable-model-invocation", False)
        )
        self._argument_hint: str = str(metadata.get("argument-hint", ""))
        raw_allowed = metadata.get("allowed-tools", [])
        self._allowed_tools: list[str] = (
            [str(t) for t in raw_allowed] if isinstance(raw_allowed, list) else []
        )
        self._context: str | None = (
            str(metadata["context"]) if "context" in metadata else None
        )
        self._agent: str | None = (
            str(metadata["agent"]) if "agent" in metadata else None
        )
        self._model: str | None = (
            str(metadata["model"]) if "model" in metadata else None
        )
        raw_hooks = metadata.get("hooks", {})
        self._hooks: dict[str, Any] = (
            dict(raw_hooks) if isinstance(raw_hooks, dict) else {}
        )

    @property
    def name(self) -> str:
        """Skill name from frontmatter or filename."""
        return self._name

    @property
    def description(self) -> str:
        """Skill description from frontmatter."""
        return self._description

    @property
    def slash_command(self) -> str:
        """Slash command token for this skill."""
        return f"/{self._name}"

    @property
    def skill_dir_path(self) -> Path:
        """Parent directory of the SKILL.md file."""
        return self._skill_path.parent

    @property
    def user_invocable(self) -> bool:
        """Whether the skill can be invoked by the user."""
        return getattr(self, "_user_invocable", True)

    @property
    def disable_model_invocation(self) -> bool:
        """Whether model-initiated invocation is disabled."""
        return getattr(self, "_disable_model_invocation", False)

    @property
    def injection_policy(self) -> str:
        """Dynamic injection handling policy."""
        return "deny"

    @property
    def argument_hint(self) -> str:
        """Argument hint shown to user."""
        return getattr(self, "_argument_hint", "")

    @property
    def allowed_tools(self) -> list[str]:
        """List of allowed tool names."""
        return getattr(self, "_allowed_tools", [])

    @property
    def context(self) -> str | None:
        """Context routing field."""
        return getattr(self, "_context", None)

    @property
    def agent(self) -> str | None:
        """Agent routing field."""
        return getattr(self, "_agent", None)

    @property
    def model(self) -> str | None:
        """Model routing field."""
        return getattr(self, "_model", None)

    @property
    def hooks(self) -> dict[str, Any]:
        """Hook configurations."""
        return getattr(self, "_hooks", {})

    def resolve_supporting_path(self, relative_path: str) -> Path:
        """Resolve a supporting file path relative to the skill directory.

        Args:
            relative_path: Path relative to skill directory

        Returns:
            Absolute resolved path

        Raises:
            ValueError: If the resolved path is outside the skill directory (path traversal)
        """
        skill_dir = self._skill_path.parent.resolve()
        resolved = (skill_dir / relative_path).resolve()
        if not resolved.is_relative_to(skill_dir):
            raise ValueError(
                f"Path traversal detected: {relative_path!r} resolves outside skill directory"
            )
        return resolved

    def render_with_arguments(self, template: str, arguments: str) -> str:
        """Render template with Claude-compatible substitutions using skill_dir_path.

        Args:
            template: Template string with substitution variables
            arguments: Whitespace-separated arguments string

        Returns:
            Template with all substitution variables resolved
        """
        return render_skill_content(template, arguments, self._skill_path.parent)

    def is_dynamic_injection_safe(self, content: str) -> bool:
        """Return whether dynamic content is safe to process."""
        return _DYNAMIC_INJECTION_RE.search(content) is None

    def system_prompt(self) -> str:
        """Return markdown body as system prompt."""
        return self._body_bytes.decode("utf-8", errors="replace").strip()

    def tools(self) -> dict[str, tuple[ToolSchema, ToolHandler]]:
        """Discover tools from scripts/ directory.

        Returns:
            Dict mapping tool name to (ToolSchema, handler) tuple
        """
        tools_dict: dict[str, tuple[ToolSchema, ToolHandler]] = {}

        # Check if scripts/ directory exists alongside SKILL.md
        scripts_dir = self._skill_path.parent / "scripts"
        if not scripts_dir.exists():
            return tools_dict

        # Discover all .py files in scripts/
        for script_path in sorted(scripts_dir.glob("*.py")):
            tool_name = script_path.stem
            schema = ToolSchema(
                name=tool_name,
                description=f"Execute {tool_name} script",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
                sandbox_compatible=False,
            )
            handler = self._create_script_handler(script_path)
            tools_dict[tool_name] = (schema, handler)

        return tools_dict

    def _create_script_handler(self, script_path: Path) -> ToolHandler:
        """Create async handler that executes script via subprocess.

        Args:
            script_path: Path to Python script

        Returns:
            Async handler function
        """

        async def handler(**kwargs: Any) -> str:
            """Execute script with JSON arguments via stdin."""
            try:
                # Prepare arguments as JSON
                args_json = json.dumps(kwargs)

                # Run subprocess with JSON stdin
                result = await asyncio.create_subprocess_exec(
                    "python3",
                    str(script_path),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    result.communicate(input=args_json.encode()),
                    timeout=self._sandbox_timeout,
                )

                stdout = stdout_bytes.decode().strip()
                stderr = stderr_bytes.decode().strip()

                if result.returncode == 0:
                    return stdout
                else:
                    error_msg = f"Script failed with exit code {result.returncode}"
                    if stderr:
                        error_msg += f"\nStderr: {stderr}"
                    return error_msg

            except asyncio.TimeoutError:
                return f"Script execution timed out after {self._sandbox_timeout}s"
            except Exception as exc:
                logger.error(
                    "markdown_skill_script_error",
                    script=str(script_path),
                    exception=str(exc),
                )
                return f"Script execution failed: {exc}"

        return handler

    def install(self, world: World, entity_id: EntityId) -> None:
        """Install skill by adding system prompt and tools.

        When called after SkillManager.activate() (the canonical lifecycle path),
        this method is a no-op because the manager has already registered tools
        and injected the system prompt. Idempotency is checked via SkillComponent:
        if the manager has indexed or activated this skill, skip all work here to
        prevent duplicate tool entries and doubled system prompts.

        Args:
            world: World instance
            entity_id: Entity to install skill on
        """
        # Guard: if SkillManager has already indexed (or activated) this skill,
        # all tool and prompt registration was done by the manager. Skip to avoid
        # duplicate registrations — SkillManager is the canonical lifecycle owner.
        skill_comp = world.get_component(entity_id, SkillComponent)
        if skill_comp is not None and self.name in skill_comp.skills:
            return

        # Standalone path (no manager): register tools, prompt, and SkillComponent.
        # Add or update SystemPromptComponent
        prompt_comp = world.get_component(entity_id, SystemPromptComponent)
        if prompt_comp is None:
            prompt_comp = SystemPromptComponent(content=self.system_prompt())
            world.add_component(entity_id, prompt_comp)
        else:
            # Append to existing prompt with separator
            prompt_comp.content += f"\n\n{self.system_prompt()}"

        # Add or update ToolRegistryComponent
        tool_reg = world.get_component(entity_id, ToolRegistryComponent)
        if tool_reg is None:
            tool_reg = ToolRegistryComponent(
                tools={},
                handlers={},
            )
            world.add_component(entity_id, tool_reg)

        # Register tools from scripts/
        for tool_name, (schema, handler) in self.tools().items():
            tool_reg.tools[tool_name] = schema
            tool_reg.handlers[tool_name] = handler

        # Track installed skill in SkillComponent
        if skill_comp is None:
            skill_comp = SkillComponent(skills={})
            world.add_component(entity_id, skill_comp)

        from ecs_agent.components.definitions import SkillMetadata

        skill_comp.skills[self.name] = SkillMetadata(
            name=self.name,
            description=self.description,
            tool_names=list(self.tools().keys()),
            has_system_prompt=bool(self.system_prompt()),
            user_invocable=self.user_invocable,
            disable_model_invocation=self.disable_model_invocation,
            argument_hint=self.argument_hint,
            allowed_tools=self.allowed_tools,
            context=self.context,
            agent=self.agent,
            model=self.model,
            hooks=self.hooks,
            skill_dir_path=str(self.skill_dir_path),
            slash_command=self.slash_command,
        )
    def uninstall(self, world: World, entity_id: EntityId) -> None:
        """Uninstall skill by removing system prompt and tools.

        Args:
            world: World instance
            entity_id: Entity to uninstall skill from
        """
        # Remove system prompt added by this skill
        prompt_comp = world.get_component(entity_id, SystemPromptComponent)
        if prompt_comp:
            # Remove this skill's prompt from the component
            skill_prompt = self.system_prompt()
            prompt_comp.content = prompt_comp.content.replace(f"\n\n{skill_prompt}", "")
            prompt_comp.content = prompt_comp.content.replace(skill_prompt, "")

        # Remove tools registered by this skill
        tool_reg = world.get_component(entity_id, ToolRegistryComponent)
        if tool_reg:
            for tool_name in self.tools().keys():
                tool_reg.tools.pop(tool_name, None)
                tool_reg.handlers.pop(tool_name, None)

        # Remove from SkillComponent tracking
        skill_comp = world.get_component(entity_id, SkillComponent)
        if skill_comp:
            skill_comp.skills.pop(self.name, None)
