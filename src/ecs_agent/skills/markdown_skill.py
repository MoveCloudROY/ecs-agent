"""Markdown-based skill parser.

Loads skills from SKILL.md files with YAML frontmatter.
"""

import asyncio
import json
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


class MarkdownSkill:
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
        """Initialize MarkdownSkill by parsing SKILL.md.

        Args:
            skill_path: Path to SKILL.md file
            sandbox_timeout: Timeout for subprocess tool execution
        """
        self._skill_path = skill_path
        self._sandbox_timeout = sandbox_timeout
        self._parse_skill_file()

    def _parse_skill_file(self) -> None:
        """Parse YAML frontmatter and markdown body."""
        content = self._skill_path.read_text()

        # Check if frontmatter exists (starts with ---)
        if content.startswith("---"):
            # Split on closing ---
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_text = parts[1].strip()
                body = parts[2].strip()
            else:
                # Malformed frontmatter, treat as no frontmatter
                frontmatter_text = ""
                body = content.strip()
        else:
            # No frontmatter
            frontmatter_text = ""
            body = content.strip()

        # Parse YAML frontmatter
        if frontmatter_text:
            try:
                metadata: dict[str, Any] = yaml.safe_load(frontmatter_text) or {}
            except yaml.YAMLError:
                logger.warning(
                    "markdown_skill_invalid_yaml",
                    skill_path=str(self._skill_path),
                )
                metadata = {}
        else:
            metadata = {}

        # Extract name and description (use defaults if missing)
        self._name: str = str(metadata.get("name", self._skill_path.stem))
        self._description: str = str(
            metadata.get("description", f"Skill from {self._skill_path.name}")
        )
        self._system_prompt_content = body

    @property
    def name(self) -> str:
        """Skill name from frontmatter or filename."""
        return self._name

    @property
    def description(self) -> str:
        """Skill description from frontmatter."""
        return self._description

    def system_prompt(self) -> str:
        """Return markdown body as system prompt."""
        return self._system_prompt_content

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

        Args:
            world: World instance
            entity_id: Entity to install skill on
        """
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
        skill_comp = world.get_component(entity_id, SkillComponent)
        if skill_comp is None:
            skill_comp = SkillComponent(skills={})
            world.add_component(entity_id, skill_comp)

        from ecs_agent.components.definitions import SkillMetadata
        skill_comp.skills[self.name] = SkillMetadata(
            name=self.name,
            description=self.description,
            tool_names=list(self.tools().keys()),
            has_system_prompt=bool(self.system_prompt()),
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
