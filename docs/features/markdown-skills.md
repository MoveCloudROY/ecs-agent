# Markdown Skills

Load skills from `.claude/skills/<name>/SKILL.md` format with YAML frontmatter, automatic tool discovery from `scripts/` directory, and lazy two-phase loading.

## Overview

Markdown Skills provide:
- **SKILL.md Format**: Define skills in markdown files with YAML frontmatter.
- **Lazy Two-Phase Loading**: Index metadata first, activate body and tools on demand.
- **Automatic Tool Discovery**: Python scripts in `scripts/` become tools.
- **Claude-Compatible Substitutions**: Support for `$ARGUMENTS`, `$N`, and environment variables.
- **Invocation Control**: Fine-grained control over who can invoke the skill.
- **Path Traversal Protection**: Securely resolve paths within the skill directory.
- **Skip+Warn Policy**: Invalid skills are skipped with a warning instead of aborting.
- **Automatic Tool Discovery**: Python scripts in `scripts/` become tools
- **System Prompt Integration**: Markdown body becomes agent system prompt
- **File-Based Auto-Discovery**: Load skills from directory structure

## SKILL.md Format

### Basic Structure

```markdown
---
name: my-skill
description: A brief description of what this skill does
---

# Skill Title

Markdown content here becomes the system prompt.

## Instructions

Additional guidance for the agent.

## Tools

Tools are discovered from the scripts/ directory.
```

### YAML Frontmatter

Required fields:
- `name`: Skill identifier (regex: `[a-z0-9-]{1,64}`)
- `description`: Brief summary of skill purpose

Optional fields:
- `user-invocable`: Whether the user can invoke via slash command (default: `true`)
- `disable-model-invocation`: Whether to exclude from model auto-invoke (default: `false`)
- `argument-hint`: Hint text shown for arguments (e.g., `<url>`)
- `allowed-tools`: List of tool names this skill is allowed to use
- `context`: Context routing identifier
- `agent`: Agent routing identifier
- `model`: Specific model to use for this skill
- `hooks`: Dictionary of lifecycle hook configurations

Example:

```yaml
---
name: data-analyst
description: Analyze datasets and generate statistical insights
user-invocable: true
argument-hint: "<csv-file-path>"
allowed-tools:
  - read_file
  - python_repl
---
```

### Substitutions

The markdown body and scripts can use Claude-compatible substitution variables:

- `$ARGUMENTS`: The entire whitespace-separated arguments string.
- `$ARGUMENTS[N]`: The Nth word from arguments (0-indexed).
- `$N`: Shorthand for Nth argument (1-indexed: `$1`, `$2`, etc.).
- `${CLAUDE_SESSION_ID}`: Resolves to the current session ID.
- `${CLAUDE_SKILL_DIR}`: Resolves to the absolute path of the skill directory.

### Markdown Body

The entire markdown body (everything after the frontmatter) becomes the system prompt. You can use substitutions here to customize instructions based on input:

```markdown
---
name: code-reviewer
description: Review code for quality and best practices
argument-hint: "<file-path>"
---

# Code Review Specialist

You are reviewing: $1

Please analyze the file at ${CLAUDE_SKILL_DIR}/$1 for:
...
```

```markdown
---
name: code-reviewer
description: Review code for quality and best practices
---

# Code Review Specialist

You are an expert code reviewer. Your role is to:

1. Identify bugs and potential issues
2. Suggest improvements for readability
3. Check for security vulnerabilities
4. Ensure code follows best practices

## Review Checklist

- [ ] Code is well-documented
- [ ] Edge cases are handled
- [ ] Tests are present and comprehensive
- [ ] No security vulnerabilities
```

## Tool Discovery

### scripts/ Directory

Create a `scripts/` directory alongside `SKILL.md`:

```
.claude/skills/my-skill/
├── SKILL.md
└── scripts/
    ├── analyze_data.py
    ├── generate_chart.py
    └── export_report.py
```

### Tool Script Format

Each `.py` file should:
1. Define a `TOOL_SCHEMA` dict (JSON Schema format)
2. Define a `TOOL_HANDLER` async function

Example `scripts/analyze_data.py`:

```python
import json
from typing import Any

# Tool schema in JSON Schema format
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_data",
        "description": "Analyze dataset and return statistical summary",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset": {
                    "type": "array",
                    "description": "List of numeric values",
                    "items": {"type": "number"},
                },
            },
            "required": ["dataset"],
        },
    },
}

# Async tool handler
async def TOOL_HANDLER(arguments: dict[str, Any]) -> str:
    """Analyze dataset and return summary."""
    dataset = arguments["dataset"]
    
    mean = sum(dataset) / len(dataset)
    min_val = min(dataset)
    max_val = max(dataset)
    
    return json.dumps({
        "mean": mean,
        "min": min_val,
        "max": max_val,
        "count": len(dataset),
    })
```

### Tool Naming

Tool names are derived from:
1. `TOOL_SCHEMA["function"]["name"]` if present
2. Otherwise, filename without `.py` extension

## Usage

Markdown skills support a **two-phase loading** lifecycle to remain token-efficient:

1.  **Index**: Load metadata (name, description, tool names) into the `SkillComponent`. No tools are registered yet, and the body is not loaded.
2.  **Activate**: Load the markdown body into the `SystemPromptComponent` and register tools into the `ToolRegistryComponent`.

### Direct Installation

For manual use, `manager.install()` performs both index and activation:

```python
from pathlib import Path
from ecs_agent.skills.markdown_skill import MarkdownSkill
from ecs_agent.skills.manager import SkillManager

# Load skill object
skill = MarkdownSkill(skill_path=Path(".claude/skills/my-skill/SKILL.md"))

# Install (Index + Activate)
manager = SkillManager()
manager.install(world, entity, skill)
```

### Lazy Discovery

When using `DiscoveryManager`, skills are indexed but not activated until needed:

```python
from ecs_agent.skills.discovery import DiscoveryManager

discovery = DiscoveryManager()
# Indexed only: metadata available, tools/prompt NOT yet loaded
await discovery.auto_discover_and_install(
    world, entity, manager, directories=[Path(".claude/skills")]
)

# Activate later when needed
manager.activate(world, entity, "my-skill")
```


### Applying to an Entity

```python
from ecs_agent import World
from ecs_agent.components import LLMComponent, ToolRegistryComponent

world = World()
entity = world.create_entity()

# Apply system prompt
world.add_component(
    entity,
    LLMComponent(
        provider=your_provider,
        model="gpt-4o",
        system_prompt=skill.system_prompt(),
    ),
)

# Register tools
tools_dict = skill.tools()
world.add_component(
    entity,
    ToolRegistryComponent(tools=tools_dict),
)
```

```python
from ecs_agent.skills.discovery import discover_markdown_skills
from pathlib import Path

# Discover all SKILL.md files (returns list of MarkdownSkill objects)
skills = discover_markdown_skills(directories=[Path(".claude/skills")])

for skill in skills:
    print(f"Found skill: {skill.name}")

## Progressive Disclosure

Skills support tiered information disclosure:

### Tier 1: Metadata

Quick overview without reading full file:

```python
skill.name  # From frontmatter
skill.description  # From frontmatter
```

### Tier 2: System Prompt

Full instructions:

```python
skill.system_prompt()  # Full markdown body
```

### Tier 3: Tools

Executable capabilities:

```python
skill.tools()  # Discovered from scripts/
```

This allows agents to:
1. Browse available skills by metadata (Tier 1)
2. Load instructions only when needed (Tier 2)
3. Activate tools only when skill is in use (Tier 3)

## Example: Complete Skill

### Directory Structure

```
.claude/skills/web-scraper/
├── SKILL.md
└── scripts/
    ├── fetch_url.py
    └── extract_text.py
```

### SKILL.md

```markdown
---
name: web-scraper
description: Fetch and extract content from web pages
---

# Web Scraping Specialist

You can fetch web pages and extract structured content.

## Available Operations

Use the `fetch_url` tool to retrieve HTML content.
Use the `extract_text` tool to parse and clean text.

## Guidelines

- Always check robots.txt before scraping
- Respect rate limits
- Handle errors gracefully
```

### scripts/fetch_url.py

```python
import httpx
from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch HTML content from URL",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
            },
            "required": ["url"],
        },
    },
}

async def TOOL_HANDLER(arguments: dict[str, Any]) -> str:
    url = arguments["url"]
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text
```

### Usage

```python
skill = MarkdownSkill(skill_path=Path(".claude/skills/web-scraper/SKILL.md"))

world.add_component(
    entity,
    LLMComponent(
        provider=provider,
        model="gpt-4o",
        system_prompt=skill.system_prompt(),
    ),
)

tools = skill.tools()
world.add_component(
    entity,
    ToolRegistryComponent(tools=tools),
)
```

The agent now has web scraping capabilities with appropriate context.

### Path Traversal Protection

Markdown skills include protection against path traversal when resolving files:

```python
# Safe resolution within skill_dir
path = skill.resolve_supporting_path("data/config.json")

# Raises ValueError (traversal detected)
path = skill.resolve_supporting_path("../../../etc/passwd")
```

### Skip+Warn Policy

The discovery system follows a "skip and warn" policy. If a `SKILL.md` has invalid YAML, a malformed name, or is missing required fields:
1.  A warning is logged with the file path.
2.  The specific skill is skipped.
3.  The rest of the discovery process continues normally.


Tool scripts execute in isolated environments:

```python
skill = MarkdownSkill(
    skill_path=Path(".claude/skills/my-skill/SKILL.md"),
    sandbox_timeout=30.0,  # Timeout for tool execution
)
```

This prevents:
- Infinite loops
- Excessive resource usage
- Hanging operations

## Best Practices

### 1. Clear Descriptions

Write concise, descriptive frontmatter:

```yaml
# Good
name: sentiment-analyzer
description: Analyze text sentiment (positive/negative/neutral) with confidence scores

# Bad
name: analyzer
description: Does stuff
```

### 2. Structured System Prompts

Use markdown headers for organization:

```markdown
## Role

You are a [role description].

## Capabilities

- Capability 1
- Capability 2

## Constraints

- Constraint 1
- Constraint 2
```

### 3. Tool Error Handling

Handle errors in tool handlers:

```python
async def TOOL_HANDLER(arguments: dict[str, Any]) -> str:
    try:
        result = await process(arguments)
        return json.dumps({"status": "success", "data": result})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})
```

### 4. Type Annotations

Use JSON Schema types precisely:

```python
"parameters": {
    "type": "object",
    "properties": {
        "count": {"type": "integer", "minimum": 1, "maximum": 100},
        "format": {"type": "string", "enum": ["json", "csv", "text"]},
    },
    "required": ["count"],
}
```

## See Also

- [Skills System](./skills.md) — Skill protocol and manager
- [Tool Discovery](./tool-discovery.md) — Auto-discovery patterns
- [Built-in Tools](./builtin-tools.md) — Standard library tools
