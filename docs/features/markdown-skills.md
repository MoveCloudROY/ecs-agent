# Markdown Skills

Load skills from `.claude/skills/<name>/SKILL.md` format with YAML frontmatter, automatic tool discovery from `scripts/` directory, and progressive disclosure.

## Overview

Markdown Skills provide:
- **SKILL.md Format**: Define skills in markdown files with YAML frontmatter
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
- `name`: Skill identifier (kebab-case recommended)
- `description`: Brief summary of skill purpose

Example:

```yaml
---
name: data-analyst
description: Analyze datasets and generate statistical insights
---
```

### Markdown Body

The entire markdown body (everything after the frontmatter) becomes the system prompt:

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

### Loading a Skill

```python
from pathlib import Path
from ecs_agent.skills.markdown_skill import MarkdownSkill

# Load skill from SKILL.md
skill = MarkdownSkill(skill_path=Path(".claude/skills/my-skill/SKILL.md"))

# Access metadata
print(skill.name)  # "my-skill"
print(skill.description)  # "A brief description..."

# Get system prompt (full markdown body)
system_prompt = skill.system_prompt()

# Get tools (auto-discovered from scripts/)
tools = skill.tools()  # dict[str, tuple[ToolSchema, ToolHandler]]
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

### File-Based Auto-Discovery

Discover all skills in a directory:

```python
from ecs_agent.skills.discovery import discover_skills_from_directory

# Discover all SKILL.md files
skills = discover_skills_from_directory(Path(".claude/skills"))

for skill in skills:
    print(f"Found skill: {skill.name}")
```

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

## Sandboxing

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
