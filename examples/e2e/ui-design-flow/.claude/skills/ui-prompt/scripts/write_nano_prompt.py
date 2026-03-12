#!/usr/bin/env python3
"""
Skill output script for ui-prompt's write_nano_prompt tool.

Reads JSON from stdin with schema:
{
  "page": "string",
  "prompt": "string"
}

Writes nano-banana prompts to ui-design/nano-banana-prompts.md
"""

import json
import sys
from pathlib import Path


def main() -> None:
    try:
        # Read JSON input from stdin
        data = json.load(sys.stdin)

        # Validate required fields
        if "page" not in data:
            print("Error: Missing required field 'page'", file=sys.stderr)
            sys.exit(1)
        if "prompt" not in data:
            print("Error: Missing required field 'prompt'", file=sys.stderr)
            sys.exit(1)

        # Resolve output directory using absolute path
        base_dir = Path(__file__).parent.parent.parent.parent.parent
        output_dir = base_dir / "ui-design"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build markdown content
        markdown_content = f"""# Nano Banana Prompt: {data["page"]}

{data["prompt"]}
"""

        # Write to nano-banana-prompts.md
        output_path = output_dir / "nano-banana-prompts.md"
        output_path.write_text(markdown_content, encoding="utf-8")

        # Print success message
        print(f"Successfully wrote nano prompt to ui-design/nano-banana-prompts.md")

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
