#!/usr/bin/env python3
"""
Skill output script for ui-navigator's write_draft tool.

Reads JSON from stdin with schema:
{
  "title": "string",
  "content": "string (markdown)"
}

Writes markdown blueprint to ui-design/draft.md
"""

import json
import sys
from pathlib import Path


def main() -> None:
    try:
        # Read JSON input from stdin
        data = json.load(sys.stdin)

        # Validate required fields
        if "title" not in data:
            print("Error: Missing required field 'title'", file=sys.stderr)
            sys.exit(1)
        if "content" not in data:
            print("Error: Missing required field 'content'", file=sys.stderr)
            sys.exit(1)

        # Resolve output directory using absolute path
        base_dir = Path(__file__).parent.parent.parent.parent.parent
        output_dir = base_dir / "ui-design"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build markdown content
        markdown_content = f"""# {data["title"]}

{data["content"]}
"""

        # Write to draft.md
        output_path = output_dir / "draft.md"
        output_path.write_text(markdown_content, encoding="utf-8")

        # Print success message
        print(f"Successfully wrote draft to ui-design/draft.md")

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
