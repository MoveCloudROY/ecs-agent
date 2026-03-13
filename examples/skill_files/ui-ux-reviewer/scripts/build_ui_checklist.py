import json
import sys


def main() -> None:
    raw = sys.stdin.read().strip()
    payload = json.loads(raw) if raw else {}

    page_type = payload.get("page_type", "page")
    emphasis = payload.get("emphasis", "accessibility")

    checklist = [
        f"{page_type}: heading + CTA hierarchy is clear",
        f"{page_type}: spacing scale is consistent",
        f"{page_type}: {emphasis} checks are complete",
        f"{page_type}: mobile viewport has no horizontal overflow",
    ]
    print("\n".join(checklist))


if __name__ == "__main__":
    main()
