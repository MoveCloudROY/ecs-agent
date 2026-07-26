"""Entrypoint: ``python -m examples.e2e.plan_and_task.debug``."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from examples.e2e.plan_and_task.debug.cli import main

if __name__ == "__main__":
    main()
