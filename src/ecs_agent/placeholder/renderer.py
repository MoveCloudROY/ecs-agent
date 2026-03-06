"""Strict placeholder renderer using bounded grammar and deterministic snapshot inputs.

Uses Python's string.Template for safe substitution with:
- $identifier and ${identifier} syntax support
- $$ escape for literal dollar signs
- No eval, no code execution, no sandbox escapes
- Explicit error on missing placeholders (no silent fallback)
- Deterministic output from frozen snapshot dict
"""

from __future__ import annotations

from string import Template
from typing import Any

from ecs_agent.logging import get_logger

logger = get_logger(__name__)


class StrictPlaceholderRenderer:
    """Strict placeholder renderer using Python's string.Template.

    Guarantees:
    - Supports $identifier and ${identifier} syntax only
    - $$ escapes to literal $
    - Missing placeholders raise explicit KeyError
    - No evaluation of expressions or code
    - Deterministic output from frozen snapshot dict
    """

    def substitute(self, template: str, snapshot: dict[str, Any]) -> str:
        """Substitute placeholders in template using snapshot values.

        Args:
            template: Template string with $identifier or ${identifier} placeholders
            snapshot: Frozen dict mapping placeholder names to values

        Returns:
            String with all placeholders replaced by snapshot values

        Raises:
            KeyError: If any placeholder is not found in snapshot
        """
        try:
            # Use Template.substitute() which raises KeyError on missing placeholders
            # (NOT safe_substitute() which leaves unresolved)
            result = Template(template).substitute(snapshot)
            logger.info(
                "placeholder_substitution_success",
                template_length=len(template),
                snapshot_keys=len(snapshot),
            )
            return result
        except KeyError as e:
            logger.error(
                "placeholder_missing_key",
                missing_key=str(e),
                template_preview=template[:100],
            )
            raise
