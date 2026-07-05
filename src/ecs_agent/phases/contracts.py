"""Typed phase-graph contracts: pure-data authoring and validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ApprovalGate:
    """Maps review verdicts recorded in a phase to target phases (None = stay)."""

    verdicts: Mapping[str, str | None]

    def __post_init__(self) -> None:
        if not self.verdicts:
            raise ValueError("ApprovalGate requires at least one verdict mapping")
        for verdict in self.verdicts:
            if not verdict:
                raise ValueError("ApprovalGate verdict keys must be non-empty strings")
        object.__setattr__(self, "verdicts", MappingProxyType(dict(self.verdicts)))


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    """Declares one phase: prompts per agent key, adjacency, and entry effects."""

    phase_id: str
    prompts: Mapping[str, str | Path] = field(default_factory=dict)
    to: tuple[str, ...] = ()
    tools: tuple[str, ...] | None = None
    approval: ApprovalGate | None = None
    on_resume: str | None = None
    terminal: bool = False

    def __post_init__(self) -> None:
        if not self.phase_id:
            raise ValueError("PhaseSpec requires a non-empty phase_id")
        if self.terminal and self.to:
            raise ValueError(
                f"terminal phase {self.phase_id!r} must not declare outgoing targets"
            )
        if self.terminal and self.approval is not None:
            raise ValueError(
                f"terminal phase {self.phase_id!r} must not declare an approval gate"
            )
        if not self.terminal and not self.to:
            raise ValueError(
                f"non-terminal phase {self.phase_id!r} must declare at least one target"
            )
        if len(self.to) != len(set(self.to)):
            raise ValueError(f"phase {self.phase_id!r} declares duplicate targets")
        object.__setattr__(self, "prompts", MappingProxyType(dict(self.prompts)))


@dataclass(frozen=True, slots=True)
class PhaseGraph:
    """Validated phase graph. Build via build_graph(); do not construct directly.

    All mapping fields are read-only views (MappingProxyType) over private
    copies: post-build mutation raises TypeError, and mutating the caller's
    original inputs after build_graph() cannot affect the graph.
    """

    graph_id: str
    initial: str
    phases_by_id: Mapping[str, PhaseSpec]
    structure_hash: str
    manages_tools: bool = False


def build_graph(
    graph_id: str,
    *,
    initial: str,
    phases: list[PhaseSpec] | tuple[PhaseSpec, ...],
) -> PhaseGraph:
    """Validate phase specs and produce an immutable graph with a structural hash."""

    if not graph_id:
        raise ValueError("build_graph requires a non-empty graph_id")
    specs = tuple(phases)
    if not specs:
        raise ValueError("build_graph requires at least one phase")

    ids = [spec.phase_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate phase_id values are not allowed")
    phases_by_id = {spec.phase_id: spec for spec in specs}
    if initial not in phases_by_id:
        raise ValueError(f"Initial phase {initial!r} was not found in phases")

    for spec in specs:
        for target in spec.to:
            if target not in phases_by_id:
                raise ValueError(
                    f"Target {target!r} from phase {spec.phase_id!r} was not found in phases"
                )
        if spec.on_resume is not None and spec.on_resume not in phases_by_id:
            raise ValueError(
                f"on_resume target {spec.on_resume!r} from phase {spec.phase_id!r} "
                "was not found in phases"
            )
        if spec.approval is not None:
            for verdict, gate_target in spec.approval.verdicts.items():
                if gate_target is not None and gate_target not in spec.to:
                    raise ValueError(
                        f"approval verdict {verdict!r} in phase {spec.phase_id!r} targets "
                        f"{gate_target!r}, which is not in its declared targets"
                    )

    return PhaseGraph(
        graph_id=graph_id,
        initial=initial,
        phases_by_id=MappingProxyType(dict(phases_by_id)),
        structure_hash=_structure_hash(graph_id, initial, specs),
        manages_tools=any(spec.tools is not None for spec in specs),
    )


def _structure_hash(graph_id: str, initial: str, specs: tuple[PhaseSpec, ...]) -> str:
    # Prompt text/paths are deliberately excluded: editing prompts must never
    # invalidate restored phase state. Only structural drift is detected.
    payload = {
        "graph_id": graph_id,
        "initial": initial,
        "phases": sorted(
            (
                {
                    "id": spec.phase_id,
                    "to": sorted(spec.to),
                    "tools": sorted(spec.tools) if spec.tools is not None else None,
                    "approval": dict(spec.approval.verdicts) if spec.approval else None,
                    "on_resume": spec.on_resume,
                    "terminal": spec.terminal,
                }
                for spec in specs
            ),
            key=lambda entry: str(entry["id"]),
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
