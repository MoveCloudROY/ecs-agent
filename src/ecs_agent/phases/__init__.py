"""Phase-graph runtime: explicit, auditable phase transitions for agents."""

from ecs_agent.phases.api import (
    HISTORY_LIMIT,
    InvalidPhaseTransitionError,
    PhaseError,
    PhaseGraphMismatchError,
    PhaseIntegrityError,
    ResumeReport,
    advance,
    allowed_targets,
    bind_phase_graph,
    force,
    is_terminal,
    latest_verdicts,
    record_approval,
    resume_phase_graph,
)
from ecs_agent.phases.contracts import ApprovalGate, PhaseGraph, PhaseSpec, build_graph
from ecs_agent.phases.prompt_provider import (
    PHASE_PROMPT_PLACEHOLDER,
    PhasePromptPlaceholderProvider,
)

__all__ = [
    "HISTORY_LIMIT",
    "PHASE_PROMPT_PLACEHOLDER",
    "ApprovalGate",
    "InvalidPhaseTransitionError",
    "PhaseError",
    "PhaseGraph",
    "PhaseGraphMismatchError",
    "PhaseIntegrityError",
    "PhasePromptPlaceholderProvider",
    "PhaseSpec",
    "ResumeReport",
    "advance",
    "allowed_targets",
    "bind_phase_graph",
    "build_graph",
    "force",
    "is_terminal",
    "latest_verdicts",
    "record_approval",
    "resume_phase_graph",
]
