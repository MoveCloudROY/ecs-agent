"""Scratchbook filesystem service."""

from ecs_agent.scratchbook.artifact_registry import (
    ArtifactDescriptor,
    ArtifactKind,
    ArtifactPersistResult,
    ArtifactRegistry,
)
from ecs_agent.scratchbook.index import (
    CorruptedIndexEntryError,
    IndexEntry,
    ScratchbookIndexer,
    compute_content_hash,
)
from ecs_agent.scratchbook.prompt_definition import (
    ScratchbookArtifactPromptDef,
    ScratchbookPromptConfig,
    normalize_artifact_type_id,
)
from ecs_agent.scratchbook.prompt_provider import ScratchbookPromptPlaceholderProvider
from ecs_agent.scratchbook.service import ScratchbookService
from ecs_agent.scratchbook.tool_sink import ToolResultsSink

__all__ = [
    "ArtifactDescriptor",
    "ArtifactKind",
    "ArtifactPersistResult",
    "ArtifactRegistry",
    "CorruptedIndexEntryError",
    "IndexEntry",
    "normalize_artifact_type_id",
    "ScratchbookArtifactPromptDef",
    "ScratchbookPromptConfig",
    "ScratchbookIndexer",
    "ScratchbookPromptPlaceholderProvider",
    "ScratchbookService",
    "compute_content_hash",
    "ToolResultsSink",
]
