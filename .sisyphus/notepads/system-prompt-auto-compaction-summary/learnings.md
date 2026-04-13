# Learnings

## [2026-04-13] Session Start
- Project uses `uv` as package manager, strict mypy (python 3.11), asyncio_mode=auto for pytest
- All components use `@dataclass(slots=True)`
- MessageRole is a Literal type in types.py
- CompactionSystem stores summary as Message(role="compaction", content="Previous conversation summary: X")
- MemorySystem relies on role="compaction" as a boundary marker (lines 20-36 of systems/memory.py)
- SystemPromptRenderSystem uses fingerprint-based render cache
- RenderedSystemPromptComponent is frozen on first render, reused until cache key changes
- ConversationArchiveComponent.archived_summaries is the append-only history store
- XML tag name must be exactly: <chat_history_summary>...</chat_history_summary>
- Empty-state rendering uses the same XML block with empty content, NOT omission
- Content is escaped at render time; stored as plain text
- No -m live marker; live tests run when LLM_API_KEY env var is present

## [2026-04-13T13:31:00Z] Task 1: state-migration
- `CurrentCompactionSummaryComponent` belongs beside `ConversationArchiveComponent`: archive remains append-only history while current summary is separate latest-state storage.
- The safest migration point is `WorldSerializer.from_dict()` plus `ConversationComponent` normalization: derive the latest legacy compaction summary from incoming messages, then strip those legacy messages before attaching the restored conversation.
- Legacy migration must only consume messages whose content starts with `"Previous conversation summary: "`; the last matching legacy message wins when multiple compaction messages exist.
- Narrowing canonical `MessageRole` without touching current compaction systems works by separating canonical roles from legacy acceptance in `Message.role` typing, rather than pretending `compaction` is still canonical everywhere.

## [2026-04-13T13:41:00Z] Task 2: summary-placeholder
- `CompactionSummaryPlaceholderProvider` should emit nothing for entities without `CompactionConfigComponent`; that keeps existing non-compaction cache keys stable while still fingerprinting compaction-enabled entities.
- Legacy prompt normalization is safest as an in-memory synthetic `SystemPromptConfigSpec` path: reuse the stored rendered snapshot's `_legacy_template` on re-render so summary changes do not permanently overwrite the original legacy prompt template.
- Appending `\n${_chat_history_summary_xml}` only when the placeholder is absent avoids double insertion while guaranteeing compaction-enabled prompts end with the XML block.

## [2026-04-13T13:50:00Z] Task 3: compaction-state-switch
- `CompactionSystem.process()` can switch carriers cleanly by filtering legacy `role="compaction"` messages out of the working set before strategy selection, then rebuilding `ConversationComponent.messages` from only the preserved system message plus retained post-compaction turns.
- Repeated compaction works best by prepending a plain-text synthetic user message containing the existing `CurrentCompactionSummaryComponent.summary` before newly selected messages, which preserves summary continuity without folding XML into the summarization prompt.
- Prompt cache invalidation belongs in compaction itself: after writing a new `CurrentCompactionSummaryComponent`, removing `RenderedSystemPromptComponent` ensures the next system-prompt render picks up the new summary XML while leaving prompt-render helpers unchanged.
