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

## [2026-04-13T13:54:00Z] Task 4: memory-boundary-logic
- The MemorySystem's search for `role="compaction"` messages (lines 20-36) was a transitional boundary marker that is no longer needed with `CurrentCompactionSummaryComponent`.
- Implementation replaced the synthetic message boundary search with a `world.get_component(entity_id, CurrentCompactionSummaryComponent)` check.
- When `CurrentCompactionSummaryComponent` is present: preserve system message (if exists) + ALL remaining non-system messages without any truncation, even if count exceeds `max_messages`. This protects post-compaction conversation history from loss.
- When `CurrentCompactionSummaryComponent` is absent: apply trailing-window truncation as before (keep last N messages + system if present). This ensures backward compatibility for entities that have never undergone compaction.
- TDD approach: wrote two new failing tests first (with/without summary component), then modified MemorySystem logic, then verified all 8 memory tests pass.
- No synthetic boundary message is reintroduced; the compaction boundary is now entirely encoded by: (1) system message position, (2) presence of CurrentCompactionSummaryComponent.
- Multi-entity truncation still works correctly; the old test `test_truncation_preserves_latest_compaction_message_and_following_history` now passes via the no-summary path (applies trailing-window behavior).

## [2026-04-13T15:22:00Z] Task 5: provider-special-casing-removal
- All three adapters (OpenAI Chat, OpenAI Responses, Anthropic) had identical `_COMPACTION_SENTINEL = "[COMPACTION SUMMARY]\n"` constants that are now removed.
- Removed branches: OpenAI Chat (lines 228-245), OpenAI Responses (lines 306-319), Anthropic (lines 100-112) — all checked for `msg.role == "compaction"`.
- Old sentinel approach encoded compaction as a user message with a hardcoded prefix string; this prevented the system from seeing the summary as system prompt context.
- New approach: System prompt containing XML summary is passed directly through the adapter without any special handling — the prompt render system ensures the XML is already embedded before provider receives messages.
- Tests rewritten to assert on actual request bodies:
  - OpenAI Chat: verified XML summary appears in messages[0]["role"] == "system"
  - OpenAI Responses: verified XML summary in `instructions` field only, NOT in `input` items
  - Anthropic: verified XML summary in returned `system` string from build_messages
- All 42 provider tests pass; no regression in standard message conversion for user/assistant/system/tool roles.
- mypy strict typecheck passes with no issues.

