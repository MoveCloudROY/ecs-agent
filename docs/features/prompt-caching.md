# Prompt Caching

Provider prompt caches serve the repeated prefix of consecutive LLM calls at
~10% of the normal input price. They are strict **prefix caches**: request
N+1 reads cached tokens only for the leading bytes it shares with an earlier
request. One rewritten message mid-history forfeits everything after it.

The framework therefore maintains one invariant on every outbound path:

> **Append-only prompts.** Within a session, each model call's rendered
> prompt must be a byte-prefix of the next call's prompt, except for content
> that genuinely changed (a phase transition, a compaction rewrite).

## How each adapter caches

| Wire format | Mechanism | Usage fields |
|---|---|---|
| Anthropic Messages | Explicit `cache_control` breakpoints (see below) | `cache_creation_input_tokens`, `cache_read_input_tokens` |
| OpenAI Chat Completions | Automatic prefix caching (≥1024 tokens, 128-token granularity) | `prompt_tokens_details.cached_tokens` |
| OpenAI Responses | Automatic prefix caching over `instructions` + `input` | `input_tokens_details.cached_tokens` |

`accounting.normalization` maps all of these onto `UsageRecord`
(`cache_read_tokens` / `cache_creation_tokens` / `cached_input_tokens`), and
`compute_cache_stats()` derives a per-call `hit_rate`.

### Anthropic breakpoints

`AnthropicMessagesAdapter` emits up to 4 `cache_control` markers per request
(the API maximum), budgeted as:

1. the last tool definition (tools render first; one marker caches the whole
   static tool block),
2. every system entry flagged `Message.cache_control=True` (the cache-stable
   system prefix from the stable/volatile split),
3. the last content block of the last message (incremental conversation
   caching),
4. when the history exceeds one lookback window and the budget allows, one
   **ladder marker** at the last message boundary within ~20 content blocks
   of the tail.

The ladder marker exists because Anthropic finds a previous cache entry by
checking at most ~20 content blocks behind each breakpoint. Every
`tool_use` and `tool_result` is its own block, so a wide parallel tool batch
(e.g. 12 concurrent calls ⇒ ~25 new blocks) would strand the previous
request's tail entry beyond the window of a single trailing marker — a
silent full-history miss on anthropic-proper endpoints. The ladder keeps an
entry reachable for turns appending up to roughly twice the window.

Disable everything with `ProviderConfig(enable_prompt_caching=False)` (or
`Model(..., enable_prompt_caching=False)`) to restore the plain request
shape.

## What keeps prompts append-only

These behaviours exist specifically to protect the invariant:

- **System prompt stable/volatile split** — volatile placeholders
  (`_phase_prompt`, `_chat_history_summary_xml`) are relocated to a separate
  system message after the cache-stable prefix. Within a phase the bytes are
  constant; a phase transition or compaction refresh intentionally rewrites
  the volatile message and pays one full-history miss (see Costs below).
- **Rendered user prompts freeze on turn advance** — trigger injection and
  `action="script"` replacement substitute the last user message at call
  time. When a newer user message arrives,
  `UserPromptNormalizationSystem` persists the rendered text into the old
  slot, so the bytes the model saw never revert to the raw command text.
- **Context pool rides a trailing message** — pool entries and slash-skill
  context are appended as one final `user` message (never merged into an
  earlier message) and persisted into the conversation when the reservation
  commits after a successful call.
- **Tool results are append-only** — `ToolExecutionSystem` lands results as
  new `tool` messages; the tool-sink and overflow-cache rewrites happen
  before the message is first sent.
- **Subagent completion notifications** append `role="user"` messages at the
  session tail.

## Verifying and experimenting

### Offline prefix audit (deterministic, CI)

`tests/test_prompt_cache_prefix_audit.py` drives production systems
(`ReasoningSystem`, `ToolExecutionSystem`, `PromptContextCollectorSystem`,
`UserPromptNormalizationSystem`) with a scripted `FakeModel`, captures every
outbound call, renders it through the real adapters for all three wire
formats, and asserts the append-only property call-over-call. Run with
`-s` to see byte-level divergence diagnostics
(`tests/cache_audit/harness.py` is the reusable audit toolkit).

### Live experiments (env-gated)

`tests/live/test_prompt_cache_experiment_live.py` replays the captured
sequences against real endpoints and prints per-call tables
(`prompt` / `cached_read` / `cache_write` / `hit_rate`):

```bash
# Anthropic side (ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL / ANTHROPIC_MODEL)
uv run pytest tests/live/test_prompt_cache_experiment_live.py -k anthropic -s

# OpenAI side (LLM_API_KEY / LLM_BASE_URL / LLM_RESPONSES_BASE_URL / LLM_MODEL)
uv run pytest tests/live/test_prompt_cache_experiment_live.py -k openai -s
```

Scenarios: the clean agentic tool loop, the context-pool loop, an Anthropic
breakpoint-lookback probe (narrow vs wide tool batches), and
Responses-endpoint cache-support probes. `LLM_CACHE_WRITE_LAG_SECONDS`
(default 150) bounds the polling budget for gateways whose cache writes
commit asynchronously; each warm step prints both the first-attempt read and
the final poll.

Reference numbers measured against an aggregator gateway (2026-07-16):
clean loop warm hit rate ≈ 96% (anthropic side, deepseek backend) and ≈ 86%
(openai chat side); the context-pool loop ≈ 91% on the anthropic side.

## Cost model of intentional invalidations

- **Phase transition / compaction refresh** — the volatile system message
  sits ahead of the history, so flipping it re-processes the whole
  conversation once. This is the intended price of changing the agent's
  standing instructions; keep phase prompts byte-stable within a phase.
  (Phase fingerprints are content-hashed: transitions between phases with
  identical prompt text do not invalidate anything.)
- **Trim / compaction rewrites of old messages** — destructive by design;
  they trade a one-time cache miss for a permanently smaller prompt.

## Gateway-specific behaviours worth knowing

Observed against an aggregator gateway (`api.rutaceae.com`, 2026-07-16);
behaviour on first-party endpoints may differ:

- **anthropic format (deepseek backend):** caching is automatic
  message-prefix matching — `cache_creation_input_tokens` is always 0, reads
  are quantized (~256-token steps), reads appear immediately, and the
  ~20-block lookback limitation of anthropic-proper `cache_control` does not
  apply. Thinking-mode models **require** assistant history to carry
  thinking blocks back (`Message.reasoning_content` is replayed by the
  adapter; synthetic replays must supply one).
- **openai chat:** async cache writes (~1–2 min before a warm read
  succeeds); back-to-back identical calls in a fast loop read 0.
- **openai responses:** identical repeats *without* tools hit (~95%), but
  tool-bearing requests (function tools + `function_call` /
  `function_call_output` items — every agentic call) reported 0 cached
  tokens even for byte-identical repeats. For cache-sensitive agentic
  workloads on this gateway, prefer `openai_chat_completions`.
