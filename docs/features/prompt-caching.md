# Prompt Caching

Provider prompt caches can reuse a matching prefix from an earlier LLM
request. Pricing and cache availability depend on the provider and model.
Changing an early message reduces the reusable prefix; an unchanged prefix
makes reuse possible but does not guarantee a server-side hit.

The framework's cache-preservation contract for append-only conversation paths is:

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

## Measured cache behaviour and limits

The latest controlled comparison is the
[2026-10-18 DeepSeek Responses audit](../rfc/2026-10-18-deepseek-context-cache-comparison.md),
using the same Agent/SDK scenario as the
[gateway branching audit](../rfc/2026-10-18-context-cache-branching-audit.md).
Each run covers 23 user turns and 25 HTTP requests, with long histories,
streaming, tool calls, nested snapshot branches, branch switching, disk
restore, three identical repeats, and two intentionally cold controls.

| Metric | Gateway `gpt-5.6-sol` | Direct DeepSeek `deepseek-flash` |
|---|---:|---:|
| Ideal prefix reuse baseline | 88.69% (computed after the run) | 88.72% (predictions recorded before sending) |
| Server-reported observed reuse | 57.71% (extension field) | 88.43% (standard field) |
| Standard `cached_tokens` / input tokens | 54.40% (billing allocation) | 88.43% |
| Warm requests with positive / zero cache | 16 / 7 | 23 / 0 |
| Three identical repeats | 68.70%, 77.86%, 99.45% observed | 99.12%, 99.12%, 99.12% |

All client prefix, protocol replay, branch isolation, and usage propagation
checks passed in both runs. DeepSeek's measured reuse is 0.29 percentage
points below its preregistered conditional baseline. This demonstrates that
the tested Agent path can achieve near-ideal reuse. The model and backend
also changed, so the comparison does not uniquely identify gateway code as
the cause of the earlier misses.

### Interpreting the metrics

- **Keep observed and billed cache separate.** In the October gateway
  sample, standard `input_tokens_details.cached_tokens` equals the gateway's
  billed-cache allocation. Its `linapi_cache_billing.observed_cached_tokens`
  field reports a different value. The audit retains both. DeepSeek uses
  the standard field and returned no such extension in this sample. These
  remain server-reported metrics, not independent measurements of KV cache.
- **Predict from the actual branch ancestor.** Before sending, the audit
  verifies the serialized prefix and stable request parameters, then uses
  the prior ancestor's input-token count as a conservative approximation.
  Exact repeats use the earlier identical request's input-token count;
  cold controls predict zero. Do not use another branch's immediately
  preceding request as the ancestor.
- **The baseline is conditional.** It assumes the matching cache has been
  written, remains available, and is reached by the request. Token alignment
  and server-side rendering can cause small differences. It is not a
  guaranteed hit rate or a universal block-size formula.
- **Zero and missing usage differ.** Explicit zero cached tokens is a
  measured miss. Absent cache details are unknown. An interrupted request
  can retain usage already received, or have unknown usage if none arrived;
  inspect invocation status and stream completeness before aggregating.
- **Snapshot branches are the tested path.** Native `ConversationTree`
  writeback, protocol-item preservation, and restore still have four strict
  expected-failure tests. The snapshot result does not establish that the
  native tree lifecycle is correct.

### Earlier gateway observations

The July 2026 records remain historical evidence in
`records/review/plan_task_cache_audit_2026-07-29/` and the July 28 endpoint
comparison. They observed both positive and zero cache reports during tool
loops; tool-bearing Responses requests did achieve cache hits. Missing
cache details on some Chat requests were also recorded.

Those runs did not establish the observed-versus-billed distinction used in
the October audit. Their reported ratios and apparent token steps should
not be treated as physical cache measurements, a universal cache block size,
or grounds to recommend one API format for all workloads. The samples also
do not uniquely distinguish routing, account pools, write visibility,
eviction, upstream processing, and reporting. See the linked audits for the
current measurements and their explicit limitations.
