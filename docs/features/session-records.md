# Cross-session records

Session recording keeps an append-only execution history separate from `ConversationComponent`, which remains the mutable context sent to a model. Compaction, trimming, workflow resets, and checkpoint restoration do not delete committed records.

## Install and own the service

```python
from pathlib import Path

from ecs_agent import (
    BuiltinToolsSkill, SessionService, SQLiteSessionStore,
    append_conversation_message, install_session_recording,
)
from ecs_agent.components import ConversationComponent
from ecs_agent.core import World
from ecs_agent.scratchbook import ArtifactRegistry
from ecs_agent.skills import SkillManager
from ecs_agent.types import Message

root = Path("./agent-data")
world = World()
agent = world.create_entity()
world.add_component(agent, ConversationComponent(messages=[]))
SkillManager().install(world, agent, BuiltinToolsSkill().bind_workspace(str(root)))
service = SessionService(
    SQLiteSessionStore(root / "sessions.sqlite3"),
    artifact_registry=ArtifactRegistry(root),
)
install_session_recording(
    world, agent, service, workspace_id=str(root.resolve()), agent_name="main",
)
append_conversation_message(world, agent, Message(role="user", content="Review the test results"))
# Run the agent and settle its background tasks before closing:
service.close()
```

`InMemorySessionStore` implements the same contract for temporary runs. The installer imports the existing context once. Applications that append input directly should use `append_conversation_message`; replacing history should use `ecs_agent.history.replace_conversation_history`.

The root application owns the service. Child Worlds share it and receive independent `SessionRefComponent` identities. Finishing a child releases its writer without closing the shared database. Services, database connections, locks, and writer tokens are runtime dependencies and never enter a World checkpoint.

By default installation requires an existing `read_file` handler that supports session resources. It registers `sessions_list` and `sessions_search`; it does not overwrite custom tools or modify `PermissionComponent`. If an allowlist is present, authorize all three tools before installing. `register_tools=False` enables recording only, as used for child Worlds; their available tools still follow `InheritancePolicy.inherit_tools`.

## Discover, search, and read

```text
sessions_list(parent_session_id?, agent_name?, cursor?, limit=20)
sessions_search(query, session_id?, agent_name?, cursor?, limit=20)
read_file(file_path, offset=1, limit=0, line_numbers=False)
```

List and search return JSON strings containing `items` and `next_cursor`. Each list item includes its persistent session identity, parent/root relationship, agent name, task summary, lifecycle status, and a fixed-head `read_uri`. Empty sessions have a null URI. Search uses literal, case-sensitive substring matching, including Chinese and complete archived tool output, and returns the matching `record_id`, snippet, and record URI. Queries default to the selected branch. List/search pages default to 20 items and are capped at 100. A cursor pins the result set and filters; repeat the initial query to discover later appends.

For another agent's evidence, discover that agent first, then pass its exact `session_id` to `sessions_search`. An unfiltered search also includes the reader's own permitted session, so its prompt and search-call arguments can match the query. These are real records but usually irrelevant to the evidence review.

Pass the returned URI directly to `read_file`:

```text
session://<session_id>/transcript?head=<record_id>&v=1
session://<session_id>/records/<record_id>?v=1
session://<session_id>/records/<record_id>/tool-result?v=1
```

The transcript follows the selected parent path up to its fixed head. A tool record includes a `tool_result_uri` for its complete body. Resource URIs locate records; they never grant permission or accept filesystem paths. They remain stable while other agents append records. Unsupported versions, missing records/artifacts, changed artifact content, and invalid record ownership produce explicit errors rather than reading a different resource.

There is no separate model-visible `sessions_history` tool. All three read tools return their output inline and can run concurrently. Historical roles and messages remain data inside the current tool response; they are not replayed as messages in the reader's provider conversation.

## Line paging

Session reads return a JSON metadata line, a `---` separator, then body display lines. The metadata contains `read_uri`, `snapshot_seq`, `start_line`, `end_line`, `next_offset`, `has_more`, and `truncated`. Continue with the same URI and `offset=next_offset`; a null offset means end of resource.

| Parameter | Workspace file | Session resource |
|---|---|---|
| `offset` | 1-based file line | 1-based display line, excluding the metadata envelope |
| `limit=0` | All remaining lines | Default 80-line page |
| Explicit `limit` | Requested file lines | At most 500 display lines, subject to the 12,000-character total budget |
| `line_numbers` | Prefix file line numbers | Prefix display line numbers |
| Editing snapshot | Recorded for edit/write freshness | Never recorded |

Body `text:` lines contain JSON-encoded chunks of up to 512 characters. Decode and concatenate these strings to reconstruct the projected text exactly, including original newlines. This permits long single-line output to be continued without losing text. Paging stops at whole display-line boundaries. The source text itself remains unchanged in storage. Transcript traversal and artifact reads are incremental; an artifact's content hash is checked before its immutable view is used.

`write_file` and `edit_file` reject session URIs. An uninstalled service disables session reads but leaves normal workspace file reads available.

## Access policy

`SessionPolicyComponent` is host-controlled:

- `tree` (default): same workspace and same root task tree, including parents, children, and siblings.
- `self`: only the caller's session.
- `descendants`: the caller and its descendants.
- `shared_session_ids`: additional explicitly named sessions in the same workspace.

Each query and read resolves the actual `world + entity_id` from `ToolExecutionContext`, including inherited handlers. Every cursor and URI is reauthorized. A changed cursor policy requires rediscovery. Query parameters can only narrow the authorized scope. Reading records grants no task cancellation, resumption, or messaging authority.

The default model projection omits system configuration, provider-private replay, signatures, binary attachments, and recognizable credential fields/text. Raw stored records retain framework-observed data. The service policy controls this API; separately granting filesystem or shell access to the database/artifact directory is an application decision.

## Writes and recovery

Records have stable IDs, per-session sequence numbers, parent links, history identity/revision, run ID, kind, timestamp, payload, and an optional artifact reference. SQLite uses transactions, WAL, full synchronous commits, and exclusive writer claims. Different sessions may append concurrently. A second live writer cannot claim the same session. Closing releases claims; a dead process's claim can be reclaimed. A reused live PID is conservatively treated as live until the host resolves ownership.

Message commits and memory updates use synchronous operations with no async cancellation point between them. An uncertain commit is checked using its stable record ID before memory advances. Store-level retries use the same `RecordDraft`; a record ID cannot be reused for different content. Large tool output is published through `ArtifactRegistry` before a record refers to it. File publication and SQLite commits are separate operations; a failed commit may leave an orphan artifact, but cannot claim a missing artifact was successfully recorded.

Tool starts are committed before execution; complete raw results are recorded before observation events and before model-context externalization. A failed result commit stops advancement and clears pending dispatch, so another tick cannot repeat an external side effect. A start without a result means completion is unconfirmed. Run and delegation statuses preserve failure, timeout, cancellation, incomplete output, and successful completion distinctly. Terminal persistence failure changes the in-memory run outcome to an error.

Compaction records include the summary, previous head, covered message count, replacement count, and history revision. Originals remain queryable. Conversation-tree node IDs and parent links are preserved; selecting or reverting a branch records the selection and invalidates remote Responses chaining. Detached branch nodes do not replace the selected transcript.

After loading a checkpoint, rebind a service using the same SQLite file and artifact root:

```python
from ecs_agent import install_session_recording, release_session_writer

release_session_writer(old_world, agent)  # only if the previous writer is still live
install_session_recording(
    restored_world, agent, service,
    workspace_id=str(root.resolve()), agent_name="main", restore=True,
)
```

The saved head becomes an explicit branch point; later records in the external store survive. `CheckpointSystem.undo` coordinates this rebinding for an in-place restore. Legacy snapshots without session identities import only their remaining messages and mark history as incomplete. A new task reset creates a new session; it does not roll back the database. Record recovery does not imply recovery of external side effects or lost background task handles.

## Subagents and examples

Synchronous delegation retains its text result and is discoverable through `sessions_list`. Background jobs retain their `ses_...` control IDs and expose `transcript_session_id`, allocated while queued. Cancellation before admission still records a terminal status. `subagent_resume` starts a new record session with `source_session_id`; it does not continue the old child model context.

The [plan/task example](../../examples/e2e/plan_and_task/README.md) enables records at `<base_dir>/sessions.sqlite3`, tracks workflow record roots, and explicitly shares their existing descendants when saving/resuming that workflow. Its root front ends settle background jobs and close the service.

Run the deterministic coder/reviewer example:

```bash
uv run python -m examples.e2e.cross_session_records --output /tmp/ecs-session-demo
```

It executes local checks, compacts the coder context, then discovers, searches, and reads complete tool evidence through the reviewer's tools. The report includes source session/record IDs. [The live integration test](../../tests/live/test_session_access_live.py) exercises the equivalent flow with a Responses API model configured through `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`, and `LLM_API_FORMAT=responses`.

The same example also supports real model-driven coder and reviewer agents. With `LLM_API_KEY` set in the environment, run:

```bash
uv run python -m examples.e2e.cross_session_records \
  --live --trace \
  --model deepseek-flash \
  --base-url https://api.deepseek.com \
  --api-format responses \
  --output /tmp/ecs-session-live
```

`--model`, `--base-url`, and `--api-format` can also come from `LLM_MODEL`, `LLM_BASE_URL`, and `LLM_API_FORMAT`. The example does not automatically load `.env`. `--live` requires a key and never silently switches to offline mode. `--api-format` accepts `responses` or `chat`, plus their canonical `openai_*` values.

Both modes save `result.json`, `interaction.json`, and `interaction.txt` in the output directory alongside the session store and artifacts. The JSON interaction contains visible prompts, assistant responses, tool calls, complete projected tool responses, and the host history replacement event; it excludes provider-private replay and reasoning fields. Tool responses include their tool name, call ID, and character count. The text display labels `CALL` and `RESULT` separately, abbreviates long tool responses, and preserves the full body in JSON. With `--trace`, the final report appears once; the complete result remains in `result.json`.

The default validation output is concise: two named smoke checks (arithmetic and a UTF-8 round trip), their results, process exit code, and a random run marker. It contains no artificial padding and normally needs one full-output page. To explicitly exercise multi-page reads and a long single line, add `--stress-output`. That mode labels its generated payload `PAGING_STRESS_FIXTURE` so neither agent mistakes padding for additional checks or tool executions. It is available in both offline and live modes.

Acceptance checks require one validation execution, discovery followed by search scoped to the exact coder session, actual record reads, contiguous full-output pages without duplicate page reads, and a final report containing the correct result phrase, random end marker, session ID, and record ID. The pages must reconstruct the original evidence exactly. `result.json` reports actual tool-call counts per agent, read offsets, duplicate page reads, self-session search hits, and validation output size. A model claim without the corresponding reads, an unscoped search, or duplicate full-output pages fails this example's acceptance and returns a nonzero CLI exit code. Context replacement is host-driven to simulate compaction, rather than an LLM-generated summary.
