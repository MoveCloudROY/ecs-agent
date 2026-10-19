"""Strict session URI parsing and bounded deterministic line views."""

from collections.abc import Generator, Iterator
from dataclasses import dataclass
import json
import re
from urllib.parse import parse_qs, urlsplit

from ecs_agent.components import SessionPolicyComponent
from ecs_agent.sessions.service import SessionService, project_value
from ecs_agent.sessions.types import SessionRecord

_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}")
_MAX_PAGE_CHARACTERS = 12000


@dataclass(slots=True, frozen=True)
class SessionResource:
    """Parsed immutable version-one resource locator."""

    session_id: str
    record_id: str
    kind: str


def parse_session_uri(uri: str) -> SessionResource:
    """Reject unknown versions, extra parameters, path tricks, and unpinned transcripts."""
    parsed = urlsplit(uri)
    if parsed.scheme != "session" or not _IDENTIFIER.fullmatch(parsed.netloc) or parsed.fragment:
        raise ValueError("Invalid session resource URI")
    query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=2)
    if query.get("v") != ["1"]:
        raise ValueError("Unsupported session resource version")
    parts = parsed.path.split("/")
    if parts == ["", "transcript"]:
        head = query.get("head", [])
        if set(query) != {"v", "head"} or len(head) != 1 or not _IDENTIFIER.fullmatch(head[0]):
            raise ValueError("Transcript URI requires one fixed head")
        return SessionResource(parsed.netloc, head[0], "transcript")
    if (
        set(query) == {"v"} and len(parts) in {3, 4} and parts[:2] == ["", "records"]
        and _IDENTIFIER.fullmatch(parts[2]) and (len(parts) == 3 or parts[3] == "tool-result")
    ):
        return SessionResource(parsed.netloc, parts[2], "record" if len(parts) == 3 else "tool-result")
    raise ValueError("Invalid session resource path")


def _text_lines(text: str, *, prefix: str) -> Iterator[str]:
    for start in range(0, len(text), 512):
        yield prefix + json.dumps(text[start:start + 512], ensure_ascii=False)


def _record_lines(service: SessionService, record: SessionRecord, *, full_tool_body: bool) -> Iterator[str]:
    yield f"record: {record.record_id} session: {record.session_id} seq: {record.seq} kind: {record.kind}"
    metadata = project_value(record.payload)
    metadata.pop("content", None)
    metadata.pop("artifact_sha256", None)
    metadata.pop("artifact_owner_session_id", None)
    yield from _text_lines(json.dumps(metadata, ensure_ascii=False, sort_keys=True), prefix="metadata: ")
    is_tool = record.payload.get("role") == "tool"
    if is_tool and not full_tool_body:
        yield f"tool_result_uri: {service.tool_result_uri(record)}"
        # The complete body is a separate resource; do not expand it in transcripts.
        yield from _text_lines(str(project_value(record.payload).get("content", ""))[:256], prefix="preview: ")
    else:
        for chunk in service.content_chunks(record):
            yield "text: " + json.dumps(chunk, ensure_ascii=False)


def read_session_resource(
    service: SessionService, caller_id: str, policy: SessionPolicyComponent, uri: str,
    *, offset: int = 1, limit: int = 0, line_numbers: bool = False,
) -> str:
    """Read display lines; the metadata envelope does not contribute to offsets.

    Text lines contain JSON strings, preserving original newlines and long-line
    segment boundaries. Concatenating their decoded values reconstructs the
    projected body exactly. Unprojected original text remains in the store.
    """
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 1:
        raise ValueError("offset must be a positive integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    resource = parse_session_uri(uri)
    service.authorize(caller_id, policy, resource.session_id)
    record = service.store.get_record(resource.record_id)
    if record.session_id != resource.session_id:
        raise ValueError("Record belongs to a different session")
    if resource.kind == "tool-result" and record.payload.get("role") != "tool":
        raise ValueError("Record is not a tool result")
    records = service.store.path(resource.session_id, resource.record_id) if resource.kind == "transcript" else iter([record])

    def lines() -> Generator[str, None, None]:
        for item in records:
            yield from _record_lines(service, item, full_tool_body=resource.kind == "tool-result")

    maximum = min(limit or 80, 500)
    selected: list[str] = []
    consumed = 0
    has_more = False
    budget_truncated = False
    # Reserve more than the maximal fixed envelope, including a 512-character URI.
    body_budget = _MAX_PAGE_CHARACTERS - 1500
    iterator = lines()
    try:
        for number, line in enumerate(iterator, start=1):
            if number < offset:
                continue
            rendered = f"{number}: {line}" if line_numbers else line
            if len(selected) >= maximum or consumed + len(rendered) + 1 > body_budget:
                has_more = True
                budget_truncated = consumed + len(rendered) + 1 > body_budget
                break
            selected.append(rendered)
            consumed += len(rendered) + 1
    finally:
        iterator.close()
    end = offset + len(selected) - 1
    envelope = {
        "read_uri": uri, "snapshot_seq": record.seq, "start_line": offset,
        "end_line": end, "next_offset": end + 1 if has_more else None,
        "has_more": has_more, "truncated": budget_truncated or (has_more and (limit == 0 or limit > maximum)),
    }
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True) + "\n---\n" + "\n".join(selected)
