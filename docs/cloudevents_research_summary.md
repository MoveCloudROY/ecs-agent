# CloudEvents Research Summary

## Core Specification (v1.0.2)

### REQUIRED Attributes

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | String | Identifies the event. Producers MUST ensure `source` + `id` is unique for each distinct event | "A234-1234-1234", UUID |
| `source` | URI-reference | Identifies the context in which an event happened (event producer) | "https://github.com/cloudevents", "/sensors/tn-1234567" |
| `type` | String | Event type descriptor, SHOULD be prefixed with reverse-DNS | "com.github.pull_request.opened" |
| `specversion` | String | CloudEvents spec version (MUST be "1.0" for v1.0.x) | "1.0" |

**Key Constraints:**
- All REQUIRED fields MUST be non-empty strings
- `source` + `id` combination MUST be unique per distinct event
- Duplicate events (network retries) MAY reuse the same `id`

### OPTIONAL Attributes

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `datacontenttype` | String (RFC 2046) | Media type of `data` field | "application/json", "text/xml" |
| `dataschema` | URI | Schema that `data` adheres to | "https://example.com/schema/v1" |
| `subject` | String | Subject of event in context of `source` | "mynewfile.jpg" |
| `time` | Timestamp (RFC 3339) | When the occurrence happened | "2018-04-05T17:31:00Z" |
| `data` | Any | Event payload (domain-specific information) | `{"orderId": "123"}` |

**Key Constraints:**
- `datacontenttype` defaults to "application/json" for JSON format events if omitted
- `time` if cannot be determined MAY be set to current time, but all producers for same `source` MUST be consistent
- `data` is OPTIONAL and has no type restrictions

### Extension Mechanism

**Rules:**
- Extensions MUST follow attribute naming convention: lowercase letters ('a'-'z') or digits ('0'-'9')
- Attribute names SHOULD be descriptive and terse, SHOULD NOT exceed 20 characters
- Extensions use the same type system as standard attributes (Boolean, Integer, String, Binary, URI, URI-reference, Timestamp)
- Extensions are serialized like standard attributes
- No custom namespacing required (flat structure)

**Example:**
```json
{
  "id": "123",
  "source": "example.com",
  "type": "example.event",
  "specversion": "1.0",
  "correlationid": "txn-abc-123",
  "causationid": "parent-event-456",
  "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
}
```

## Correlation Extension

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `correlationid` | String | OPTIONAL | Groups all events in same logical flow/business transaction |
| `causationid` | String | OPTIONAL | The `id` of the event that directly caused this event |

### Purpose

- **`correlationid`**: Answers "Which events are part of the same business transaction?"
  - All events in a workflow share the same correlation ID
  - Generated at entry point (API gateway, UI interaction)
  - Persists unchanged throughout entire flow

- **`causationid`**: Answers "Which specific event directly triggered this event?"
  - Points to the `id` of the parent event
  - Forms parent-child causal chains
  - Enables event graph reconstruction

### Request-Response Pattern

```python
# Initial request
{
  "id": "order-123",
  "source": "https://example.com/orders",
  "type": "com.example.order.placed",
  "correlationid": "txn-abc-123",  # NEW correlation
  "data": {"orderId": "123"}
}

# Triggered response 1
{
  "id": "payment-789",
  "source": "https://example.com/payments",
  "type": "com.example.payment.processed",
  "correlationid": "txn-abc-123",  # SAME correlation
  "causationid": "order-123",       # PARENT event id
  "data": {"amount": 150.0}
}

# Triggered response 2 (parallel to response 1)
{
  "id": "inventory-456",
  "source": "https://example.com/inventory",
  "type": "com.example.inventory.checked",
  "correlationid": "txn-abc-123",  # SAME correlation
  "causationid": "order-123",       # SAME parent
  "data": {"available": true}
}

# Nested response (child of inventory check)
{
  "id": "shipping-012",
  "source": "https://example.com/shipping",
  "type": "com.example.shipping.scheduled",
  "correlationid": "txn-abc-123",  # SAME correlation
  "causationid": "inventory-456",   # PARENT is inventory event
  "data": {"carrier": "FastShip"}
}
```

### Transaction Grouping

**Fan-out pattern**: One event triggers multiple downstream events
- All children share same `causationid` (parent's `id`)
- All maintain same `correlationid`

**Best Practices:**
1. Generate correlation IDs at system entry points using UUIDs
2. Always set causation ID to the `id` of the triggering event
3. Use consistently throughout a flow once started
4. Consider retention implications for queries

## Distributed Tracing Extension

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `traceparent` | String | REQUIRED* | W3C Trace Context: version-traceid-spanid-flags |
| `tracestate` | String | OPTIONAL | Vendor-specific trace identification info |

*REQUIRED only when this extension is used

### W3C Trace Context Format

**`traceparent` structure** (section 3.2 of W3C spec):
```
version-traceid-spanid-traceflags

Example: 00-0af7651916cd43dd8448eb211c80319c-b9c7c989f97918e1-01
         ││ │                                │                  ││
         ││ └─ trace-id (16 bytes, 32 hex)  │                  ││
         │└─── version (1 byte, 2 hex)      │                  ││
         │                                   │                  ││
         │                                   └─ span-id (8 bytes, 16 hex)
         └───────────────────────────────────── trace-flags (1 byte, 2 hex)
```

**Components:**
- `version`: Currently "00"
- `trace-id`: 16-byte unique identifier for entire trace (32 hex chars)
- `span-id`: 8-byte identifier for this span/operation (16 hex chars)
- `trace-flags`: Bit field for trace options (01 = sampled)

**`tracestate` structure** (section 3.3):
```
vendor1=value1,vendor2=value2
Example: rojo=00-abc...,congo=xyz...
```

### Propagation Rules

**Single-hop transmission** (source → sink directly):
- CloudEvents extension MUST carry same trace info as protocol headers
- HTTP example:
  ```
  HTTP Headers:
    traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
    tracestate: vendor=value
  
  CloudEvent:
    ce-traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
    ce-tracestate: vendor=value
  ```

**Multi-hop transmission** (source → middleware → sink):
- Extension MUST carry trace info of **starting trace** (not each hop)
- Protocol headers carry hop-specific trace info
- Middleware MAY add extension if source didn't include it
- Example:
  ```
  HTTP traceparent:    00-<new-trace-for-this-hop>-<new-span>-01
  ce-traceparent:      00-<original-trace-from-start>-<original-span>-01
  ```

### OpenTelemetry Integration

**When to use this extension:**
- Event-driven systems with multiple async hops
- Need to correlate events with distributed traces
- Events cross trust boundaries (can't rely on protocol headers)

**How OTel uses it:**
1. At event creation: Inject current span context into `traceparent`
2. At event reception: Extract trace context from `traceparent`
3. Create new span as child of extracted context
4. Propagate updated context in protocol headers for next hop

**Reference:** [OpenTelemetry Semantic Conventions for CloudEvents](https://opentelemetry.io/docs/specs/semconv/cloudevents/cloudevents-spans/)

## Python Implementation Patterns

### Envelope Dataclass Design (Real-World Examples)

**Pattern 1: Azure SDK** (most CloudEvents-compliant)
```python
from dataclasses import dataclass
from typing import Any

@dataclass(slots=True)
class CloudEvent:
    # REQUIRED
    id: str
    source: str
    type: str
    specversion: str = "1.0"
    
    # OPTIONAL
    datacontenttype: str | None = None
    dataschema: str | None = None
    subject: str | None = None
    time: str | None = None  # RFC 3339 string
    data: Any | None = None
    
    # Extensions (flat, not nested)
    correlationid: str | None = None
    causationid: str | None = None
    traceparent: str | None = None
    tracestate: str | None = None
```

**Pattern 2: Microsoft AutoGen** (with metadata wrapper)
```python
@dataclass(slots=True)
class EnvelopeMetadata:
    """Telemetry propagation metadata."""
    traceparent: str | None = None
    tracestate: str | None = None
    links: Sequence[Link] | None = None  # OpenTelemetry links

@dataclass(slots=True)
class MessageEnvelope:
    """Message bus envelope with CloudEvents fields."""
    # Core CloudEvents
    id: str
    source: str
    type: str
    specversion: str = "1.0"
    
    # Optional
    time: str | None = None
    data: Any | None = None
    
    # Extension metadata
    metadata: EnvelopeMetadata | None = None
    
    # Correlation
    correlation_id: str | None = None
    causation_id: str | None = None
```

**Pattern 3: Vercel Workers** (TypedDict for JSON compatibility)
```python
from typing import TypedDict

class CloudEventData(TypedDict):
    messageId: str
    queueName: str
    consumerGroup: str

class CloudEvent(TypedDict, total=False):
    # REQUIRED (total=False means validate at runtime)
    type: str
    source: str
    id: str
    
    # OPTIONAL
    datacontenttype: str
    data: CloudEventData
```

### Validation Patterns

**Pattern 1: Constructor Validation**
```python
@dataclass(slots=True)
class CloudEvent:
    id: str
    source: str
    type: str
    specversion: str = "1.0"
    
    def __post_init__(self) -> None:
        """Validate REQUIRED fields."""
        if not self.id:
            raise ValueError("id MUST be non-empty string")
        if not self.source:
            raise ValueError("source MUST be non-empty string")
        if not self.type:
            raise ValueError("type MUST be non-empty string")
        if self.specversion != "1.0":
            raise ValueError("specversion MUST be '1.0'")
```

**Pattern 2: Factory Method**
```python
@classmethod
def create(
    cls,
    source: str,
    type: str,
    data: Any,
    *,
    id: str | None = None,
    correlation_id: str | None = None,
    causation_id: str | None = None,
) -> CloudEvent:
    """Create CloudEvent with defaults."""
    import uuid
    from datetime import datetime, timezone
    
    return cls(
        id=id or str(uuid.uuid4()),
        source=source,
        type=type,
        specversion="1.0",
        time=datetime.now(timezone.utc).isoformat(),
        data=data,
        correlationid=correlation_id,
        causationid=causation_id,
    )
```

### Serialization Patterns

**Pattern 1: Simple JSON (Azure SDK style)**
```python
import json
from dataclasses import asdict

def to_json(event: CloudEvent) -> str:
    """Serialize CloudEvent to JSON."""
    d = asdict(event)
    # Remove None values for cleaner JSON
    return json.dumps({k: v for k, v in d.items() if v is not None})

def from_json(data: str) -> CloudEvent:
    """Deserialize CloudEvent from JSON."""
    return CloudEvent(**json.loads(data))
```

**Pattern 2: With datetime handling**
```python
from datetime import datetime, timezone

def to_dict(event: CloudEvent) -> dict[str, Any]:
    """Convert to dict with proper datetime serialization."""
    d = asdict(event)
    
    # Convert datetime to RFC 3339 string
    if isinstance(d.get("time"), datetime):
        d["time"] = d["time"].isoformat()
    
    # Remove None values
    return {k: v for k, v in d.items() if v is not None}
```

### Trace Context Propagation (Microsoft AutoGen pattern)

```python
from opentelemetry import trace
from opentelemetry.propagate import inject, extract

def get_telemetry_envelope_metadata() -> EnvelopeMetadata:
    """Extract current trace context into envelope metadata."""
    carrier: dict[str, str] = {}
    inject(carrier)  # Injects traceparent, tracestate into carrier
    
    return EnvelopeMetadata(
        traceparent=carrier.get("traceparent"),
        tracestate=carrier.get("tracestate"),
    )

def set_telemetry_context_from_envelope(metadata: EnvelopeMetadata) -> None:
    """Restore trace context from envelope metadata."""
    carrier = {}
    if metadata.traceparent:
        carrier["traceparent"] = metadata.traceparent
    if metadata.tracestate:
        carrier["tracestate"] = metadata.tracestate
    
    context = extract(carrier)
    trace.set_span_in_context(context)
```

## Recommendations for ecs-agent

### Envelope Schema

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import uuid

@dataclass(slots=True)
class MessageBusEnvelope:
    """CloudEvents v1.0 compliant message bus envelope.
    
    Combines core CloudEvents attributes with correlation and distributed
    tracing extensions for comprehensive event tracking in ecs-agent.
    """
    
    # REQUIRED CloudEvents attributes
    id: str  # Unique event ID (UUID recommended)
    source: str  # Event producer (e.g., "ecs-agent://entity/123")
    type: str  # Event type (e.g., "ecs.agent.reasoning.completed")
    specversion: str = "1.0"
    
    # OPTIONAL CloudEvents attributes
    datacontenttype: str = "application/json"
    dataschema: str | None = None
    subject: str | None = None  # e.g., entity ID
    time: str | None = None  # RFC 3339 timestamp
    data: Any | None = None  # Event payload
    
    # Correlation extension (track logical flows)
    correlationid: str | None = None  # Groups related events
    causationid: str | None = None  # Direct parent event ID
    
    # Distributed tracing extension (OpenTelemetry integration)
    traceparent: str | None = None  # W3C Trace Context
    tracestate: str | None = None  # Vendor trace info
    
    @classmethod
    def create(
        cls,
        source: str,
        type: str,
        data: Any,
        *,
        subject: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> MessageBusEnvelope:
        """Create envelope with auto-generated ID and timestamp.
        
        Args:
            source: Event producer identifier
            type: Event type (reverse-DNS recommended)
            data: Event payload
            subject: Event subject in context of source
            correlation_id: Logical flow ID (new flow if None)
            causation_id: Parent event ID (for causal chains)
        
        Returns:
            New envelope with defaults applied.
        """
        return cls(
            id=str(uuid.uuid4()),
            source=source,
            type=type,
            specversion="1.0",
            time=datetime.now(timezone.utc).isoformat(),
            data=data,
            subject=subject,
            correlationid=correlation_id or str(uuid.uuid4()),
            causationid=causation_id,
        )
    
    def __post_init__(self) -> None:
        """Validate REQUIRED CloudEvents fields."""
        if not self.id:
            raise ValueError("id MUST be non-empty string")
        if not self.source:
            raise ValueError("source MUST be non-empty string")
        if not self.type:
            raise ValueError("type MUST be non-empty string")
        if self.specversion != "1.0":
            raise ValueError(f"specversion MUST be '1.0', got {self.specversion!r}")
```

### Validation Rules

**At creation** (factory method):
- Auto-generate `id` (UUID4)
- Auto-generate `correlationid` if not provided
- Auto-generate `time` (current UTC timestamp)
- Set `specversion` to "1.0"
- Set `datacontenttype` to "application/json"

**At publish** (system boundary):
- Validate all REQUIRED fields are non-empty
- Validate `source` + `id` uniqueness (deduplication)
- Inject current trace context into `traceparent` if not set
- Log envelope metadata (structured logging)

**At receive** (consumer):
- Validate `specversion` is supported
- Extract trace context from `traceparent` for span creation
- Restore correlation context from `correlationid`
- Validate `data` against `dataschema` if specified

### Extension Fields Strategy

**Mandatory Extensions:**
- `correlationid`: REQUIRED at envelope creation
  - Auto-generated if not provided
  - Used for transaction grouping and debugging

**Optional Extensions:**
- `causationid`: RECOMMENDED for event-driven flows
  - Set to parent event's `id` when processing triggers new events
  - Enables causal chain reconstruction
  
- `traceparent`: AUTO-INJECTED when OpenTelemetry enabled
  - Populated from current span context
  - Used for distributed trace correlation
  
- `tracestate`: AUTO-INJECTED if present in current context
  - Vendor-specific trace metadata
  - Preserves trace state across hops

**Custom Extensions (future):**
- Follow CloudEvents naming: lowercase, ≤20 chars
- Example: `agentid`, `entityid`, `systemid`
- Document in extension registry

### Example Usage

```python
# Initial event in flow
envelope = MessageBusEnvelope.create(
    source="ecs-agent://world/1",
    type="ecs.agent.reasoning.started",
    data={"entity_id": 123, "query": "solve problem"},
    subject="entity/123",
)

# Downstream event (caused by reasoning completion)
response_envelope = MessageBusEnvelope.create(
    source="ecs-agent://world/1",
    type="ecs.agent.tool.executed",
    data={"tool": "calculator", "result": 42},
    subject="entity/123",
    correlation_id=envelope.correlationid,  # SAME flow
    causation_id=envelope.id,  # PARENT event
)
```

### Size Constraints

Per CloudEvents spec:
- Intermediaries MUST forward events ≤64KB
- Consumers SHOULD accept events ≥64KB
- **Recommendation**: Keep envelopes <16KB for safety
  - Store large payloads externally, link in `data`
  - Use `subject` or extensions for routing metadata
  
