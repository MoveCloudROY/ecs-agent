from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class MCPConfigComponent:
    server_name: str
    transport_type: Literal["stdio", "sse", "http"]
    config: dict[str, Any]


@dataclass(slots=True)
class MCPClientComponent:
    session: Any
    connected: bool = False
    cached_tools: list[dict[str, Any]] = field(default_factory=list)
