"""Langfuse mapping for PhaseChangedEvent."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ecs_agent.observability.subscriber import ObservabilitySubscriber
from ecs_agent.types import EntityId, PhaseChangedEvent


class _FakeSink:
    def __init__(self) -> None:
        self.records: list[Any] = []

    async def emit(self, record: Any) -> None:
        self.records.append(record)


def _event() -> PhaseChangedEvent:
    return PhaseChangedEvent(
        entity_id=EntityId(3),
        graph_id="job",
        from_phase="RUNNING",
        to_phase="BLOCKED",
        reason="on_resume",
        forced=True,
        tick=42,
    )


def test_subscriptions_include_phase_changed() -> None:
    sink = _FakeSink()
    subscriber = ObservabilitySubscriber(sink=sink)
    subscribed_types = {event_type for event_type, _ in subscriber.subscriptions()}
    assert PhaseChangedEvent in subscribed_types


async def test_handle_phase_changed_emits_mapped_record(monkeypatch) -> None:
    sink = _FakeSink()
    subscriber = ObservabilitySubscriber(sink=sink)
    state = SimpleNamespace(has_user_turn=True, pending_turn_records=[])
    captured: dict[str, Any] = {}

    def fake_event_record(state_arg: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"sentinel": True}

    monkeypatch.setattr(subscriber, "_state_for_current_run", lambda: state)
    monkeypatch.setattr(subscriber, "_event_record", fake_event_record)

    await subscriber.handle_phase_changed(_event())

    assert sink.records == [{"sentinel": True}]
    assert captured["name"] == "phase.transition"
    assert captured["entity_id"] == 3
    assert captured["tick"] == 42
    assert captured["input"] == {"graph_id": "job", "from_phase": "RUNNING"}
    assert captured["output"] == {"to_phase": "BLOCKED", "reason": "on_resume", "forced": True}


async def test_handle_phase_changed_noop_without_run_state(monkeypatch) -> None:
    sink = _FakeSink()
    subscriber = ObservabilitySubscriber(sink=sink)
    monkeypatch.setattr(subscriber, "_state_for_current_run", lambda: None)
    await subscriber.handle_phase_changed(_event())
    assert sink.records == []
