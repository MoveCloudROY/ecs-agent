"""Explicit stable references for host-provided dynamic prompt values."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ecs_agent.prompts.contracts import PlaceholderSpec, PromptTemplateSource, SystemPromptConfigSpec

_BINDINGS: dict[str, Callable[[], str]] = {}


def register_prompt_binding(name: str, callback: Callable[[], str]) -> None:
    """Register a host callback before loading checkpoints that reference it."""
    if not name or not callable(callback):
        raise ValueError("Prompt bindings require a name and a callable")
    if name in _BINDINGS and _BINDINGS[name] is not callback:
        raise ValueError(f"Prompt binding '{name}' is already registered")
    _BINDINGS[name] = callback


@dataclass(frozen=True, slots=True)
class PromptBinding:
    """A named, checkpoint-safe callback reference; its code remains host-owned."""

    name: str

    def __call__(self) -> str:
        callback = _BINDINGS.get(self.name)
        if callback is None:
            raise ValueError(f"Register prompt binding '{self.name}' before using this config")
        return callback()


def prompt_config_to_dict(config: SystemPromptConfigSpec) -> dict[str, Any]:
    """Encode static values and registered callbacks without executable objects."""
    placeholders = []
    for placeholder in config.placeholders:
        value = placeholder.value
        if callable(value):
            name = value.name if isinstance(value, PromptBinding) else next((key for key, callback in _BINDINGS.items() if callback is value), None)
            if name is None or name not in _BINDINGS:
                raise ValueError(f"Placeholder '{placeholder.name}' requires a registered PromptBinding for persistence")
            encoded: str | dict[str, str] = {"binding": name}
        else:
            encoded = value
        placeholders.append({"name": placeholder.name, "value": encoded})
    return {
        "template_source": {"inline": config.template_source.inline, "file_path": config.template_source.file_path},
        "placeholders": placeholders, "revision": config.revision, "source_path": config.source_path,
    }


def prompt_config_from_dict(data: dict[str, Any]) -> SystemPromptConfigSpec:
    """Restore config and require explicit bindings before accepting callbacks."""
    placeholders = []
    for entry in data.get("placeholders", []):
        value = entry["value"]
        if isinstance(value, dict):
            name = value.get("binding")
            if name not in _BINDINGS or set(value) != {"binding"}:
                raise ValueError(f"Missing or invalid prompt binding: {name}")
            value = PromptBinding(name)
        elif not isinstance(value, str):
            raise ValueError(f"Invalid persisted value for placeholder '{entry['name']}'")
        placeholders.append(PlaceholderSpec(entry["name"], value))
    return SystemPromptConfigSpec(
        PromptTemplateSource(**data["template_source"]), placeholders,
        revision=int(data.get("revision", 0)), source_path=data.get("source_path"),
    )
