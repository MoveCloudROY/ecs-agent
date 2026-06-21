"""Shared OpenAI message and tool format conversion utilities."""

from __future__ import annotations

from typing import Any

from ecs_agent.types import Message, ToolSchema


def convert_tools_to_openai(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]
