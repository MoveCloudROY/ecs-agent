"""OpenAI-compatible HTTP provider using httpx."""

from typing import Any
import httpx
from ecs_agent.types import Message, CompletionResult, ToolSchema, ToolCall, Usage


class OpenAIProvider:
    """OpenAI-compatible LLM provider using httpx AsyncClient."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._client = httpx.AsyncClient(trust_env=False)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> CompletionResult:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        request_body: dict[str, Any] = {
            "model": self._model,
            "messages": self._convert_messages_to_openai(messages),
        }

        if tools is not None:
            request_body["tools"] = self._convert_tools_to_openai(tools)

        response = await self._client.post(url, json=request_body, headers=headers)
        response.raise_for_status()

        response_data = response.json()
        return self._parse_response(response_data)

    def _convert_messages_to_openai(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []
        for msg in messages:
            openai_msg: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            openai_messages.append(openai_msg)
        return openai_messages

    def _convert_tools_to_openai(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        openai_tools: list[dict[str, Any]] = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            openai_tools.append(openai_tool)
        return openai_tools

    def _parse_response(self, response_data: dict[str, Any]) -> CompletionResult:
        message_data = response_data["choices"][0]["message"]

        role = message_data["role"]
        content = message_data.get("content") or ""

        tool_calls: list[ToolCall] | None = None
        if "tool_calls" in message_data and message_data["tool_calls"]:
            tool_calls = []
            for tc in message_data["tool_calls"]:
                tool_call = ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                )
                tool_calls.append(tool_call)

        message = Message(role=role, content=content, tool_calls=tool_calls)

        usage_data = response_data.get("usage")
        usage: Usage | None = None
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data["prompt_tokens"],
                completion_tokens=usage_data["completion_tokens"],
                total_tokens=usage_data["total_tokens"],
            )

        return CompletionResult(message=message, usage=usage)
