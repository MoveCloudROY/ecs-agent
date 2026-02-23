"""Test response_format parameter support for structured outputs."""

import unittest
from unittest.mock import AsyncMock, Mock, patch
from pydantic import BaseModel
import httpx

from ecs_agent.providers.openai_provider import (
    OpenAIProvider,
    pydantic_to_response_format,
)
from ecs_agent.types import Message


class User(BaseModel):
    """Simple test model for structured output."""

    name: str
    age: int
    email: str


class Address(BaseModel):
    """Test model with nested fields."""

    street: str
    city: str
    postal_code: str


class TestPydanticToResponseFormat(unittest.TestCase):
    """Test pydantic_to_response_format helper function."""

    def test_generates_json_schema_structure(self):
        """Verify helper generates correct json_schema response_format."""
        result = pydantic_to_response_format(User)

        self.assertEqual(result["type"], "json_schema")
        self.assertIn("json_schema", result)
        self.assertEqual(result["json_schema"]["name"], "User")
        self.assertTrue(result["json_schema"]["strict"])

    def test_includes_model_schema(self):
        """Verify schema includes model properties."""
        result = pydantic_to_response_format(User)
        schema = result["json_schema"]["schema"]

        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertIn("name", schema["properties"])
        self.assertIn("age", schema["properties"])
        self.assertIn("email", schema["properties"])

    def test_schema_has_correct_types(self):
        """Verify schema property types are correct."""
        result = pydantic_to_response_format(User)
        schema = result["json_schema"]["schema"]

        self.assertEqual(schema["properties"]["name"]["type"], "string")
        self.assertEqual(schema["properties"]["age"]["type"], "integer")
        self.assertEqual(schema["properties"]["email"]["type"], "string")

    def test_raises_on_non_model(self):
        """Verify helper rejects non-Pydantic models."""
        with self.assertRaises(TypeError):
            pydantic_to_response_format(dict)

    def test_raises_on_model_instance(self):
        """Verify helper rejects model instances (not classes)."""
        user_instance = User(name="John", age=30, email="john@example.com")
        with self.assertRaises(TypeError):
            pydantic_to_response_format(user_instance)

    def test_works_with_nested_models(self):
        """Verify helper works with nested Pydantic models."""
        result = pydantic_to_response_format(Address)
        self.assertEqual(result["json_schema"]["name"], "Address")
        self.assertIn("street", result["json_schema"]["schema"]["properties"])


class TestOpenAIProviderResponseFormat(unittest.TestCase):
    """Test OpenAIProvider response_format parameter support."""

    def setUp(self):
        """Set up test provider."""
        self.provider = OpenAIProvider(
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="gpt-4o",
        )

    @patch("httpx.AsyncClient.post")
    async def test_response_format_none_unchanged(self, mock_post):
        """Verify request_body unchanged when response_format is None."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_post.return_value = mock_response

        messages = [Message(role="user", content="test")]
        result = await self.provider.complete(messages)

        # Verify post was called with correct structure (no response_format)
        self.assertTrue(mock_post.called)
        call_kwargs = mock_post.call_args[1]
        request_body = call_kwargs["json"]

        self.assertNotIn("response_format", request_body)
        self.assertEqual(result.message.content, "hello")

    @patch("httpx.AsyncClient.post")
    async def test_response_format_json_object_included(self, mock_post):
        """Verify response_format added to request_body when provided."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "choices": [
                {"message": {"role": "assistant", "content": '{"key": "value"}'}}
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            },
        }
        mock_post.return_value = mock_response

        response_format = {"type": "json_object"}
        messages = [Message(role="user", content="test")]

        result = await self.provider.complete(messages, response_format=response_format)

        # Verify response_format in request body
        call_kwargs = mock_post.call_args[1]
        request_body = call_kwargs["json"]

        self.assertIn("response_format", request_body)
        self.assertEqual(request_body["response_format"], response_format)

    @patch("httpx.AsyncClient.post")
    async def test_response_format_json_schema_included(self, mock_post):
        """Verify json_schema response_format included in request."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "{}"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            },
        }
        mock_post.return_value = mock_response

        response_format = pydantic_to_response_format(User)
        messages = [Message(role="user", content="Extract user info")]

        result = await self.provider.complete(messages, response_format=response_format)

        # Verify response_format in request body
        call_kwargs = mock_post.call_args[1]
        request_body = call_kwargs["json"]

        self.assertIn("response_format", request_body)
        self.assertEqual(request_body["response_format"]["type"], "json_schema")
        self.assertEqual(request_body["response_format"]["json_schema"]["name"], "User")
        self.assertTrue(request_body["response_format"]["json_schema"]["strict"])

    @patch("httpx.AsyncClient.post")
    async def test_request_body_structure(self, mock_post):
        """Verify request_body has correct structure with response_format."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "test"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_post.return_value = mock_response

        messages = [Message(role="user", content="test")]
        response_format = {"type": "json_object"}

        await self.provider.complete(messages, response_format=response_format)

        call_kwargs = mock_post.call_args[1]
        request_body = call_kwargs["json"]

        # Verify all required fields
        self.assertIn("model", request_body)
        self.assertEqual(request_body["model"], "gpt-4o")
        self.assertIn("messages", request_body)
        self.assertIn("response_format", request_body)

        # Verify response_format is dict-like
        self.assertIsInstance(request_body["response_format"], dict)

    @patch("httpx.AsyncClient.post")
    async def test_with_tools_and_response_format(self, mock_post):
        """Verify both tools and response_format can coexist."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "{}"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            },
        }
        mock_post.return_value = mock_response

        from ecs_agent.types import ToolSchema

        messages = [Message(role="user", content="test")]
        tools = [
            ToolSchema(
                name="test_tool",
                description="Test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]
        response_format = {"type": "json_object"}

        await self.provider.complete(
            messages, tools=tools, response_format=response_format
        )

        call_kwargs = mock_post.call_args[1]
        request_body = call_kwargs["json"]

        # Verify both present
        self.assertIn("tools", request_body)
        self.assertIn("response_format", request_body)
        self.assertEqual(len(request_body["tools"]), 1)


# Async test runner helper
def async_test(coro):
    """Decorator to run async tests."""

    def wrapper(*args, **kwargs):
        import asyncio

        return asyncio.run(coro(*args, **kwargs))

    return wrapper


# Apply decorator to async test methods
TestOpenAIProviderResponseFormat.test_response_format_none_unchanged = async_test(
    TestOpenAIProviderResponseFormat.test_response_format_none_unchanged
)
TestOpenAIProviderResponseFormat.test_response_format_json_object_included = async_test(
    TestOpenAIProviderResponseFormat.test_response_format_json_object_included
)
TestOpenAIProviderResponseFormat.test_response_format_json_schema_included = async_test(
    TestOpenAIProviderResponseFormat.test_response_format_json_schema_included
)
TestOpenAIProviderResponseFormat.test_request_body_structure = async_test(
    TestOpenAIProviderResponseFormat.test_request_body_structure
)
TestOpenAIProviderResponseFormat.test_with_tools_and_response_format = async_test(
    TestOpenAIProviderResponseFormat.test_with_tools_and_response_format
)


import pytest
from ecs_agent.providers import FakeProvider
from ecs_agent.types import CompletionResult


class TestFakeProviderResponseFormat(unittest.TestCase):
    """Test FakeProvider response_format parameter support."""

    def test_fake_provider_stores_response_format(self):
        """FakeProvider should store response_format for test assertions."""
        resp = CompletionResult(message=Message(role="assistant", content="Response"))
        provider = FakeProvider(responses=[resp])

        response_format = {"type": "json_object"}

        async def run_test():
            await provider.complete(
                [Message(role="user", content="Test")], response_format=response_format
            )

        import asyncio
        asyncio.run(run_test())

        self.assertEqual(provider.last_response_format, {"type": "json_object"})

    def test_fake_provider_protocol_conformance_with_response_format(self):
        """FakeProvider should conform to Protocol with response_format parameter."""
        resp = CompletionResult(message=Message(role="assistant", content="Test"))
        provider = FakeProvider(responses=[resp])

        response_format = {"type": "json_object", "schema": {"type": "object"}}

        async def run_test():
            result = await provider.complete(
                [Message(role="user", content="Query")],
                tools=None,
                stream=False,
                response_format=response_format,
            )
            return result

        import asyncio
        result = asyncio.run(run_test())

        # Verify response is returned correctly
        self.assertEqual(result.message.content, "Test")
        # Verify response_format was stored
        self.assertEqual(provider.last_response_format, response_format)


if __name__ == "__main__":
    unittest.main()
