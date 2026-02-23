"""Structured Output example using JSON mode and Pydantic schema generation.

This example demonstrates how to use response_format with the ECS-based LLM Agent framework
to get structured JSON responses that validate against a Pydantic model schema.

Key concepts:
- Define a Pydantic BaseModel to specify the structure of the LLM response
- Use pydantic_to_response_format() helper to convert the model to OpenAI-compatible format
- Pass response_format to provider.complete() to enable JSON mode with schema validation
- Parse the JSON response string and validate it against the Pydantic model

Usage:
   1. Copy .env.example to .env and fill in your API credentials
   2. Run: uv run python examples/structured_output_agent.py

Environment variables:
   LLM_API_KEY   — API key for the LLM provider (required)
   LLM_BASE_URL  — Base URL for the API (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
   LLM_MODEL     — Model name (default: qwen-plus)

Note: Structured output (JSON mode) requires the full response to be available at once,
so streaming is NOT supported with response_format. This example uses FakeProvider as a
fallback when no LLM_API_KEY is available.
"""

from __future__ import annotations

import asyncio
import os
import sys


from pydantic import BaseModel, Field

from ecs_agent.logging import configure_logging, get_logger
from ecs_agent.providers import OpenAIProvider, FakeProvider
from ecs_agent.providers.openai_provider import pydantic_to_response_format
from ecs_agent.types import CompletionResult, Message, Usage

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models — Define the structure of the LLM response
# ---------------------------------------------------------------------------


class CityInfo(BaseModel):
    """Information about a city returned in structured JSON format."""

    name: str = Field(..., description="The name of the city")
    country: str = Field(..., description="The country where the city is located")
    population: int = Field(..., description="Approximate population in millions")
    climate: str = Field(
        ..., description="Primary climate type (e.g., 'Temperate', 'Tropical')"
    )
    landmarks: list[str] = Field(
        ..., description="List of famous landmarks or attractions in the city"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run a structured output agent that extracts city information in JSON format."""
    # Configure logging
    configure_logging(json_output=False)

    # --- Load config from environment ---
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("LLM_MODEL", "qwen-plus")

    # --- Create LLM provider ---
    if api_key:
        print(f"Using model: {model}")
        print(f"Base URL: {base_url}")
        provider = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
    else:
        print("No LLM_API_KEY provided. Using FakeProvider for demonstration.")
        # Create a fake provider with a pre-configured response matching the CityInfo schema
        fake_response = CompletionResult(
            message=Message(
                role="assistant",
                content='{"name": "Tokyo", "country": "Japan", "population": 14, "climate": "Temperate", "landmarks": ["Senso-ji Temple", "Tokyo Tower", "Shibuya Crossing", "Meiji Shrine"]}',
            ),
            usage=Usage(prompt_tokens=50, completion_tokens=100, total_tokens=150),
        )
        provider = FakeProvider(responses=[fake_response])

    print()

    # --- Define the messages for the LLM ---
    messages = [
        Message(
            role="user",
            content="Extract information about Tokyo in JSON format. Include the city name, country, population (in millions), climate, and a list of landmarks.",
        )
    ]

    # --- Generate response_format from Pydantic model ---
    logger.debug("generating_response_format", model_name=CityInfo.__name__)
    response_format = pydantic_to_response_format(CityInfo)

    # --- Call the LLM with structured output enabled ---
    logger.debug(
        "calling_llm_with_structured_output",
        model=model,
        response_format_type=response_format.get("type"),
    )
    result = await provider.complete(messages, response_format=response_format)

    # --- Parse and validate the response ---
    print()
    print("=" * 60)
    print("STRUCTURED OUTPUT RESULT")
    print("=" * 60)

    # The response content should be a JSON string matching the CityInfo schema
    json_str = result.message.content
    logger.debug("parsing_response", response_length=len(json_str))

    try:
        # Validate and parse the JSON response against the Pydantic model
        city_info = CityInfo.model_validate_json(json_str)
        logger.info("structured_output_parsed", city_name=city_info.name)

        # Display the structured result
        print(f"\nCity Name:    {city_info.name}")
        print(f"Country:      {city_info.country}")
        print(f"Population:   {city_info.population} million")
        print(f"Climate:      {city_info.climate}")
        print("Landmarks:")
        for landmark in city_info.landmarks:
            print(f"  - {landmark}")

        # Display usage statistics
        if result.usage:
            print()
            print("=" * 60)
            print("USAGE STATISTICS")
            print("=" * 60)
            print(f"Prompt tokens:     {result.usage.prompt_tokens}")
            print(f"Completion tokens: {result.usage.completion_tokens}")
            print(f"Total tokens:      {result.usage.total_tokens}")

    except Exception as e:
        logger.error(
            "failed_to_parse_structured_output", error=str(e), response=json_str
        )
        print(f"\nError parsing response: {e}")
        print(f"Raw response: {json_str}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
