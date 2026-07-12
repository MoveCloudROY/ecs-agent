# Structured Output

Structured output enables the LLM to return data in a consistent JSON format that adheres to a specific schema. This is achieved by leveraging JSON mode and Pydantic model validation.

## API Reference

To use structured output, first define a Pydantic model and then convert it into a compatible `response_format` using the `pydantic_to_response_format()` helper.

```python
from ecs_agent.providers.openai_provider import pydantic_to_response_format

response_format = pydantic_to_response_format(MyPydanticModel)
```

### Conversion Logic
The helper `pydantic_to_response_format(model: type)` generates a dictionary in the following format:
```json
{
    "type": "json_schema",
    "json_schema": {
        "name": "ModelName",
        "schema": { ...model_json_schema... },
        "strict": true
    }
}
```

## Usage Example

The following example shows how to use a Pydantic model to extract structured data from the LLM.

```python
import asyncio
from pydantic import BaseModel, Field
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.openai_provider import pydantic_to_response_format
from ecs_agent.types import Message

# 1. Define the Pydantic model
class CityInfo(BaseModel):
    name: str = Field(..., description="The name of the city")
    country: str = Field(..., description="The country where the city is located")
    population: float = Field(..., description="Population in millions")
    landmarks: list[str] = Field(..., description="A list of famous landmarks")

async def main():
    model = Model(
        "qwen3.5-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="...",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )
    
    # 2. Convert to response_format
    response_format = pydantic_to_response_format(CityInfo)
    
    messages = [Message(role="user", content="Tell me about Tokyo.")]
    
    # 3. Call the provider with response_format
    result = await model.complete(messages, response_format=response_format)
    
    # 4. Parse the JSON content back into the model
    json_str = result.message.content
    city_info = CityInfo.model_validate_json(json_str)
    
    print(f"City: {city_info.name} ({city_info.country})")
    print(f"Population: {city_info.population} million")
    print(f"Landmarks: {', '.join(city_info.landmarks)}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Requirements

- **Pydantic**: Requires `pydantic >= 2.0.0` for model validation and schema generation.

## Responses API

The same `response_format` value also works with `ApiFormat.OPENAI_RESPONSES`.
The `/v1/responses` endpoint does not accept the top-level `response_format`
parameter, so the adapter translates it into the Responses-native
`text.format` block (and flattens the `json_schema` nesting) automatically —
see [Responses API › Structured Output](./responses-api.md#structured-output).
You pass the exact same helper output regardless of `api_format`.

## Caveats

- **Streaming**: Structured output is **NOT** compatible with streaming (`stream=True`). If you need JSON-formatted responses, you must wait for the full completion to finish.
- **Model Support**: Not all LLMs or providers support JSON mode with a schema. Check your provider's documentation for specific model capabilities.
- **Completion Time**: Generating structured output can sometimes take longer than standard text generation as the model must adhere strictly to the provided JSON schema.
