import httpx

from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models import ModelResponse
from pydantic_ai.messages import TextPart, ModelMessage, ModelRequest
from pydantic_ai.usage import Usage


class OllamaFunctionModel(FunctionModel):
    """`FunctionModel` wrapper for calling a local Ollama instance."""

    _system: str = "ollama"


def ollama_model(model_name: str = "llama3", base_url: str = "http://localhost:11434") -> OllamaFunctionModel:
    async def _call(messages: list[ModelMessage], _info) -> ModelResponse:
        prompt_parts: list[str] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    content = getattr(part, "content", None)
                    if isinstance(content, str):
                        prompt_parts.append(content)
        prompt = "\n".join(prompt_parts)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False},
            )
            data = response.json()
            reply = data.get("response", "No response received from Ollama.")
        return ModelResponse(parts=[TextPart(reply)], usage=Usage(), model_name=f"ollama:{model_name}")

    return OllamaFunctionModel(function=_call, model_name=f"ollama:{model_name}")
