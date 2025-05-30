from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from src.config import BASE_URL, MODEL_NAME

ollama_model = OpenAIModel(
    model_name=MODEL_NAME,
    provider=OpenAIProvider(base_url=f"{BASE_URL}/v1")
)


