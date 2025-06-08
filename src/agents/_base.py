import httpx

class OllamaModel:
    def __init__(self, model_name: str = "llama3", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    async def request(self, prompt: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            data = response.json()
            return data.get("response", "No response received from Ollama.")

def ollama_model(model_name="llama3"):
    return OllamaModel(model_name=model_name)
