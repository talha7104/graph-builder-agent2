from agents._base import ollama_model
from pydantic_ai.agent import Agent
from pydantic_ai.schema import AgentName

def create_doc_parser_agent(model_name="llama3") -> Agent:
    return Agent(
        name=AgentName.doc_parser_agent,
        model=ollama_model(model_name),
        instructions="Extract sections and metadata from the provided document content."
    )
