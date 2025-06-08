from ._agent_registry import AgentRegistry
from models import AgentName, DistilledUnit
from agents.prompts import SINGLE_DISTILLER_AGENT
from pydantic_ai import Agent

from ._base import ollama_model

@AgentRegistry.register(AgentName.single_distiller_agent)
def create_single_distiller_agent() -> Agent:
    agent = Agent(
        name=AgentName.single_distiller_agent.value,
        model=ollama_model,
        system_prompt=SINGLE_DISTILLER_AGENT,
        result_type=DistilledUnit,
        retries=5
    )

    return agent