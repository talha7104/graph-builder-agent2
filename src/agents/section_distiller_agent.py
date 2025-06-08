from ._agent_registry import AgentRegistry
from models import AgentName, Section
from agents.prompts import SECTION_DISTILLER_AGENT
from pydantic_ai import Agent

from ._base import ollama_model

@AgentRegistry.register(AgentName.section_distiller_agent)
def create_section_distiller_agent() -> Agent:
    agent = Agent(
        name=AgentName.section_distiller_agent.value,
        model=ollama_model,
        system_prompt=SECTION_DISTILLER_AGENT,
        result_type=Section,
        retries=3
    )

    return agent