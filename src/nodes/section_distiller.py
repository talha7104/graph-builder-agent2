from dataclasses import dataclass

from pydantic_graph import BaseNode, GraphRunContext, End

from src.models import MyUsage, AgentName, Section, SectionContent, Mention, build_dynamic_relation_model
from src.agents import AgentRegistry, create_dynamic_relation_extractor_agent
from src._utils import update_usage

from typing import Any


@dataclass
class SectionDistillerNode(BaseNode[MyUsage, None, Any]):
    section: SectionContent

    async def run(self, ctx: GraphRunContext[MyUsage, None]) -> End:

        distilled_result = await AgentRegistry.get(
            AgentName.section_distiller_agent
        ).run(user_prompt=f"section_title:{self.section.title}\n section_content: {self.section.content}",
              model_settings={"temperature": 0})

        usage = distilled_result.usage()
        ctx.state = update_usage(ctx.state, usage)

        distilled_section: Section = distilled_result.output

        mentions = distilled_section.mentions
        mention_strings = [item.string for item in mentions]

        relation_model = build_dynamic_relation_model(mention_strings=mention_strings)
        relation_extractor_agent = create_dynamic_relation_extractor_agent(
            relation_model=relation_model,
            mention_strings=mention_strings,
        )

        relation_result = await relation_extractor_agent.run(
            user_prompt=f"section_title:{self.section.title}\n section_content: {self.section.content}",
            model_settings={"temperature": 0}
        )

        usage = relation_result.usage()
        ctx.state = update_usage(ctx.state, usage)

        distilled_section.relations = relation_result.output

        return End(distilled_section)
