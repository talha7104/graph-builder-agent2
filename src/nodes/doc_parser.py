from dataclasses import dataclass

from pydantic_graph import BaseNode, GraphRunContext, End

from models import MyUsage, AgentName, Unit, DistilledUnit
from agents._agent_registry import AgentRegistry
from nodes.section_distiller import SectionDistillerNode
from _utils import update_usage

from typing import Any

from _utils import task_group_gather

@dataclass
class DocParserNode(BaseNode[MyUsage, None, Any]):
    raw_document: str

    async def run(
            self,
            ctx: GraphRunContext[MyUsage, None],
    ) -> End:

        result = await AgentRegistry.get(
            AgentName.doc_parser_agent
        ).run(self.raw_document, model_settings={"temperature": 0})

        usage = result.usage()
        ctx.state = update_usage(ctx.state, usage)

        unit: Unit = result.output
        sections = unit.sections

        results = await task_group_gather(
            [
                lambda section=section: SectionDistillerNode(section=section).run(ctx)
                for section in sections
            ]
        )

        distilled_unit = DistilledUnit(
            title=unit.title,
            summary=unit.summary,
            sections=[item.data for item in results],
        )

        return End(data=distilled_unit)