import asyncio
import json

from pydantic_graph import Graph

from src.nodes import SectionDistillerNode, DocParserNode
from src.models import MyUsage
from src.nodes.single_distiller import SingleDistillerNode

sample = "...Your sample"


graph = Graph(nodes=[DocParserNode, SectionDistillerNode])
state=MyUsage()

async def main():
    import time
    s = time.perf_counter()
    result = await graph.run(DocParserNode(sample), state=state)
    # result = await graph.run(DocParserNode(sample), state=state)
    print(result)
    print("=========")
    print(result.state)
    print("Runtime:", time.perf_counter() - s)
asyncio.run(main())
