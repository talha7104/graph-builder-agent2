import asyncio
import json

from pydantic_graph import Graph

from src.nodes import SectionDistillerNode, DocParserNode
from src.models import MyUsage
from src.nodes.single_distiller import SingleDistillerNode

# sample = "...Your sample"
data = json.load(open("/home/ju/PycharmProjects/docgraph-construction/notebooks/history_textbook.json"))
sample = "Tình hình chính trị, kinh tế, văn hóa dưới triều Nguyễn (Nửa đầu thế kỉ XIX)"

for i,(key, value) in enumerate(data['Lớp 10']['Việt Nam ở nửa đầu thế kỉ XIX']["Tình hình chính trị, kinh tế, văn hóa dưới triều Nguyễn (Nửa đầu thế kỉ XIX)"].items()):
    sample += f"\n{str(i)} {key}\n{value}"

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