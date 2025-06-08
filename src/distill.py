import argparse
import asyncio
import time
from pydantic_graph import Graph
from agents._agent_registry import AgentRegistry
from agents.doc_parser_agent import create_doc_parser_agent
from models import MyUsage
from nodes import SectionDistillerNode, DocParserNode

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3", help="Ollama model to use")
args = parser.parse_args()

AgentRegistry._agents.clear()  # Optional: Reset any previously registered agents
AgentRegistry._agents["doc_parser_agent"] = create_doc_parser_agent(model_name=args.model)

sample = r"C:\Users\talha\myprojects\graph-builder-agent\src\data\docs.pdf"
graph = Graph(nodes=[DocParserNode, SectionDistillerNode])
state = MyUsage()

async def main():
    s = time.perf_counter()
    result = await graph.run(DocParserNode(sample), state=state)
    print("Final Output:\n", result)
    print("Graph State:\n", result.state)
    print("Time Taken:", time.perf_counter() - s)

asyncio.run(main())
