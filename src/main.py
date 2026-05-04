import os
import sys
import asyncio
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# OpenAI Agents SDK imports

from agents import Runner, OpenAIResponsesModel
from openai import AsyncOpenAI

from src.agent.tools import build_cli_tools, get_arxiv_server

# Part 3: The Brain
from src.agent.orchestrator import ResearchOrchestrator

try:
    import dotenv
except ImportError:
    print("Please install python-dotenv: pip install python-dotenv")
    sys.exit(1)


async def main():
    # Load Environment Variables explicitly
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ CRITICAL: OPENROUTER_API_KEY not found in environment!")
        print("Please set it in a `.env` file at the root directory.")
        return
    print(os.environ.get("OPENAI_BASE_URL"))
    print(os.environ.get("OPENAI_API_KEY"))
    # Configure the global client for the OpenAI Agents SDK to hit OpenRouter
    custom_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    model = OpenAIResponsesModel(model="deepseek-v3.2", openai_client=custom_client)

    prompt = (
        "Search ArXiv for a brand new paper on Graph Neural Networks (GNNs). "
        "Download it so it gets saved to raw_pdfs, then ingest the downloaded paper "
        "into my local memory. Finally, retrieve chunks from local memory to summarize it."
    )
    print(f"\n👤 USER: {prompt}\n")
    print(f"==================================================")

    # Initialize MCP Server
    print("🔌 Starting Live Internet MCP connection...")
    arxiv_server = get_arxiv_server()
    await arxiv_server.connect()

    tools = await arxiv_server.list_tools()
    print("✅ ArXiv MCP linked! Available tools:", [t.name for t in tools])

    # Initialize our Orchestrator (The Brain)
    print("🧠 Initializing the Agent using OpenAI Agents SDK...")
    orchestrator = ResearchOrchestrator(model="deepseek-v3.2")
    agent = orchestrator.get_agent()

    # Inject tools and MCP Server directly into the agent
    print("🔌 Attaching Local Memory Tool and Live ArXiv MCP Server...")
    agent.tools = build_cli_tools(arxiv_server=arxiv_server)["tools"]
    agent.mcp_servers = [arxiv_server]

    # Initialize runner
    runner = Runner()

    # Query the Agent using the run orchestrator
    print("🧠 Agent running to synthesize tool results...")
    try:
        result = await runner.run(agent, input=prompt)

        print("\n==================================================")
        print("📝 FINAL SUMMARY:\n")
        print(result.final_output)
    finally:
        print("\n🧹 Disconnecting tools and finishing run.")
        await arxiv_server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
