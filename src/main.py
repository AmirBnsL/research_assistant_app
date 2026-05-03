import os
import sys
import asyncio
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# OpenAI Agents SDK imports
import agents
from agents import Runner, OpenAIResponsesModel
from openai import AsyncOpenAI

# Part 1: The Memory
from src.rag.rerank import search_local_memory

# Part 2: The Hands
from src.agent.tools.mcp_arxiv import get_arxiv_server

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
        
    # Configure the global client for the OpenAI Agents SDK to hit OpenRouter
    custom_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    model = OpenAIResponsesModel(
        model="openai/gpt-4o-mini", 
        openai_client=custom_client
    )

    prompt = (
        "Search ArXiv for a brand new paper on Graph Neural Networks (GNNs). "
        "Download it so it gets saved to raw_pdfs, then ingest the downloaded paper "
        "into my local memory. Finally, retrieve chunks from local memory to summarize it."
    )
    print(f"\n👤 USER: {prompt}\n")
    print(f"==================================================")
    
    @agents.function_tool
    def tool_search_local_memory(query: str) -> str:
        """Searches the local vector database of uploaded and processed academic papers for the query."""
        print(f"\n   -> 🛠️ Executing: search_local_memory(query={query})")
        res = search_local_memory(query)
        print(f"      🗄️ Local DB successfully returned chunk memories.")
        return res

    @agents.function_tool
    async def tool_download_and_ingest_paper(paper_id: str) -> str:
        """
        Downloads a paper from ArXiv by ID (e.g., "2301.07041") and automatically ingests it to ChromaDB.
        MUST BE USED instead of the raw `download_paper` tool to prevent AI context window overflow.
        """
        print(f"\n   -> 🛠️ Executing: tool_download_and_ingest_paper({paper_id})")
        try:
            # Perform the huge download silently without blowing up the agent's LLM context
            await arxiv_server.call_tool("download_paper", {"paper_id": paper_id})
            
            # Now run the ingest function logic directly
            import glob
            from src.rag.embedding import embed_and_store
            
            pdf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_pdfs"))
            md_files = glob.glob(os.path.join(pdf_dir, "*.md"))
            
            if not md_files:
                return "No new markdown papers found. Make sure the download succeeded."
                
            ingested = []
            for file_path in md_files:  
                filename = os.path.basename(file_path)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                print(f"      📥 Ingesting recently downloaded paper: {filename}...")
                embed_and_store(content, filename)
                ingested.append(filename)
                
                # Move to processed so it's not ingested twice
                processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "user_uploads"))
                os.makedirs(processed_dir, exist_ok=True)
                os.rename(file_path, os.path.join(processed_dir, filename))
                
            return f"Successfully downloaded and ingested {len(ingested)} papers: {', '.join(ingested)}. You can now search them via search_local_memory!"
        except Exception as e:
            print(f"      ❌ Exception occurred during download or ingestion: {e}")
            return f"Failed to download and ingest paper {paper_id}. Error: {e}"
        
    @agents.function_tool
    def tool_ingest_downloaded_papers() -> str:
        """
        Scans the `data/raw_pdfs` folder for newly downloaded ArXiv markdown papers from the `download_paper` tool,
        chunks them, and embeds them into the local ChromaDB database.
        ALWAYS call this tool immediately after successfully using `download_paper` or `read_paper`!
        """
        print(f"\n   -> 🛠️ Executing: tool_ingest_downloaded_papers()")
        import glob
        from src.rag.embedding import embed_and_store
        
        pdf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_pdfs"))
        md_files = glob.glob(os.path.join(pdf_dir, "*.md"))
        
        if not md_files:
            print("      ⚠️ No new markdown papers found in raw_pdfs.")
            return "No new markdown papers found. Make sure to call download_paper first."
            
        ingested = []
        for file_path in md_files:
            filename = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            print(f"      📥 Ingesting recently downloaded paper: {filename}...")
            embed_and_store(content, filename)
            ingested.append(filename)
            
            # Move to processed so it's not ingested twice
            processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "user_uploads"))
            os.makedirs(processed_dir, exist_ok=True)
            os.rename(file_path, os.path.join(processed_dir, filename))
            
        return f"Successfully ingested {len(ingested)} papers: {', '.join(ingested)}. You can now search them via local memory."

    # Initialize MCP Server
    print("🔌 Starting Live Internet MCP connection...")
    arxiv_server = get_arxiv_server()
    await arxiv_server.connect()
    
    tools = await arxiv_server.list_tools()
    print("✅ ArXiv MCP linked! Available tools:", [t.name for t in tools])

    # Initialize our Orchestrator (The Brain)
    print("🧠 Initializing the Agent using OpenAI Agents SDK...")
    orchestrator = ResearchOrchestrator(model=model)
    agent = orchestrator.get_agent()
    
    # Inject tools and MCP Server directly into the agent
    print("🔌 Attaching Local Memory Tool and Live ArXiv MCP Server...")
    agent.tools = [tool_search_local_memory, tool_download_and_ingest_paper, tool_ingest_downloaded_papers]
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
