import asyncio
import json

from mcp.types import CallToolResult

from src.agent.tools.mcp_arxiv import get_arxiv_server


arxiv_mcp = get_arxiv_server()


async def run_mcp_server():
    print("Starting ArXiv MCP Server...")
    await arxiv_mcp.connect()
    print("ArXiv MCP Server is running. Press Ctrl+C to stop.")
    # download paper
    papers: CallToolResult = await arxiv_mcp.call_tool("search_papers", {"query": "deep learning for NLP"})
    # jsonify the results to print them nicely
    result_str = papers.content[0].text
    result_json = json.loads(result_str)
    for paper in result_json["papers"]:
        print(f"Found paper: {paper['title']} (ID: {paper['id']})")
        await arxiv_mcp.call_tool("download_paper", {"paper_id": paper["id"]})
        print(f"Downloading and ingesting paper ID {paper['id']}...")

    await arxiv_mcp.cleanup()

asyncio.run(run_mcp_server())
