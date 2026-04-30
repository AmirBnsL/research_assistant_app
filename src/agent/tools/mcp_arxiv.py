"""MCP Arxiv bridge placeholder."""
from agents.mcp import MCPServerStdio


async def get_arxiv_server():
    # This automatically downloads and runs the blazickjp server
    # The --storage-path tells it exactly where to put the PDFs for your RAG
    return MCPServerStdio(
        params={
            "command": "uvx",
            "args": ["arxiv-mcp-server", "--storage-path", "./data/raw_pdfs/"],
        },
        name="ArXiv-Search-Engine",
    )
