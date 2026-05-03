import os
from agents.mcp import MCPServerStdio

def get_arxiv_server() -> MCPServerStdio:
    """Returns an MCPServerStdio instance for the openai-agents SDK."""
    abs_storage = os.path.abspath("./data/raw_pdfs/")
    os.makedirs(abs_storage, exist_ok=True)
    
    return MCPServerStdio(
        params={
            "command": "uvx",
            "args": ["arxiv-mcp-server", "--storage-path", abs_storage]
        },
        name = "Arxiv-Search")
    
    
    

