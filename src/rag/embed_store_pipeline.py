# main_ingestion.py

from src.rag.embedding import embed_and_store

# 1. This represents the string you got from your 'read_paper' MCP tool
mcp_markdown_output = """
Distributionally Robust Receive Combining
Shixiong Wang, Wei Dai, and Geoffrey Ye Li...
Abstract
This article investigates signal estimation in wireless transmission...
Index Terms:
Wireless Transmission, Smart Antenna, Machine Learning...
I
Introduction
In wireless transmission, detection and estimation of transmitted signals...
"""

# 2. Fire the pipeline
# We name the file appropriately so we can find it in ChromaDB later
embed_and_store(
    raw_text=mcp_markdown_output, filename="Distributionally_Robust_Combining.md"
)
