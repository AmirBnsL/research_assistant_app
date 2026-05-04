import os
import shutil

import agents

from src.rag.embedding import embed_and_store
from src.rag.rerank import search_local_memory


def build_cli_tools(*, arxiv_server):
    search_local_memory_tool = _build_search_local_memory_tool()
    download_and_ingest_tool = _build_download_and_ingest_paper_tool(
        arxiv_server=arxiv_server,
    )
    ingest_downloaded_papers_tool = _build_ingest_downloaded_papers_tool()

    return {
        "search_local_memory_tool": search_local_memory_tool,
        "download_and_ingest_tool": download_and_ingest_tool,
        "ingest_downloaded_papers_tool": ingest_downloaded_papers_tool,
        "tools": [
            search_local_memory_tool,
            download_and_ingest_tool,
            ingest_downloaded_papers_tool,
        ],
    }


def _build_search_local_memory_tool():
    @agents.function_tool
    def tool_search_local_memory(query: str) -> str:
        """Searches the local vector database of uploaded and processed academic papers for the query."""
        print(f"\n   -> 🛠️ Executing: search_local_memory(query={query})")
        res = search_local_memory(query)
        print("      🗄️ Local DB successfully returned chunk memories.")
        return res

    return tool_search_local_memory


def _build_download_and_ingest_paper_tool(*, arxiv_server):
    @agents.function_tool
    async def tool_download_and_ingest_paper(paper_id: str) -> str:
        """
        Downloads a paper from ArXiv by ID (e.g., "2301.07041") and automatically ingests it to ChromaDB.
        MUST BE USED instead of the raw `download_paper` tool to prevent AI context window overflow.
        """
        print(f"\n   -> 🛠️ Executing: tool_download_and_ingest_paper({paper_id})")
        try:
            await arxiv_server.call_tool("download_paper", {"paper_id": paper_id})

            pdf_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw_pdfs")
            )
            md_files = [
                file_name for file_name in os.listdir(pdf_dir) if file_name.endswith(".md")
            ]

            if not md_files:
                return "No new markdown papers found. Make sure the download succeeded."

            ingested = []
            for file_name in md_files:
                file_path = os.path.join(pdf_dir, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                print(f"      📥 Ingesting recently downloaded paper: {file_name}...")
                embed_and_store(content, file_name)
                ingested.append(file_name)

                processed_dir = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "data", "user_uploads")
                )
                os.makedirs(processed_dir, exist_ok=True)
                shutil.move(file_path, os.path.join(processed_dir, file_name))

            return (
                f"Successfully downloaded and ingested {len(ingested)} papers: {', '.join(ingested)}. "
                "You can now search them via search_local_memory!"
            )
        except Exception as e:
            print(f"      ❌ Exception occurred during download or ingestion: {e}")
            return f"Failed to download and ingest paper {paper_id}. Error: {e}"

    return tool_download_and_ingest_paper


def _build_ingest_downloaded_papers_tool():
    @agents.function_tool
    def tool_ingest_downloaded_papers() -> str:
        """
        Scans the `data/raw_pdfs` folder for newly downloaded ArXiv markdown papers from the `download_paper` tool,
        chunks them, and embeds them into the local ChromaDB database.
        ALWAYS call this tool immediately after successfully using `download_paper` or `read_paper`!
        """
        print("\n   -> 🛠️ Executing: tool_ingest_downloaded_papers()")

        pdf_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw_pdfs")
        )
        md_files = [
            file_name for file_name in os.listdir(pdf_dir) if file_name.endswith(".md")
        ]

        if not md_files:
            print("      ⚠️ No new markdown papers found in raw_pdfs.")
            return "No new markdown papers found. Make sure to call download_paper first."

        ingested = []
        for file_name in md_files:
            file_path = os.path.join(pdf_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            print(f"      📥 Ingesting recently downloaded paper: {file_name}...")
            embed_and_store(content, file_name)
            ingested.append(file_name)

            processed_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "data", "user_uploads")
            )
            os.makedirs(processed_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(processed_dir, file_name))

        return (
            f"Successfully ingested {len(ingested)} papers: {', '.join(ingested)}. "
            "You can now search them via local memory."
        )

    return tool_ingest_downloaded_papers