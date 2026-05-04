import json
import os
import shutil

import agents

from src.api.database import Evidence, SessionLocal
from src.rag.embedding import embed_and_store
from src.rag.rerank import search_local_memory


def build_chat_tools(*, message_id: str, session_id: str, arxiv_server):
    search_arxiv_tool = _build_search_arxiv_tool(
        message_id=message_id,
        session_id=session_id,
        arxiv_server=arxiv_server,
    )
    search_local_memory_tool = _build_search_local_memory_tool(
        message_id=message_id,
        session_id=session_id,
    )
    download_and_ingest_tool = _build_download_and_ingest_paper_tool(
        message_id=message_id,
        session_id=session_id,
        arxiv_server=arxiv_server,
    )

    return {
        "search_arxiv_tool": search_arxiv_tool,
        "search_local_memory_tool": search_local_memory_tool,
        "download_and_ingest_tool": download_and_ingest_tool,
        "tools": [
            search_local_memory_tool,
            download_and_ingest_tool,
            search_arxiv_tool,
        ],
    }


def _build_search_arxiv_tool(*, message_id: str, session_id: str, arxiv_server):
    @agents.function_tool(name_override="search_arxiv_internet")
    async def tool_search_arxiv(
        query: str,
        max_results: int | None = 10,
        date_from: str | None = None,
        date_to: str | None = None,
        categories: list[str] | None = None,
        sort_by: str = "relevance",
    ) -> str:
        """Searches ArXiv for papers."""
        try:
            if isinstance(categories, str):
                categories = [categories]
            payload = {
                "query": query,
                "max_results": max_results,
                "sort_by": sort_by,
            }
            if date_from:
                payload["date_from"] = date_from
            if date_to:
                payload["date_to"] = date_to
            if categories:
                payload["categories"] = categories

            result = await arxiv_server.call_tool("search_papers", payload)
            result_str = result.content[0].text
            papers = json.loads(result_str).get("papers", [])

            if not papers:
                return "No papers found."

            with SessionLocal() as db:
                for paper in papers[:5]:
                    db.add(
                        Evidence(
                            message_id=message_id,
                            source=f"ArXiv Search ({paper.get('id', 'Unknown')})",
                            title=paper.get("title", "Unknown"),
                            authors=", ".join(paper.get("authors", ["Unknown"])),
                            score=1.0,
                            session_id=session_id,
                        )
                    )
                db.commit()
            return result_str
        except Exception as e:
            return f"ArXiv search failed: {str(e)}"

    return tool_search_arxiv


def _build_search_local_memory_tool(*, message_id: str, session_id: str):
    @agents.function_tool(name_override="search_local_memory")
    def tool_search_local_memory(query: str) -> str:
        """Searches downloaded papers. Use after downloading."""
        context, evidence_list = search_local_memory(query, session_id=session_id)
        with SessionLocal() as db:
            for ev in evidence_list:
                db.add(
                    Evidence(
                        message_id=message_id,
                        source=ev.get("source", "Unknown"),
                        title=ev.get("title", "Unknown"),
                        authors=ev.get("authors", "Unknown"),
                        score=ev.get("score", 1.0),
                        session_id=session_id,
                    )
                )
            db.commit()
        return context

    return tool_search_local_memory


def _build_download_and_ingest_paper_tool(
    *, message_id: str, session_id: str, arxiv_server
):
    @agents.function_tool(name_override="download_and_ingest_paper")
    async def tool_download_and_ingest_paper(paper_id: str) -> str:
        """Downloads ArXiv paper by ID."""
        with SessionLocal() as db:
            db.add(
                Evidence(
                    message_id=message_id,
                    source="ArXiv Live API",
                    title=f"ArXiv Paper {paper_id}",
                    authors="Unknown",
                    score=1.0,
                    session_id=session_id,
                )
            )
            db.commit()

        try:
            await arxiv_server.call_tool("download_paper", {"paper_id": paper_id})
        except Exception as e:
            return f"Error: {str(e)}"

        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        raw_dir = os.path.join(base_dir, "data", "raw_pdfs")
        processed_dir = os.path.join(base_dir, "data", "user_uploads")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        downloaded_filename = next(
            (
                f
                for f in os.listdir(raw_dir)
                if f.startswith(paper_id) and f.endswith(".md")
            ),
            None,
        )
        if not downloaded_filename:
            return f"Paper {paper_id} not found."

        file_path = os.path.join(raw_dir, downloaded_filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        embed_and_store(content, downloaded_filename, session_id=session_id)
        shutil.move(file_path, os.path.join(processed_dir, downloaded_filename))

        return (
            f"Successfully downloaded paper {downloaded_filename}. "
            "CRITICAL INSTRUCTION: You MUST now immediately call `search_local_memory` using keywords "
            "from this paper to read it and answer the user."
        )

    return tool_download_and_ingest_paper

