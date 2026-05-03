import asyncio
import json
import os
import shutil
import uuid

import agents
from agents import (
    ItemHelpers,
    Runner,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    set_default_openai_client,
    tool_input_guardrail,
    tool_output_guardrail,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

from src.agent.orchestrator import ResearchOrchestrator
from src.agent.tools.mcp_arxiv import get_arxiv_server
from src.api.database import Evidence, SessionLocal, get_db, engine
from src.rag.embedding import embed_and_store
from src.rag.rerank import search_local_memory

load_dotenv()

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str


@tool_output_guardrail
def block_useless_memory(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    output_str = str(data.output)
    if "No context found" in output_str or output_str.startswith(
        "Local Paper Fragment:\n["
    ):
        return ToolGuardrailFunctionOutput.reject_content(
            message="Local memory search returned irrelevant references. DO NOT search local memory again. Use search_arxiv_internet instead.",
            output_info={"reason": "irrelevant_results"},
        )
    return ToolGuardrailFunctionOutput(output_info="Context is valid")


@tool_input_guardrail
def validate_arxiv_search(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    try:
        args = (
            json.loads(data.context.tool_arguments)
            if data.context.tool_arguments
            else {}
        )
    except json.JSONDecodeError:
        return ToolGuardrailFunctionOutput(output_info="Invalid JSON arguments")

    query = args.get("query", "").strip()
    if not query or len(query) < 3:
        return ToolGuardrailFunctionOutput.reject_content(
            message="🚨 Tool call blocked: Your search query is too short or empty. Please provide specific keywords.",
            output_info={"reason": "empty_query"},
        )
    return ToolGuardrailFunctionOutput(output_info="Query validated")


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    message_id = str(uuid.uuid4())

    async def event_generator():
        api_key = os.environ.get("OPENROUTER_API_KEY")

        yield {"event": "start", "data": json.dumps({"message_id": message_id})}

        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        set_default_openai_client(client)
        set_tracing_disabled(True)

        # 1. Initialize the SDK memory session (Creates DB tables automatically)
        chat_session = SQLAlchemySession.from_url(
            session_id=request.session_id,
            url="sqlite+aiosqlite:///./data/assistant.db",
            create_tables=True,
        )

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
                                session_id=request.session_id,
                            )
                        )
                    db.commit()
                return result_str
            except Exception as e:
                return f"ArXiv search failed: {str(e)}"

        @agents.function_tool(name_override="search_local_memory")
        def tool_search_local_memory(query: str) -> str:
            """Searches downloaded papers. Use after downloading."""
            context, evidence_list = search_local_memory(
                query, session_id=request.session_id
            )
            with SessionLocal() as db:
                for ev in evidence_list:
                    db.add(
                        Evidence(
                            message_id=message_id,
                            source=ev.get("source", "Unknown"),
                            title=ev.get("title", "Unknown"),
                            authors=ev.get("authors", "Unknown"),
                            score=ev.get("score", 1.0),
                            session_id=request.session_id,
                        )
                    )
                db.commit()
            return context

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
                        session_id=request.session_id,
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

            embed_and_store(content, downloaded_filename, session_id=request.session_id)
            shutil.move(file_path, os.path.join(processed_dir, downloaded_filename))

            return (
                f"Successfully downloaded paper {downloaded_filename}. "
                "CRITICAL INSTRUCTION: You MUST now immediately call `search_local_memory` using keywords "
                "from this paper to read it and answer the user."
            )

        @agents.function_tool
        def draft_research_outreach(
            author_name: str, paper_title: str, topic_summary: str
        ) -> str:
            """Drafts a personalized outreach email to a researcher."""
            return f"Subject: Inquiry regarding your work on {paper_title}\n\nDear {author_name},\n\nI've been researching {topic_summary} and was impressed by your paper..."

        arxiv_server = get_arxiv_server()
        await arxiv_server.connect()

        tool_search_arxiv.tool_input_guardrails = [validate_arxiv_search]
        tool_search_local_memory.tool_output_guardrails = [block_useless_memory]

        orchestrator = ResearchOrchestrator(
            model="openai/gpt-oss-20b:free"
        )  # Note: Upgrade this model when ready!
        agent = orchestrator.get_agent()
        agent.tools = [
            tool_search_local_memory,
            tool_download_and_ingest_paper,
            tool_search_arxiv,
            draft_research_outreach,
        ]

        runner = Runner()

        try:
            # 2. Pass the SDK session here so the agent remembers the context!
            result = runner.run_streamed(
                agent, input=request.message, session=chat_session
            )

            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(
                    event.data, ResponseTextDeltaEvent
                ):
                    yield {
                        "event": "token",
                        "data": json.dumps({"text": event.data.delta}),
                    }
                elif event.type == "run_item_stream_event":
                    if event.name == "tool_called":
                        yield {
                            "event": "thought",
                            "data": json.dumps({"action": event.item.tool_name}),
                        }
                    elif event.name == "tool_output":
                        yield {
                            "event": "tool_output",
                            "data": json.dumps(
                                {"output": str(event.item.output)[:500]}
                            ),
                        }
                    elif event.name == "message_output_created":
                        yield {
                            "event": "message",
                            "data": json.dumps(
                                {"text": ItemHelpers.text_message_output(event.item)}
                            ),
                        }
                elif event.type == "agent_updated_stream_event":
                    yield {
                        "event": "agent_switch",
                        "data": json.dumps({"agent": event.new_agent.name}),
                    }

            yield {"event": "complete", "data": json.dumps({"status": "done"})}

        except Exception as e:
            yield {"event": "error", "data": json.dumps({"detail": str(e)})}
        finally:
            await arxiv_server.cleanup()

    return EventSourceResponse(event_generator())


@router.get("/session/{session_id}/evidence")
async def get_session_evidence(session_id: str, db: Session = Depends(get_db)):
    """Fetches all unique evidence/papers gathered during a specific chat session."""
    evidence_records = (
        db.query(Evidence).filter(Evidence.session_id == session_id).all()
    )

    unique_evidence = {}
    for ev in evidence_records:
        if ev.title not in unique_evidence:
            unique_evidence[ev.title] = {
                "source": ev.source,
                "title": ev.title,
                "authors": ev.authors,
                "score": ev.score,
            }

    return {"session_id": session_id, "evidence": list(unique_evidence.values())}
