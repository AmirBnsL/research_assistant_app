import json
import os
import uuid

from agents import (
    ItemHelpers,
    Runner,
    set_default_openai_client,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

from src.agent.guardrails import block_useless_memory, validate_arxiv_search
from src.agent.orchestrator import ResearchOrchestrator
from src.agent.tools import build_chat_tools, get_arxiv_server
from src.api.database import Evidence, get_db

load_dotenv()

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str


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

        arxiv_server = get_arxiv_server()
        await arxiv_server.connect()

        chat_tools = build_chat_tools(
            message_id=message_id,
            session_id=request.session_id,
            arxiv_server=arxiv_server,
        )

        chat_tools["search_arxiv_tool"].tool_input_guardrails = [validate_arxiv_search]
        chat_tools["search_local_memory_tool"].tool_output_guardrails = [
            block_useless_memory
        ]

        orchestrator = ResearchOrchestrator(
            model="openai/gpt-oss-20b:free"
        )  # Note: Upgrade this model when ready!
        agent = orchestrator.get_agent()
        agent.tools = chat_tools["tools"]

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
