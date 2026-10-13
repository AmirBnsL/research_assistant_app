import gradio as gr
import os
import json
import uuid
import asyncio
from dotenv import load_dotenv

# --- MONKEY PATCH ---
import agents._tool_identity
original_get_tool_call_name = agents._tool_identity.get_tool_call_name
def patch_get_tool_call_name(tool_call):
    name = original_get_tool_call_name(tool_call)
    if name and "<|channel|>" in name:
        name = name.split("<|channel|>")[0]
        if hasattr(tool_call, "name"):
            tool_call.name = name
        elif isinstance(tool_call, dict) and "name" in tool_call:
            tool_call["name"] = name
    return name
agents._tool_identity.get_tool_call_name = patch_get_tool_call_name
# --------------------

from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    ItemHelpers,
    Runner,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

from src.agent.guardrails import block_useless_memory, validate_arxiv_search
from src.agent.orchestrator import ResearchOrchestrator
from src.agent.tools import build_chat_tools, get_arxiv_server
from src.api.database import Evidence, SessionLocal

load_dotenv()

def ensure_session(session_id):
    if isinstance(session_id, str) and session_id and "<function" not in session_id:
        return session_id
    return str(uuid.uuid4())

def apply_final_message(content, streamed, final, stream_at=None):
    """Keep streamed tokens, and use the final message without writing it twice."""
    if not final:
        return content
    if streamed and stream_at is not None:
        end = stream_at + len(streamed)
        if content[stream_at:end] == streamed:
            if streamed == final:
                return content
            return content[:stream_at] + final + content[end:]
        if content.endswith(streamed):
            if streamed == final:
                return content
            return content[: -len(streamed)] + final
    if content.endswith(final):
        return content
    return content + final

async def chat_stream(message, history, session_id):
    if callable(session_id):
        try:
            session_id = session_id()
        except:
            session_id = str(uuid.uuid4())
    if not session_id or not isinstance(session_id, str) or "<function" in str(session_id):
        session_id = str(uuid.uuid4())

    # In Gradio 6 older format without type="messages", we just use list
    history = list(history or [])
    if not message or not str(message).strip():
        yield "", history, session_id
        return

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    yield "", history, session_id

    message_id = str(uuid.uuid4())
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    print(f"Using api key: {'Yes' if api_key else 'No'}")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    
    # --- MONKEY PATCH OPENAI CLIENT ---
    original_create = client.chat.completions.create
    async def patched_create(*args, **kwargs):
        response = await original_create(*args, **kwargs)
        if kwargs.get("stream"):
            async def stream_wrapper():
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.tool_calls:
                        for tc in chunk.choices[0].delta.tool_calls:
                            if tc.function and tc.function.name and "<|channel|>" in tc.function.name:
                                tc.function.name = tc.function.name.split("<|channel|>")[0]
                    yield chunk
            return stream_wrapper()
        return response
    client.chat.completions.create = patched_create
    # ----------------------------------

    set_default_openai_client(client)
    set_tracing_disabled(True)

    # Initialize the SDK memory session
    chat_session = SQLAlchemySession.from_url(
        session_id=session_id,
        url="sqlite+aiosqlite:///./data/assistant.db",
        create_tables=True,
    )

    arxiv_server = get_arxiv_server()
    await arxiv_server.connect()

    chat_tools = build_chat_tools(
        message_id=message_id,
        session_id=session_id,
        arxiv_server=arxiv_server,
    )

    chat_tools["search_arxiv_tool"].tool_input_guardrails = [validate_arxiv_search]
    chat_tools["search_local_memory_tool"].tool_output_guardrails = [
        block_useless_memory
    ]

    orchestrator = ResearchOrchestrator(
        model="openai/gpt-oss-20b:free"
    )
    agent = orchestrator.get_agent()
    agent.tools = chat_tools["tools"]

    runner = Runner()

    try:
        result = runner.run_streamed(
            agent, input=message, session=chat_session
        )

        streamed = ""
        stream_at = None
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                if event.data.delta:
                    if stream_at is None:
                        stream_at = len(history[-1]["content"])
                    streamed += event.data.delta
                    history[-1]["content"] += event.data.delta
                    yield "", history, session_id

            elif event.type == "run_item_stream_event":
                if event.name == "tool_called":
                    thought_msg = (
                        f"\n> *⚙️ Agent Action: Calling tool `{event.item.tool_name}`*\n\n"
                    )
                    history[-1]["content"] += thought_msg
                    yield "", history, session_id

                elif event.name == "tool_output":
                    output_str = str(event.item.output)[:500]
                    if "An error occurred" in output_str:
                        history[-1]["content"] += f"\n> ⚠️ *Tool Error: {output_str}*\n\n"
                    yield "", history, session_id

                elif event.name == "message_output_created":
                    txt = ItemHelpers.text_message_output(event.item)
                    if txt:
                        updated = apply_final_message(
                            history[-1]["content"], streamed, txt, stream_at
                        )
                        streamed = ""
                        stream_at = None
                        if updated != history[-1]["content"]:
                            history[-1]["content"] = updated
                            yield "", history, session_id
                    else:
                        streamed = ""
                        stream_at = None

    except Exception as e:
        history[-1]["content"] += f"\n\n**Error:** {str(e)}"
        yield "", history, session_id
    finally:
        await arxiv_server.cleanup()

def fetch_evidence(session_id):
    """
    Fetches evidence from the local database directly.
    """
    if not session_id:
        return "No session active."

    try:
        with SessionLocal() as db:
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
                    
            evidence_list = list(unique_evidence.values())

        if not evidence_list:
            return "No evidence retrieved for this session yet."

        md_output = ""
        for i, ev in enumerate(evidence_list, 1):
            md_output += f"### {i}. {ev.get('title', 'Unknown Title')}\n"
            md_output += f"**Authors:** {ev.get('authors', 'Unknown')}\n\n"
            md_output += (
                f"**Source:** [{ev.get('source', 'Link')}]({ev.get('source', '#')})\n"
            )
            md_output += f"**Relevance Score:** {ev.get('score', 'N/A')}\n\n---\n"

        return md_output

    except Exception as e:
        return f"Error fetching evidence: {str(e)}"

# ==========================================
# Gradio UI Layout
# ==========================================
with gr.Blocks() as demo:
    session_state = gr.State(None)

    gr.Markdown("# 🧠 Agentic arXiv RAG")
    gr.Markdown(
        "Ask questions about scientific literature. The agent will autonomously search arXiv, chunk the PDFs, and synthesize an answer."
    )

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=600)
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="E.g., What is the impact of RoPE in modern transformers?",
                    label="Query",
                    scale=8,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### 📄 Context & Evidence")
            gr.Markdown("Papers dynamically retrieved by the agent will appear here.")
            evidence_btn = gr.Button("🔄 Refresh Evidence", size="sm")
            evidence_display = gr.Markdown(value="*Awaiting queries...*")

    msg_submit_event = msg_input.submit(
        chat_stream,
        inputs=[msg_input, chatbot, session_state],
        outputs=[msg_input, chatbot, session_state],
    )

    btn_submit_event = submit_btn.click(
        chat_stream,
        inputs=[msg_input, chatbot, session_state],
        outputs=[msg_input, chatbot, session_state],
    )

    def clear_input():
        return ""

    msg_submit_event.then(clear_input, outputs=[msg_input]).then(
        fetch_evidence, inputs=[session_state], outputs=[evidence_display]
    )
    btn_submit_event.then(clear_input, outputs=[msg_input]).then(
        fetch_evidence, inputs=[session_state], outputs=[evidence_display]
    )

    evidence_btn.click(
        fetch_evidence, inputs=[session_state], outputs=[evidence_display]
    )

    demo.load(ensure_session, inputs=[session_state], outputs=[session_state])

if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
