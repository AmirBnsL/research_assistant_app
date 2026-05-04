import gradio as gr
import requests
from requests.auth import HTTPBasicAuth
import json
import uuid
import sseclient

# Point this to your running FastAPI server
API_BASE_URL = "http://localhost:8000/api/chat"
AUTH = HTTPBasicAuth("admin", "admin")


def chat_stream(message, history, session_id):
    """
    Sends the user's message to the FastAPI backend and streams the SSE response
    back to the Gradio Chatbot interface.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    # Append the user's message to the chat history
    history.append({"role": "user", "content": message})
    # Add an empty assistant message that we will stream text into
    history.append({"role": "assistant", "content": ""})
    yield history, session_id

    payload = {"message": message, "session_id": session_id}

    try:
        # Stream=True is critical to capture the Server-Sent Events
        response = requests.post(f"{API_BASE_URL}/stream", json=payload, stream=True, auth=AUTH)
        response.raise_for_status()

        client = sseclient.SSEClient(response.iter_lines())

        for event in client.events():
            if not event.data:
                continue

            try:
                data = json.loads(event.data)
            except json.JSONDecodeError:
                continue

            # Handle the streaming tokens
            if event.event == "token":
                history[-1]["content"] += data["text"]
                yield history, session_id

            # Expose the agent's internal thoughts (Tool Calling)
            elif event.event == "thought":
                thought_msg = (
                    f"\n> *⚙️ Agent Action: Calling tool `{data['action']}`*\n\n"
                )
                history[-1]["content"] += thought_msg
                yield history, session_id

            # Expose error messages
            elif event.event == "error":
                history[-1]["content"] += f"\n\n**Error:** {data.get('detail')}"
                yield history, session_id

            # Stop generator when complete
            elif event.event == "complete":
                break

    except Exception as e:
        history[-1][
            "content"
        ] += f"\n\n**Connection Error:** Make sure your FastAPI backend is running. Details: {str(e)}"
        yield history, session_id


def fetch_evidence(session_id):
    """
    Calls the evidence endpoint to retrieve papers stored in ChromaDB/SQLite.
    Formats the JSON response into a nice Markdown string for the side panel.
    """
    if not session_id:
        return "No session active."

    try:
        response = requests.get(f"{API_BASE_URL}/session/{session_id}/evidence", auth=AUTH)
        response.raise_for_status()
        data = response.json()

        evidence_list = data.get("evidence", [])
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
    # Hidden state to persist the session ID across rerenders
    session_state = gr.State(lambda: str(uuid.uuid4()))

    gr.Markdown("# 🧠 Agentic arXiv RAG")
    gr.Markdown(
        "Ask questions about scientific literature. The agent will autonomously search arXiv, chunk the PDFs, and synthesize an answer."
    )

    with gr.Row():
        # Left Column: Chat Interface
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=600)
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="E.g., What is the impact of RoPE in modern transformers?",
                    label="Query",
                    scale=8,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

        # Right Column: Evidence Tracker
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Context & Evidence")
            gr.Markdown("Papers dynamically retrieved by the agent will appear here.")
            evidence_btn = gr.Button("🔄 Refresh Evidence", size="sm")
            evidence_display = gr.Markdown(value="*Awaiting queries...*")

    # ==========================================
    # Event Routing
    # ==========================================

    # 1. Trigger stream when pressing Enter in textbox
    msg_submit_event = msg_input.submit(
        chat_stream,
        inputs=[msg_input, chatbot, session_state],
        outputs=[chatbot, session_state],
    )

    # 2. Trigger stream when clicking the Send button
    btn_submit_event = submit_btn.click(
        chat_stream,
        inputs=[msg_input, chatbot, session_state],
        outputs=[chatbot, session_state],
    )

    # 3. After generation finishes, clear the input box and auto-refresh the evidence panel
    def clear_input():
        return ""

    msg_submit_event.then(clear_input, outputs=[msg_input]).then(
        fetch_evidence, inputs=[session_state], outputs=[evidence_display]
    )
    btn_submit_event.then(clear_input, outputs=[msg_input]).then(
        fetch_evidence, inputs=[session_state], outputs=[evidence_display]
    )

    # 4. Allow manual evidence refresh
    evidence_btn.click(
        fetch_evidence, inputs=[session_state], outputs=[evidence_display]
    )

if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
