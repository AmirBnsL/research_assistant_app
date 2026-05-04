# AI Research Assistant

An autonomous research assistant that combines FastAPI, Gradio, and an Agentic RAG pipeline to search, download, and analyze academic papers from ArXiv.

## Features

- **Autonomous Agent:** Uses the OpenAI Agents SDK to decide when to search ArXiv vs. query local memory.
- **RAG Pipeline:** Ingests academic papers (PDF to Markdown), chunks them, and stores them in ChromaDB for semantic retrieval.
- **Tools:** Integrated with ArXiv MCP for live internet searches and local vector storage for paper analysis.
- **Dual Interface:** Accessible via a REST API (FastAPI) or a user-friendly Chat UI (Gradio).

## Project Structure

- `src/agent/`: Brain of the assistant, orchestrator, and tool definitions.
- `src/api/`: FastAPI server and routers for chat and document management.
- `src/rag/`: Embedding, retrieval, and reranking logic.
- `src/gradio/`: Gradio-based frontend for interactive chat.
- `data/`: Storage for raw PDFs and ChromaDB vector store.

## Getting Started

### 1. Prerequisites
- Python 3.10+
- OpenAI API Key (or OpenRouter)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1 # or your proxy URL
```

### 4. Running the Application

You need to run the backend and frontend separately:

**Run Backend (API):**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
uvicorn src.api.server:app --reload
```

**Run Frontend (UI):**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python src/gradio/gradio_app.py
```

## Usage
1. Open the Gradio URL (usually `http://127.0.0.1:7860`).
2. Type a research topic (e.g., "Find papers on Graph Neural Networks").
3. The agent will autonomously search ArXiv, download relevant papers, and answer your questions based on the ingested content.
