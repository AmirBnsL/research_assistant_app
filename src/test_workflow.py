import json
import pytest
from io import BytesIO
from fastapi.testclient import TestClient

# Import your main FastAPI app instance
# Adjust this import based on where your `app = FastAPI()` is defined!
from src.main import app

client = TestClient(app)

SESSION_ID = "pytest_test_session_123"

# We use a global variable to pass the doc_id between tests for this workflow
test_context = {"doc_id": None}


def test_document_upload():
    """Test that a PDF can be uploaded and database records are initialized."""
    # Create a dummy PDF in memory (no need to write to disk for the test upload)
    dummy_file_content = b"%PDF-1.4\nDummy PDF content for testing."
    dummy_file = BytesIO(dummy_file_content)

    response = client.post(
        "/api/documents/upload",
        files={"file": ("test_paper.pdf", dummy_file, "application/pdf")},
        data={"session_id": SESSION_ID},
    )

    assert response.status_code == 200
    data = response.json()

    assert "doc_id" in data
    assert data["status"] == "Processing started"
    assert data["session_id"] == SESSION_ID

    # Save the doc_id for subsequent tests
    test_context["doc_id"] = data["doc_id"]


def test_list_documents():
    """Test that the uploaded document appears in the database list."""
    response = client.get("/api/documents/")

    assert response.status_code == 200
    docs = response.json()
    assert isinstance(docs, list)

    # Verify our uploaded document is in the list
    doc_ids = [doc["id"] for doc in docs]
    assert test_context["doc_id"] in doc_ids


def test_document_progress_stream():
    """Test the Server-Sent Events (SSE) progress endpoint."""
    doc_id = test_context["doc_id"]

    # TestClient allows us to stream responses seamlessly
    with client.stream("GET", f"/api/documents/progress/{doc_id}") as response:
        assert response.status_code == 200

        # Read the first event from the SSE stream
        first_event = next(response.iter_lines())
        assert first_event.startswith("event:") or first_event.startswith("data:")


def test_chat_stream_initialization():
    """Test that the agent stream endpoint connects and returns the start event."""
    payload = {"message": "Hello, what papers do you have?", "session_id": SESSION_ID}

    # We test just the connection and the first yielded event
    # to avoid waiting for the full LLM response during fast unit tests.
    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200

        lines = [line for line in response.iter_lines() if line.strip()]

        # The first meaningful event should be our "start" event with the message_id
        start_event = lines[0]
        start_data = lines[1]

        assert start_event == "event: start"
        assert "message_id" in start_data


def test_session_evidence():
    """Test that we can retrieve evidence for the current session."""
    response = client.get(f"/api/chat/session/{SESSION_ID}/evidence")

    assert response.status_code == 200
    data = response.json()

    assert data["session_id"] == SESSION_ID
    assert "evidence" in data
    assert isinstance(data["evidence"], list)


def test_delete_document():
    """Test that the document and its metadata can be deleted cleanly."""
    doc_id = test_context["doc_id"]

    response = client.delete(f"/api/documents/{doc_id}")
    assert response.status_code == 204  # 204 No Content means successful deletion

    # Verify it's actually gone from the list
    list_response = client.get("/api/documents/")
    docs = list_response.json()
    doc_ids = [doc["id"] for doc in docs]
    assert doc_id not in doc_ids
