import pytest
import time
import json
import os
from fastapi.testclient import TestClient
from src.api.server import app
from src.api.database import SessionLocal, Document, Evidence, engine, Base

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    # Ensure tables are created
    Base.metadata.create_all(bind=engine)
    yield
    # Cleanup after all tests in this module
    # Base.metadata.drop_all(bind=engine) # Optional: don't drop if we want to inspect

def test_full_rag_flow():
    session_id = f"test_session_{int(time.time())}"
    filename = "test_paper.pdf"
    
    # 1. Upload Document
    print(f"\n[Test] Uploading {filename} with session {session_id}...")
    # Create a small dummy text file (pretending to be PDF for the tool)
    with open(filename, "w") as f:
        f.write("This is a test document about Quantum Computing in 2024. It discusses qubits and entanglement.")
    
    try:
        with open(filename, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": (filename, f, "application/pdf")},
                data={"session_id": session_id}
            )
        
        assert response.status_code == 200
        upload_data = response.json()
        doc_id = upload_data["doc_id"]
        assert upload_data["session_id"] == session_id
        
        # 2. Wait for Processing
        print(f"[Test] Waiting for document {doc_id} to process...")
        timeout = 30
        start_time = time.time()
        processed = False
        while time.time() - start_time < timeout:
            # We can't easily test SSE with TestClient in a simple loop, 
            # but we can check the Document table or a status endpoint if it exists.
            # Let's check the list endpoint.
            list_resp = client.get("/api/documents/")
            assert list_resp.status_code == 200
            docs = list_resp.json()
            doc = next((d for d in docs if d["id"] == doc_id), None)
            if doc and doc["status"] == "completed":
                processed = True
                break
            time.sleep(1)
        
        assert processed, "Document processing timed out or failed"
        print("[Test] Document processed successfully.")

        # 3. Chat with RAG
        print(f"[Test] Sending chat message for session {session_id}...")
        chat_payload = {
            "message": "What is the main topic of my uploaded document?",
            "session_id": session_id
        }
        
        # TestClient.post with stream=True for SSE
        with client.stream("POST", "/api/chat/stream", json=chat_payload) as response:
            assert response.status_code == 200
            
            # Read SSE stream
            rag_called = False
            final_answer = ""
            
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        # Check for thought event indicating RAG tool call
                        if data.get("event") == "thought" or (isinstance(data, dict) and "action" in data):
                             if data.get("action") == "search_local_memory" or data.get("thought") == "search_local_memory":
                                 rag_called = True
                                 print("[Test] RAG tool 'search_local_memory' was called.")
                        
                        # Accumulate message tokens
                        if data.get("event") == "token":
                            final_answer += data.get("text", "")
                            
                    except json.JSONDecodeError:
                        continue
            
            print(f"[Test] Final Answer: {final_answer}")
            assert len(final_answer) > 0, "Agent returned an empty answer"
            # Since we only uploaded one doc for this session, it should be able to answer
            # We check if it mentions 'Quantum Computing' or 'qubits'
            assert any(word in final_answer.lower() for word in ["quantum", "qubits", "entanglement", "computing"]), \
                "Answer does not seem to contain information from the document"

        # 4. Verify Evidence exists in DB for this session
        with SessionLocal() as db:
            evidences = db.query(Evidence).filter(Evidence.session_id == session_id).all()
            assert len(evidences) > 0, "No evidence records found in DB for the session"
            print(f"[Test] Found {len(evidences)} evidence records.")

    finally:
        if os.path.exists(filename):
            os.remove(filename)

def test_session_isolation():
    # Upload to Session A, then query with Session B - should NOT see Session A's data
    session_a = f"session_a_{int(time.time())}"
    session_b = f"session_b_{int(time.time())}"
    
    filename_a = "secret_a.pdf"
    with open(filename_a, "w") as f:
        f.write("The secret code for session A is ALPHA-99.")
        
    try:
        # Upload to A
        with open(filename_a, "rb") as f:
            client.post("/api/documents/upload", files={"file": (filename_a, f, "application/pdf")}, data={"session_id": session_a})
        
        # Wait a bit for processing (simplified for isolation test)
        time.sleep(5) 
        
        # Query with Session B
        chat_payload = {
            "message": "What is the secret code?",
            "session_id": session_b
        }
        
        with client.stream("POST", "/api/chat/stream", json=chat_payload) as response:
            final_answer = ""
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if data.get("event") == "token":
                        final_answer += data.get("text", "")
            
            print(f"[Test Isolation] Session B Answer: {final_answer}")
            assert "ALPHA-99" not in final_answer, "Session B accessed Session A's data!"
            print("[Test Isolation] Success: Session B did not see Session A's data.")
            
    finally:
        if os.path.exists(filename_a):
            os.remove(filename_a)
