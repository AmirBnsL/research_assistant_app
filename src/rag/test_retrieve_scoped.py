import pytest
import os
import shutil
import uuid
import chromadb
from chromadb.utils import embedding_functions

from src.rag.retrieve import retrieve_top_k
from src.rag.embedding import embed_and_store

@pytest.fixture
def temp_chroma_db():
    # Setup
    test_db_path = "./data/chromadb_store_test"
    
    # Store original db_path
    original_db_path = None
    import src.rag.retrieve as retrieve_module
    import src.rag.embedding as embedding_module
    
    # Mocking paths is a bit tricky if they're hardcoded, let's just use the actual implementation
    # and clean up afterwards.
    
    yield
    
    # Teardown - this is a simplified test without patching the exact path inside retrieve.py
    # For a real robust test, you'd patch os.path.join in those modules.
    
def test_retrieve_filters_by_session_id():
    # 1. Store docs with session_id
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())
    
    embed_and_store("Machine learning for healthcare.", "health_paper.pdf", session_id=session_a)
    embed_and_store("Machine learning for finance.", "finance_paper.pdf", session_id=session_b)
    
    # 2. Retrieve with session_a
    docs_a, metas_a = retrieve_top_k("Machine learning", k=5, session_id=session_a)
    assert len(docs_a) > 0, "Should retrieve chunks for session_a"
    for meta in metas_a:
        assert meta["session_id"] == session_a, "Should only contain session_a metadata"
        
    # 3. Retrieve with session_b
    docs_b, metas_b = retrieve_top_k("Machine learning", k=5, session_id=session_b)
    assert len(docs_b) > 0, "Should retrieve chunks for session_b"
    for meta in metas_b:
        assert meta["session_id"] == session_b, "Should only contain session_b metadata"

