import os
import uuid
import chromadb
from chromadb.utils import embedding_functions
from src.rag.ingestion import chunk_academic_paper


def embed_and_store_batch(
    chunks: list[str],
    filename: str,
    doc_id: str,
    job_progress: dict,
    start_progress: int = 20,
    session_id: str | None = None,
):
    total_chunks = len(chunks)
    if total_chunks == 0:
        job_progress[doc_id] = 100
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    db_path = os.path.join(project_root, "data", "chromadb_store")

    client = chromadb.PersistentClient(path=db_path)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="academic_papers",
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 10
    remaining_progress = 100 - start_progress
    print(f"💾 Embedding {total_chunks} chunks for '{filename}'...")

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_ids = [str(uuid.uuid4()) for _ in batch_chunks]
        batch_metadatas = [
            {
                "source": filename,
                "chunk_index": i + j,
                "doc_id": doc_id,
                **({"session_id": session_id} if session_id else {}),
            }
            for j in range(len(batch_chunks))
        ]
        collection.add(documents=batch_chunks, ids=batch_ids, metadatas=batch_metadatas)

        current_processed = min(i + batch_size, total_chunks)
        job_progress[doc_id] = start_progress + int(
            (current_processed / total_chunks) * remaining_progress
        )

    print(f"✅ Successfully ingested '{filename}' into ChromaDB!")
    job_progress[doc_id] = 100


def embed_and_store(content: str, filename: str, session_id: str | None = None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    db_path = os.path.join(project_root, "data", "chromadb_store")

    client = chromadb.PersistentClient(path=db_path)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="academic_papers",
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

    where_clause = (
        {"$and": [{"source": filename}, {"session_id": session_id}]}
        if session_id
        else {"source": filename}
    )
    existing_records = collection.get(where=where_clause, include=["metadatas"])

    if existing_records and len(existing_records["ids"]) > 0:
        print(f"⏩ Document '{filename}' already exists. Skipping.")
        return

    print(f"⚙️ Chunking document: {filename}...")
    chunks = chunk_academic_paper(content)

    if not chunks:
        return

    chunk_ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": filename,
            "chunk_index": i,
            **({"session_id": session_id} if session_id else {}),
        }
        for i in range(len(chunks))
    ]
    collection.upsert(documents=chunks, ids=chunk_ids, metadatas=metadatas)
    print(
        f"✅ Successfully ingested '{filename}' ({len(chunks)} chunks) into ChromaDB!"
    )
