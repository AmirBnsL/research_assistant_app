import os
import chromadb
from chromadb.utils import embedding_functions


def retrieve_top_k(
    query: str, k: int = 20, session_id: str | None = None
) -> tuple[list[str], list[dict]]:
    print(f"🔎 Retrieving top {k} candidate chunks for query: '{query}'...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    db_path = os.path.join(project_root, "data", "chromadb_store")

    client = chromadb.PersistentClient(path=db_path)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        collection = client.get_collection(
            name="academic_papers", embedding_function=embedding_func
        )
    except Exception as e:
        print(f"⚠️ Could not load collection 'academic_papers'. ({e})")
        return [], []

    where_clause = {"session_id": session_id} if session_id else None

    if where_clause:
        results = collection.query(query_texts=[query], n_results=k, where=where_clause)
    else:
        results = collection.query(query_texts=[query], n_results=k)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    print(f"✅ Retrieved {len(docs)} chunks.")
    return docs, metas
