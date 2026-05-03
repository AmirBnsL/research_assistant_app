import os
import chromadb
from chromadb.utils import embedding_functions


def test_semantic_search(query_text: str, n_results: int = 2):
    """
    Connects to the local ChromaDB, searches for the most relevant chunks
    based on the user's query, and prints the results.
    """
    print(f"\n🔎 Searching database for: '{query_text}'")

    # 1. Dynamically find the database path (same logic as ingestion)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    db_path = os.path.join(project_root, "data", "chromadb_store")

    # 2. Connect to the database
    try:
        client = chromadb.PersistentClient(path=db_path)
    except Exception as e:
        print(f"❌ Could not connect to database at {db_path}. Error: {e}")
        return

    # 3. Load the EXACT SAME embedding model
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # 4. Fetch the collection
    # We use get_collection instead of get_or_create so it fails if the DB is empty
    try:
        collection = client.get_collection(
            name="user_profile", embedding_function=embedding_func
        )
    except ValueError:
        print("❌ Collection 'user_profile' does not exist. Run embedding.py first!")
        return

    # 5. Perform the Query!
    results = collection.query(query_texts=[query_text], n_results=n_results)

    # 6. Display the Results neatly
    print("\n" + "=" * 50)
    print("🏆 TOP SEARCH RESULTS")
    print("=" * 50)

    # Chroma returns lists inside of lists, so we grab the first item [0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(documents)):
        print(f"\nResult #{i+1} (Distance Score: {distances[i]:.4f})")
        print(
            f"📄 Source: {metadatas[i]['source']} | Chunk: {metadatas[i]['chunk_index']}"
        )
        print(f"📝 Text: {documents[i]}")
        print("-" * 50)


# ==========================================
# TEST RUNNER
# ==========================================
if __name__ == "__main__":
    # Test 1: A direct question related to our mock paper
    test_semantic_search("Why did you use the MiniLM model?")

    # Test 2: A conceptual question that shares few exact keywords
    test_semantic_search("What is RAG used for?")
