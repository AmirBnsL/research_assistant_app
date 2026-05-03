from sentence_transformers import CrossEncoder
from src.rag.retrieve import retrieve_top_k


def rerank_results(
    query: str,
    retrieved_chunks: list[str],
    retrieved_metadatas: list[dict],
    top_n: int = 5,
):
    if not retrieved_chunks:
        print("⚠️ No chunks effectively retrieved. Returning empty list.")
        return []

    print(
        f"⚖️ Reranking {len(retrieved_chunks)} candidate chunks using CrossEncoder..."
    )
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [[query, chunk] for chunk in retrieved_chunks]
    scores = model.predict(pairs)

    scored_items = sorted(
        zip(scores, retrieved_chunks, retrieved_metadatas),
        key=lambda x: x[0],
        reverse=True,
    )
    top_items = scored_items[:top_n]
    print(f"✅ Finished reranking and selected top {len(top_items)} chunks.")
    return top_items


def search_local_memory(
    query: str, session_id: str | None = None
) -> tuple[str, list[dict]]:
    print(f"\n[Local Memory Search -> '{query}' for session '{session_id}']")
    candidates, metadatas = retrieve_top_k(query=query, k=20, session_id=session_id)

    top_items = rerank_results(
        query=query, retrieved_chunks=candidates, retrieved_metadatas=metadatas, top_n=3
    )

    if not top_items:
        return "No relevant local literature found.", []

    context_chunks = []
    evidence_list = []

    for score, chunk, meta in top_items:
        context_chunks.append(f"Local Paper Fragment:\n{chunk}")
        evidence_list.append(
            {
                "source": meta.get("source", "Local Database"),
                "title": meta.get("source", "Unknown Document"),
                "authors": "Local Library",
                "score": round(float(score), 4),
            }
        )

    context = "\n\n---\n\n".join(context_chunks)
    print(f"[Finished Search -> Returned {len(top_items)} reranked chunks]\n")
    return context, evidence_list
