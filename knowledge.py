import hashlib
import os
import config

_collection = None   # built the first time it is needed


def _load_chunks():
    """CHUNKING: splits knowledge.txt into one chunk per '## Title' entry."""
    with open(config.KB_FILE, encoding="utf-8") as f:
        text = f.read()

    chunks, title, lines = [], None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if title and lines:
                chunks.append((title, " ".join(lines)))
            title, lines = line[3:].strip(), []
        elif line.strip():
            lines.append(line.strip())
    if title and lines:
        chunks.append((title, " ".join(lines)))
    return chunks


def _get_collection():
    """Builds (or reuses) the vector database. Rebuilds only when knowledge.txt changes."""
    global _collection
    if _collection is not None:
        return _collection

    import chromadb                                   # imported here so agent.py starts fast
    from chromadb.config import Settings

    if not os.path.exists(config.KB_FILE):
        raise FileNotFoundError(f"{config.KB_FILE} not found in the project folder.")

    with open(config.KB_FILE, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    os.makedirs(config.KB_DB_FOLDER, exist_ok=True)
    hash_path = os.path.join(config.KB_DB_FOLDER, "kb_hash.txt")
    old_hash = ""
    if os.path.exists(hash_path):
        with open(hash_path) as f:
            old_hash = f.read().strip()

    client = chromadb.PersistentClient(path=config.KB_DB_FOLDER,
                                       settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name="stock_knowledge")

    if old_hash != file_hash or collection.count() == 0:
        print("Building knowledge base (first run downloads a small embedding model, please wait)...")
        client.delete_collection("stock_knowledge")
        collection = client.get_or_create_collection(name="stock_knowledge")

        chunks = _load_chunks()
        collection.add(                                # EMBEDDINGS are created automatically here
            ids=[f"kb-{i}" for i in range(len(chunks))],
            documents=[f"{title}. {body}" for title, body in chunks],
            metadatas=[{"title": title} for title, _ in chunks],
        )
        with open(hash_path, "w") as f:
            f.write(file_hash)
        print(f"Knowledge base ready: {len(chunks)} entries indexed.")

    _collection = collection
    return collection


def _raw_search(question, n_results):
    """RETRIEVAL: finds the closest entries. Smaller distance = closer meaning."""
    col = _get_collection()
    result = col.query(query_texts=[question], n_results=min(n_results, col.count()))
    return [
        {"title": meta["title"], "text": doc, "distance": round(dist, 3)}
        for doc, meta, dist in zip(result["documents"][0],
                                   result["metadatas"][0],
                                   result["distances"][0])
    ]


def search_knowledge(question):
    """TOOL: returns the knowledge-base entries relevant to a question."""
    question = str(question).strip()[:300]
    if not question:
        return {"error": "Empty question."}
    try:
        found = _raw_search(question, config.KB_RESULTS)
    except Exception as e:
        return {"error": f"Knowledge base failed: {e}"}

    relevant = [{"title": r["title"], "text": r["text"]}
                for r in found if r["distance"] <= config.KB_MAX_DISTANCE]
    if not relevant:
        return {"results": [], "note": "Nothing relevant was found in the knowledge base."}
    return {"results": relevant}


# ---------- Quick manual test: shows distances so you can tune KB_MAX_DISTANCE ----------
if __name__ == "__main__":
    tests = [
        "what is a P/E ratio",
        "why can't I sell my stock today, it just stopped moving",
        "how do I apply for a new share offering",
        "is a stock priced at 5 rupees cheap",
        "how do I cook biryani",
    ]
    for q in tests:
        print(f"\nQ: {q}")
        for r in _raw_search(q, 3):
            mark = "KEEP" if r["distance"] <= config.KB_MAX_DISTANCE else "drop"
            print(f"   {r['distance']:<6} {mark}  {r['title']}")