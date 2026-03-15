"""
Build the RAG vector index from training data.

One-time operation — run once, then the index persists on disk.
Takes ~30 seconds for 855 clauses.
"""

import json
from pathlib import Path

from src.rag.embedder import ClauseEmbedder
from src.rag.vector_store import ClauseVectorStore


def build_index(
    train_path: str = "data/synthetic/train.jsonl",
    persist_dir: str = "data/rag/chroma_db",
    rebuild: bool = False,
):
    """
    Build the vector index from training examples.

    Args:
        train_path: Path to training JSONL file
        persist_dir: Where to store ChromaDB
        rebuild: If True, delete existing index and rebuild from scratch

    What this does:
    1. Load all training examples (855 clauses)
    2. Embed each clause text using sentence-transformers
    3. Store embeddings + metadata in ChromaDB
    4. Index persists to disk — never need to rebuild unless data changes
    """
    print("=" * 70)
    print("  Building RAG Vector Index")
    print("=" * 70)

    # Load training data
    examples = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    print(f"[INDEX] Loaded {len(examples)} training examples from {train_path}")

    # Initialize components
    embedder = ClauseEmbedder()
    store = ClauseVectorStore(persist_dir=persist_dir)

    # Check if index already exists
    if store.collection.count() > 0 and not rebuild:
        print(f"[INDEX] Index already has {store.collection.count()} documents.")
        print(f"[INDEX] Use --rebuild flag to rebuild from scratch.")
        return

    if rebuild:
        print("[INDEX] Rebuilding index from scratch...")
        store.clear()

    # Extract clause texts for embedding
    clause_texts = [ex["input"]["clause_text"] for ex in examples]

    # Embed all clauses
    print(f"[INDEX] Embedding {len(clause_texts)} clauses...")
    embeddings = embedder.embed_batch(clause_texts)
    print(f"[INDEX] Embedding shape: {embeddings.shape}")

    # Store in ChromaDB
    print("[INDEX] Storing in ChromaDB...")
    store.add_examples(examples, embeddings)

    # Verify
    print(f"\n[INDEX] Index built successfully!")
    print(f"[INDEX] Total documents: {store.collection.count()}")

    # Print category distribution
    category_counts = {}
    for ex in examples:
        ct = ex["input"]["clause_type"]
        category_counts[ct] = category_counts.get(ct, 0) + 1

    print(f"\n[INDEX] Category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s} {count:4d}")

    # Quick retrieval test
    print(f"\n[INDEX] Running quick retrieval test...")
    test_clause = clause_texts[0]
    test_type = examples[0]["input"]["clause_type"]
    query_embedding = embedder.embed_single(test_clause)
    results = store.query(query_embedding, n_results=3, clause_type=test_type)

    print(f"[INDEX] Query: '{test_clause[:80]}...'")
    print(f"[INDEX] Top 3 results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['clause_type']}] dist={r['distance']:.4f} '{r['clause_text'][:60]}...'")

    print(f"\n{'='*70}")
    print(f"  Index Ready! ({store.collection.count()} documents)")
    print(f"{'='*70}")
