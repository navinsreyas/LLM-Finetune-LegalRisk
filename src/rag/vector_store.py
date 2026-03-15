"""
ChromaDB vector store for legal clause retrieval.

Stores embedded training examples and retrieves top-k similar clauses
for few-shot prompting.
"""

import chromadb
from chromadb.config import Settings
import json
from pathlib import Path


class ClauseVectorStore:
    def __init__(self, persist_dir: str = "data/rag/chroma_db"):
        """
        Initialize ChromaDB with persistent storage.

        Args:
            persist_dir: Directory to store the vector database on disk.
                         Survives restarts — no need to re-embed every time.

        Why persistent storage:
        - Once index is built, it's saved to disk
        - Subsequent runs load instantly (no re-embedding)
        - 855 clauses embed in ~30 seconds, but only need to do it once
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get the collection
        # hnsw:space = cosine means we use cosine similarity for matching
        self.collection = self.client.get_or_create_collection(
            name="legal_clauses",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        print(f"[STORE] ChromaDB initialized at {self.persist_dir}")
        print(f"[STORE] Collection 'legal_clauses' has {self.collection.count()} documents")

    def add_examples(self, examples: list[dict], embeddings: list):
        """
        Add training examples to the vector store.

        Args:
            examples: List of training examples (each has 'input' and 'output')
            embeddings: Corresponding embedding vectors from ClauseEmbedder

        What gets stored:
        - Document: The clause text (what we search against)
        - Metadata: Clause type + complete risk assessment JSON
        - Embedding: 384-dim vector for similarity search
        """
        ids = []
        documents = []
        metadatas = []
        embedding_list = []

        for i, (example, embedding) in enumerate(zip(examples, embeddings)):
            # Use the clause text as the document
            clause_text = example["input"]["clause_text"]
            clause_type = example["input"]["clause_type"]

            # Store the complete output (risk assessment) as metadata
            # ChromaDB metadata values must be strings, ints, floats, or bools
            metadata = {
                "clause_type": clause_type,
                "output_json": json.dumps(example["output"]),  # Store as JSON string
                "source": example["input"].get("source", "unknown"),
                "index": i,
            }

            ids.append(f"clause_{i}")
            documents.append(clause_text)
            metadatas.append(metadata)
            embedding_list.append(embedding.tolist())

        # Add in batches (ChromaDB has a limit per call)
        batch_size = 500
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embedding_list[start:end],
            )

        print(f"[STORE] Added {len(ids)} examples. Total: {self.collection.count()}")

    def query(self, query_embedding, n_results: int = 5, clause_type: str = None) -> list[dict]:
        """
        Find the most similar clauses to a query.

        Args:
            query_embedding: Vector from ClauseEmbedder.embed_single()
            n_results: Number of similar examples to retrieve
            clause_type: Optional filter — only retrieve same clause type

        Returns:
            List of dicts with 'clause_text', 'clause_type', 'output', 'distance'

        Why filter by clause_type:
        - Termination clauses should see termination examples
        - IP clauses should see IP examples
        - Gives RAG its best shot at relevant few-shot examples
        """
        where_filter = None
        if clause_type:
            where_filter = {"clause_type": clause_type}

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved.append({
                "clause_text": doc,
                "clause_type": meta["clause_type"],
                "output": json.loads(meta["output_json"]),
                "distance": dist,  # Lower = more similar (cosine distance)
            })

        return retrieved

    def clear(self):
        """Delete all documents from the collection (for rebuilding)."""
        self.client.delete_collection("legal_clauses")
        self.collection = self.client.get_or_create_collection(
            name="legal_clauses",
            metadata={"hnsw:space": "cosine"}
        )
        print("[STORE] Collection cleared")
