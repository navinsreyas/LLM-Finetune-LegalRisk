"""
Retrieval pipeline: given a new clause, find the most similar training examples.
"""

from src.rag.embedder import ClauseEmbedder
from src.rag.vector_store import ClauseVectorStore


class ClauseRetriever:
    def __init__(
        self,
        persist_dir: str = "data/rag/chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the retrieval pipeline.

        Args:
            persist_dir: ChromaDB storage location
            embedding_model: Sentence-transformers model to use

        This combines:
        - ClauseEmbedder: Converts text → vectors
        - ClauseVectorStore: Stores and searches vectors
        """
        self.embedder = ClauseEmbedder(model_name=embedding_model)
        self.store = ClauseVectorStore(persist_dir=persist_dir)

    def retrieve(
        self,
        clause_text: str,
        n_results: int = 5,
        filter_by_type: str = None
    ) -> list[dict]:
        """
        Retrieve the most similar training examples for a given clause.

        Args:
            clause_text: The new clause to find similar examples for
            n_results: Number of examples to retrieve (default 5)
            filter_by_type: Optional clause type filter

        Returns:
            List of similar examples with clause_text, clause_type, output, distance

        The magic of RAG:
        1. Embed the query clause (converts to 384-dim vector)
        2. Search ChromaDB for nearest neighbors (cosine similarity)
        3. Return the training examples that are most similar
        4. These become few-shot examples in the prompt
        """
        # Step 1: Embed the query clause
        query_embedding = self.embedder.embed_single(clause_text)

        # Step 2: Search the vector store
        results = self.store.query(
            query_embedding=query_embedding,
            n_results=n_results,
            clause_type=filter_by_type,
        )

        return results

    @property
    def index_size(self) -> int:
        """Number of clauses in the index."""
        return self.store.collection.count()
