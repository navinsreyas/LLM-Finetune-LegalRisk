"""
Clause embedding model for RAG retrieval.

Uses all-MiniLM-L6-v2 to convert clause text into 384-dim vectors.
Runs on CPU to leave GPU free for Llama-3.2 inference.
"""

from sentence_transformers import SentenceTransformer
import numpy as np


class ClauseEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model ID for sentence-transformers
        """
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dimension = 384  # Output dimension of all-MiniLM-L6-v2
        print(f"[EMBED] Loaded {model_name} (dim={self.dimension})")

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single clause text.

        Args:
            text: Clause text to embed

        Returns:
            384-dimensional normalized embedding vector
        """
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Embed a batch of clause texts.

        Args:
            texts: List of clause texts to embed
            batch_size: Number of texts to process at once

        Returns:
            Array of shape (len(texts), 384) with normalized embeddings
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
