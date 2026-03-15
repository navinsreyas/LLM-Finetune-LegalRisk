"""
Build the RAG vector index.

Usage:
    python scripts/build_rag_index.py
    python scripts/build_rag_index.py --rebuild
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.build_index import build_index


def main():
    parser = argparse.ArgumentParser(description="Build RAG vector index")
    parser.add_argument(
        "--train-path", type=str, default="data/synthetic/train.jsonl",
        help="Path to training data JSONL"
    )
    parser.add_argument(
        "--persist-dir", type=str, default="data/rag/chroma_db",
        help="Directory to store ChromaDB"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Delete existing index and rebuild from scratch"
    )
    args = parser.parse_args()

    build_index(
        train_path=args.train_path,
        persist_dir=args.persist_dir,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
