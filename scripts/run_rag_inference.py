"""
Run RAG inference on the test set and save results.

Usage:
    python scripts/run_rag_inference.py
    python scripts/run_rag_inference.py --n-examples 3   # Retrieve 3 instead of 5
    python scripts/run_rag_inference.py --no-filter       # Don't filter by clause type
    python scripts/run_rag_inference.py --dry-run         # Process only 5 test examples
"""

import argparse
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Run RAG inference on test set")
    parser.add_argument("--test-path", type=str, default="data/synthetic/test.jsonl")
    parser.add_argument("--output-path", type=str, default="evaluation/rag_predictions.jsonl")
    parser.add_argument("--persist-dir", type=str, default="data/rag/chroma_db")
    parser.add_argument("--n-examples", type=int, default=5, help="Number of examples to retrieve")
    parser.add_argument("--no-filter", action="store_true", help="Don't filter retrieval by clause type")
    parser.add_argument("--dry-run", action="store_true", help="Process only 5 test examples")
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load test data
    test_examples = []
    with open(args.test_path, "r", encoding="utf-8") as f:
        for line in f:
            test_examples.append(json.loads(line.strip()))

    if args.dry_run:
        test_examples = test_examples[:5]
        print(f"[DRY RUN] Processing only 5 test examples")

    print(f"[RAG] Loaded {len(test_examples)} test examples")

    # Initialize RAG pipeline
    pipeline = RAGPipeline(
        persist_dir=args.persist_dir,
        n_examples=args.n_examples,
        filter_by_type=not args.no_filter,
    )

    # Run inference
    results = []
    json_success_count = 0
    total_latency = 0

    print(f"\n[RAG] Running inference on {len(test_examples)} test examples...")
    print(f"[RAG] Retrieving {args.n_examples} examples per query")
    print(f"[RAG] Filter by clause type: {not args.no_filter}")

    for i, example in enumerate(tqdm(test_examples, desc="RAG inference")):
        clause_text = example["input"]["clause_text"]
        clause_type = example["input"]["clause_type"]
        ground_truth = example["output"]

        # Run RAG
        rag_result = pipeline.generate(clause_text, clause_type)

        # Package result
        result = {
            "test_index": i,
            "input": example["input"],
            "ground_truth": ground_truth,
            "prediction": rag_result["output"],  # Parsed JSON (or None if failed)
            "raw_prediction": rag_result["raw_text"],
            "json_parse_success": rag_result["json_parse_success"],
            "retrieved_examples": rag_result["retrieved_examples"],
            "n_retrieved": rag_result["n_retrieved"],
            "input_tokens": rag_result["input_tokens"],
            "latency_ms": rag_result["latency_ms"],
            "method": "rag",
        }
        results.append(result)

        if rag_result["json_parse_success"]:
            json_success_count += 1
        total_latency += rag_result["latency_ms"]

        # Progress update every 10 examples
        if (i + 1) % 10 == 0:
            success_rate = json_success_count / (i + 1) * 100
            avg_latency = total_latency / (i + 1)
            print(f"  [{i+1}/{len(test_examples)}] JSON success: {success_rate:.1f}%, "
                  f"Avg latency: {avg_latency:.0f}ms")

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Print summary
    json_rate = json_success_count / len(test_examples) * 100
    avg_latency = total_latency / len(test_examples)

    # Category breakdown
    category_stats = {}
    for r in results:
        ct = r["input"]["clause_type"]
        if ct not in category_stats:
            category_stats[ct] = {"total": 0, "json_success": 0}
        category_stats[ct]["total"] += 1
        if r["json_parse_success"]:
            category_stats[ct]["json_success"] += 1

    print(f"\n{'='*70}")
    print(f"  RAG Inference Complete")
    print(f"{'='*70}")
    print(f"  Test examples:        {len(test_examples)}")
    print(f"  JSON parse success:   {json_success_count}/{len(test_examples)} ({json_rate:.1f}%)")
    print(f"  Average latency:      {avg_latency:.0f} ms")
    print(f"  Total time:           {total_latency/1000:.1f} seconds")
    print(f"  Results saved to:     {output_path}")
    print(f"\n  Category Breakdown:")
    for cat, stats in sorted(category_stats.items()):
        rate = stats["json_success"] / stats["total"] * 100
        print(f"    {cat:20s}  {stats['json_success']}/{stats['total']} ({rate:.0f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
