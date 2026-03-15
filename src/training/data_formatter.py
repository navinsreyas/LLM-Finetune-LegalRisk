"""
Data Formatter for LegalRisk-LLM Training Pipeline.

Converts Phase 1C JSONL (input/output/metadata structure) into Llama-3.2-Instruct
chat format that SFTTrainer expects.

CRITICAL: The system prompt defined here MUST be identical during training and inference.
If it changes, model behavior becomes unpredictable.
"""

import json
from pathlib import Path
from datasets import Dataset
from typing import Dict


# SYSTEM PROMPT - NEVER MODIFY AFTER TRAINING STARTS
# This establishes the model's role and output format. Changing this invalidates trained weights.
SYSTEM_PROMPT = """You are a senior legal counsel with expertise in contract risk assessment.
Analyze the provided contract clause and respond with a structured JSON risk assessment.

Output format:
{
  "clause_type": "termination|liability|non_compete|ip|governing_law|confidentiality|indemnification",
  "risk_level": "Low|Medium|High|Critical",
  "risk_score": 0.0-1.0,
  "key_concerns": ["concern_1", "concern_2", "concern_3"],
  "recommendation": "actionable legal advice",
  "confidence": 0.0-1.0
}"""


def format_single_example(example: Dict) -> Dict:
    """
    Convert one Phase 1C training example to Llama-3.2 chat format.

    Input format (from Phase 1C):
        {
            "input": {
                "clause_text": "...",
                "clause_type": "termination",
                "context_window": "...",
                "source": "cuad",
                "source_doc": "..."
            },
            "output": {
                "clause_type": "termination",
                "risk_level": "medium",
                "risk_score": 0.42,
                "key_concerns": ["...", "...", "..."],
                "recommendation": "...",
                "confidence": 0.85
            },
            "metadata": {...}
        }

    Output format (for SFTTrainer):
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }

    Why this format? SFTTrainer's apply_chat_template() expects exactly this structure.
    It will automatically add Llama-3.2's special tokens:
    - <|begin_of_text|>
    - <|start_header_id|>system<|end_header_id|>
    - <|eot_id|>
    Never manually add these tokens - let the tokenizer handle it.

    Args:
        example: Dict from Phase 1C JSONL with input/output/metadata

    Returns:
        Dict with "messages" key containing system/user/assistant turns
    """
    inp = example["input"]
    out = example["output"]

    clause_type = inp.get("clause_type", "unknown")
    clause_text = inp.get("clause_text", "")

    user_message = f"""Analyze this {clause_type} clause for legal risks:

CLAUSE:
{clause_text}"""

    context_window = inp.get("context_window", "")
    if context_window and context_window.strip():
        user_message += f"\n\nCONTEXT:\n{context_window}"

    # Build assistant message (the JSON output the model should learn to produce)
    # Use compact JSON (no indentation) to save tokens during training
    # Why compact? With 860 examples × 3 epochs × ~250 output tokens/example,
    # indentation would waste ~100,000 tokens = ~$0.30 in compute. Small but unnecessary.
    assistant_message = json.dumps(out, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def load_and_format_dataset(
    jsonl_path: str | Path,
    limit: int | None = None
) -> Dataset:
    """
    Load a Phase 1C JSONL file and convert all examples to chat format.

    Args:
        jsonl_path: Path to train.jsonl, val.jsonl, or test.jsonl
        limit: Optional limit on number of examples (for sanity check mode)

    Returns:
        HuggingFace Dataset ready for SFTTrainer

    Example:
        >>> train_dataset = load_and_format_dataset("data/synthetic/train.jsonl")
        >>> print(len(train_dataset))  # 860
        >>> print(train_dataset[0]["messages"][0]["role"])  # "system"
    """
    jsonl_path = Path(jsonl_path)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Training data not found: {jsonl_path}")

    examples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            raw = json.loads(line.strip())
            formatted = format_single_example(raw)
            examples.append(formatted)

    dataset = Dataset.from_list(examples)

    print(f"[DATA] Loaded {len(dataset)} examples from {jsonl_path.name}")

    return dataset


def get_system_prompt() -> str:
    """
    Return the system prompt used during training.

    Why a getter function? During inference (Phase 4B), we need the EXACT same
    system prompt. Importing from this module ensures consistency.

    Returns:
        System prompt string
    """
    return SYSTEM_PROMPT


if __name__ == "__main__":
    # Quick test: Load and format 3 examples from train set
    print("="*70)
    print("Data Formatter Test")
    print("="*70)

    train_path = Path("data/synthetic/train.jsonl")

    if train_path.exists():
        dataset = load_and_format_dataset(train_path, limit=3)

        print(f"\n[TEST] Loaded {len(dataset)} examples")
        print(f"\n[TEST] Example 0 structure:")
        print(f"  - messages length: {len(dataset[0]['messages'])}")
        print(f"  - roles: {[msg['role'] for msg in dataset[0]['messages']]}")

        print(f"\n[TEST] System message (first 200 chars):")
        print(f"  {dataset[0]['messages'][0]['content'][:200]}...")

        print(f"\n[TEST] User message (first 300 chars):")
        print(f"  {dataset[0]['messages'][1]['content'][:300]}...")

        print(f"\n[TEST] Assistant message:")
        assistant_msg = dataset[0]['messages'][2]['content']
        try:
            parsed = json.loads(assistant_msg)
            print(f"  Valid JSON: {json.dumps(parsed, indent=2)[:400]}...")
        except json.JSONDecodeError:
            print(f"  ERROR: Invalid JSON")
            print(f"  Raw: {assistant_msg[:200]}...")

        print(f"\n{'='*70}")
        print(f"[SUCCESS] Data formatter working correctly!")
        print(f"{'='*70}\n")
    else:
        print(f"[ERROR] Training data not found at {train_path}")
        print(f"[ERROR] Run Phase 1C first to generate training data")
