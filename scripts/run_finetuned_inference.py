"""
Run fine-tuned model inference on the test set.

Usage:
    python scripts/run_finetuned_inference.py --method qlora
    python scripts/run_finetuned_inference.py --method dora
    python scripts/run_finetuned_inference.py --method ia3
    python scripts/run_finetuned_inference.py --method qlora --dry-run
"""

import argparse
import json
import sys
import time
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.data_formatter import SYSTEM_PROMPT


# Method → adapter path mapping
ADAPTER_PATHS = {
    "qlora": "models/qlora",
    "dora": "models/dora",
    "ia3": "models/ia3",
}

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"


def load_finetuned_model(method: str, device: str = "cuda"):
    """
    Load base model + PEFT adapter.

    Args:
        method: One of "qlora", "dora", "ia3"
        device: "cuda" or "cpu"

    Returns:
        (model, tokenizer) tuple

    Why different loading for QLoRA/DoRA vs IA³:
    - QLoRA/DoRA: Use 4-bit quantization (matching training setup)
    - IA³: No quantization (matching training setup)
    """
    adapter_path = ADAPTER_PATHS[method]
    print(f"[MODEL] Loading base model: {BASE_MODEL}")

    # Load with 4-bit quantization (matching training setup for QLoRA/DoRA)
    if method in ("qlora", "dora"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:  # ia3 — no quantization
        print(f"[MODEL] Using full fp16 (no quantization)")

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    # Load adapter
    print(f"[MODEL] Loading {method} adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[MODEL] {method.upper()} model loaded successfully")
    return model, tokenizer


def generate_prediction(model, tokenizer, clause_text: str, clause_type: str,
                        max_new_tokens: int = 512, device: str = "cuda") -> dict:
    """
    Generate a risk assessment for a single clause.

    Args:
        model: Fine-tuned model with adapter
        tokenizer: Tokenizer
        clause_text: The clause to analyze
        clause_type: Clause type
        max_new_tokens: Maximum tokens to generate
        device: "cuda" or "cpu"

    Returns:
        dict with output, raw_text, json_parse_success, latency_ms
    """
    start_time = time.time()

    user_message = (
        f"Analyze this {clause_type} clause for legal risks:\n\n"
        f"{clause_text}\n\n"
        f"Respond with a JSON risk assessment."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = outputs[0][input_length:]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Parse JSON
    parsed_output = None
    try:
        clean_text = raw_text
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:]
            clean_text = clean_text.strip()
        parsed_output = json.loads(clean_text)
    except json.JSONDecodeError:
        try:
            start_idx = raw_text.find("{")
            end_idx = raw_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                parsed_output = json.loads(raw_text[start_idx:end_idx])
        except (json.JSONDecodeError, ValueError):
            parsed_output = None

    latency_ms = (time.time() - start_time) * 1000

    return {
        "output": parsed_output,
        "raw_text": raw_text,
        "json_parse_success": parsed_output is not None,
        "input_tokens": input_length,
        "latency_ms": latency_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Run fine-tuned model inference")
    parser.add_argument("--method", type=str, required=True, choices=["qlora", "dora", "ia3"])
    parser.add_argument("--test-path", type=str, default="data/synthetic/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Process only 5 examples")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.method}_predictions.jsonl"

    # Load test data
    test_examples = []
    with open(args.test_path, "r", encoding="utf-8") as f:
        for line in f:
            test_examples.append(json.loads(line.strip()))

    if args.dry_run:
        test_examples = test_examples[:5]
        print(f"[DRY RUN] Processing only 5 test examples")

    print(f"[EVAL] Loaded {len(test_examples)} test examples")

    # Load model
    model, tokenizer = load_finetuned_model(args.method)

    # Run inference
    results = []
    json_success_count = 0
    total_latency = 0

    print(f"\n[EVAL] Running {args.method.upper()} inference...")

    for i, example in enumerate(tqdm(test_examples, desc=f"{args.method} inference")):
        clause_text = example["input"]["clause_text"]
        clause_type = example["input"]["clause_type"]
        ground_truth = example["output"]

        pred = generate_prediction(model, tokenizer, clause_text, clause_type)

        result = {
            "test_index": i,
            "input": example["input"],
            "ground_truth": ground_truth,
            "prediction": pred["output"],
            "raw_prediction": pred["raw_text"],
            "json_parse_success": pred["json_parse_success"],
            "input_tokens": pred["input_tokens"],
            "latency_ms": pred["latency_ms"],
            "method": args.method,
        }
        results.append(result)

        if pred["json_parse_success"]:
            json_success_count += 1
        total_latency += pred["latency_ms"]

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Summary
    json_rate = json_success_count / len(test_examples) * 100
    avg_latency = total_latency / len(test_examples)

    print(f"\n{'='*70}")
    print(f"  {args.method.upper()} Inference Complete")
    print(f"{'='*70}")
    print(f"  Test examples:        {len(test_examples)}")
    print(f"  JSON parse success:   {json_success_count}/{len(test_examples)} ({json_rate:.1f}%)")
    print(f"  Average latency:      {avg_latency:.0f} ms")
    print(f"  Results saved to:     {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
