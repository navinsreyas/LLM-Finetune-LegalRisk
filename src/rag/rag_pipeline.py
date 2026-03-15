"""
RAG Inference Pipeline for legal risk analysis.

Uses retrieval-augmented generation:
1. Embed the query clause
2. Retrieve top-k similar clauses with their risk assessments
3. Build a few-shot prompt with retrieved examples
4. Generate risk assessment using BASE Llama-3.2-3B (no adapter)
"""

import json
import torch
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rag.retriever import ClauseRetriever


# MUST match the system prompt from fine-tuning (data_formatter.py)
SYSTEM_PROMPT = """You are a senior legal counsel with expertise in contract risk assessment.
Analyze the provided contract clause and respond with a structured JSON risk assessment.

Output format:
{
  "clause_type": "termination|liability|non_compete|ip|governing_law",
  "risk_level": "Low|Medium|High|Critical",
  "risk_score": 0.0-1.0,
  "key_concerns": ["concern_1", "concern_2", "concern_3"],
  "recommendation": "actionable legal advice",
  "confidence": 0.0-1.0
}"""


def try_repair_json(text: str) -> dict | None:
    """
    Attempt to repair incomplete JSON by adding missing closing braces.

    Common issue: Model generation hits max_new_tokens and stops mid-JSON.
    Example: '{"risk_level": "High", "risk_score": 0.8'

    This tries to salvage it by:
    1. Finding the outermost {
    2. Counting unclosed { and [
    3. Adding the right number of } and ]

    Args:
        text: Raw model output (possibly incomplete JSON)

    Returns:
        Parsed dict if repair successful, None otherwise
    """
    start_idx = text.find("{")
    if start_idx == -1:
        return None

    # Extract JSON portion
    json_text = text[start_idx:]

    # Count unclosed braces and brackets
    open_braces = json_text.count("{") - json_text.count("}")
    open_brackets = json_text.count("[") - json_text.count("]")

    # Try to close them
    repaired = json_text + ("]" * open_brackets) + ("}" * open_braces)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        # If still fails, try removing trailing incomplete key-value
        # (e.g., '"confidence": 0.7' without comma or closing brace)
        try:
            # Find last complete key-value pair (ends with , or } or ])
            last_valid = max(
                repaired.rfind(","),
                repaired.rfind("}"),
                repaired.rfind("]")
            )
            if last_valid > 0:
                truncated = repaired[:last_valid + 1]
                # Close remaining braces
                open_braces = truncated.count("{") - truncated.count("}")
                open_brackets = truncated.count("[") - truncated.count("]")
                repaired = truncated + ("]" * open_brackets) + ("}" * open_braces)
                return json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def build_rag_prompt(query_clause: str, query_type: str, retrieved_examples: list[dict]) -> str:
    """
    Build the few-shot prompt with retrieved examples.

    This is where RAG's magic happens — instead of the model recalling
    training knowledge, we SHOW it similar examples in the prompt.

    Args:
        query_clause: The clause to analyze
        query_type: Clause type (termination, liability, etc.)
        retrieved_examples: List of similar examples from retrieval

    Returns:
        Formatted user message with examples + query
    """
    # Build the examples section
    examples_text = ""
    for i, ex in enumerate(retrieved_examples, 1):
        examples_text += f"\n--- Example {i} ---\n"
        examples_text += f"Clause ({ex['clause_type']}): {ex['clause_text'][:500]}\n"
        examples_text += f"Risk Assessment: {json.dumps(ex['output'])}\n"

    # Build the full user message
    user_message = (
        f"Here are {len(retrieved_examples)} similar clauses with their risk assessments "
        f"as reference examples:\n"
        f"{examples_text}\n"
        f"--- Your Task ---\n"
        f"Now analyze this {query_type} clause for legal risks, following the same format "
        f"as the examples above:\n\n"
        f"{query_clause}\n\n"
        f"Respond with ONLY a JSON risk assessment, no other text."
    )

    return user_message


class RAGPipeline:
    def __init__(
        self,
        model_name: str = "unsloth/Llama-3.2-3B-Instruct",
        persist_dir: str = "data/rag/chroma_db",
        n_examples: int = 5,
        filter_by_type: bool = True,
        device: str = "cuda",
    ):
        """
        Initialize the RAG pipeline.

        Args:
            model_name: Base model (no adapter — this is the whole point)
            persist_dir: ChromaDB index location
            n_examples: Number of examples to retrieve per query
            filter_by_type: If True, only retrieve same clause type examples
            device: "cuda" or "cpu"

        Why NO quantization for RAG:
        - Fine-tuned models used 4-bit quantization
        - Give RAG full float16 precision for fairest comparison
        - If RAG still loses to quantized fine-tuned models, that's compelling!
        - (If you get OOM, add load_in_4bit=True to match fine-tuned setup)
        """
        self.n_examples = n_examples
        self.filter_by_type = filter_by_type
        self.device = device

        # Load retriever (embedding model + vector store)
        print("[RAG] Loading retriever...")
        self.retriever = ClauseRetriever(persist_dir=persist_dir)
        print(f"[RAG] Index contains {self.retriever.index_size} clauses")

        # Load BASE model (no adapter!)
        print(f"[RAG] Loading base model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        print("[RAG] Pipeline ready")

    def generate(self, clause_text: str, clause_type: str, max_new_tokens: int = 1024) -> dict:
        """
        Run full RAG pipeline on a single clause.

        Args:
            clause_text: The clause to analyze
            clause_type: Clause type for filtering retrieval
            max_new_tokens: Maximum tokens to generate (increased to 1024 to avoid
                           cutting off long legal recommendations)

        Returns:
            dict with 'output' (parsed JSON), 'raw_text' (model output),
            'retrieved_examples' (what was retrieved), 'latency_ms'

        The RAG process:
        1. Retrieve similar examples from vector store
        2. Build few-shot prompt with those examples
        3. Generate risk assessment using BASE model
        4. Parse JSON output (with fallback extraction and repair)
        """
        start_time = time.time()

        filter_type = clause_type if self.filter_by_type else None
        retrieved = self.retriever.retrieve(
            clause_text=clause_text,
            n_results=self.n_examples,
            filter_by_type=filter_type,
        )

        user_message = build_rag_prompt(clause_text, clause_type, retrieved)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,       # Greedy decoding for reproducibility
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][input_length:]
        raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

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
            # Attempt 2: Extract JSON from text (handles wrapped JSON)
            start_idx = raw_text.find("{")
            end_idx = raw_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                try:
                    parsed_output = json.loads(raw_text[start_idx:end_idx])
                except (json.JSONDecodeError, ValueError):
                    pass  # Will try repair below

        if parsed_output is None:
            parsed_output = try_repair_json(raw_text)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "output": parsed_output,
            "raw_text": raw_text,
            "retrieved_examples": [
                {
                    "clause_type": ex["clause_type"],
                    "distance": ex["distance"],
                    "clause_text_preview": ex["clause_text"][:100] + "..."
                }
                for ex in retrieved
            ],
            "n_retrieved": len(retrieved),
            "input_tokens": input_length,
            "latency_ms": latency_ms,
            "json_parse_success": parsed_output is not None,
        }
