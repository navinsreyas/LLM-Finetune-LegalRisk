"""
RAG Inference Pipeline for legal risk analysis.

Uses retrieval-augmented generation:
1. Embed the query clause
2. Retrieve top-k similar clauses with their risk assessments
3. Build a few-shot prompt with retrieved examples
4. Generate risk assessment using BASE Llama-3.2-3B (no adapter)
"""

import json
import os
import time
from pathlib import Path

from prometheus_client import Histogram

from src.rag.retriever import ClauseRetriever

# NOTE: torch / transformers are imported lazily inside the "local" provider path
# (RAGPipeline.__init__ and _generate_local) so a groq-only deployment never imports
# them or loads any local weights.

GROQ_CALL_DURATION = Histogram(
    "groq_call_duration_seconds",
    "Seconds spent inside the Groq chat.completions.create call only.",
)

# How many clauses were actually retrieved per query. Target is n_examples (usually 5),
# but type-filtering can return fewer. Small integer buckets fit the expected range.
RETRIEVAL_CHUNKS = Histogram(
    "retrieval_chunks_count",
    "Number of clauses retrieved per query (fewer than n_examples if filtering finds less).",
    buckets=(0, 1, 2, 3, 4, 5, 10),
)


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
        """
        self.n_examples = n_examples
        self.filter_by_type = filter_by_type
        self.device = device

        self.provider = os.environ.get("LLM_PROVIDER", "local").lower()

        self.model = None
        self.tokenizer = None
        self.groq_client = None
        self.groq_model = None

        # Load retriever (embedding model + vector store) -- shared by both providers
        print("[RAG] Loading retriever...")
        self.retriever = ClauseRetriever(persist_dir=persist_dir)
        print(f"[RAG] Index contains {self.retriever.index_size} clauses")

        if self.provider == "groq":
            # Remote generation: skip the heavy local model/tokenizer load entirely
            # (saves GPU/RAM on a cheap server).
            from groq import Groq

            self.groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
            self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            print(f"[RAG] Provider=groq, model={self.groq_model} (local model NOT loaded)")
        else:
            # Local generation (default): existing behavior, unchanged.
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

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
        """
        start_time = time.time()

        filter_type = clause_type if self.filter_by_type else None
        retrieved = self.retriever.retrieve(
            clause_text=clause_text,
            n_results=self.n_examples,
            filter_by_type=filter_type,
        )
        # Metric: how many clauses this query actually got (may be < n_examples).
        RETRIEVAL_CHUNKS.observe(len(retrieved))

        user_message = build_rag_prompt(clause_text, clause_type, retrieved)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # Generation step -- the ONLY provider-dependent part. Both helpers return
        # (raw_text, prompt_token_count) so everything below is identical.
        if self.provider == "groq":
            raw_text, input_length = self._generate_groq(messages, max_new_tokens)
        else:
            raw_text, input_length = self._generate_local(messages, max_new_tokens)

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

    def _generate_local(self, messages: list[dict], max_new_tokens: int) -> tuple[str, int]:
        """
        Local generation path (default). Unchanged from the original behavior:
        greedy decode with the on-device base model.

        Returns (raw_text, prompt_token_count).
        """
        import torch

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
        return raw_text, input_length

    def _generate_groq(self, messages: list[dict], max_new_tokens: int) -> tuple[str, int]:
        """
        Remote generation via Groq's API, using the SAME (system + user) prompt the
        local path builds. No local model or tokenizer is touched.
        """
        # Metric: time ONLY the Groq API call (this method runs only on the groq path,
        # so the local generation path never records this).
        with GROQ_CALL_DURATION.time():
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
        raw_text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        input_length = getattr(usage, "prompt_tokens", 0) or 0
        return raw_text, input_length
