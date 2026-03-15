"""
Synthetic Data Generator for LegalRisk-LLM Pipeline.

Uses Claude Sonnet API to generate structured risk assessments for contract clauses.
This creates the "answer key" that pairs with CUAD clauses (the "exam questions").

ARCHITECTURAL DECISION: Why synthetic data generation?
We have labeled clauses from CUAD, but no risk assessments. Instead of manually
labeling 1,200+ examples (50+ hours of senior counsel time at $400/hr = $20,000+),
we use Claude Sonnet to generate high-quality synthetic labels for ~$5.

The key insight: Claude Sonnet is a strong legal reasoner. While we wouldn't deploy
it directly (too expensive at inference), we can use it as a "teacher model" to
create training data for a smaller, fine-tuned model.
"""

import anthropic
import json
import time
import random
from pathlib import Path
from datetime import datetime
import logging


# PROMPT TEMPLATES
# Why these specific prompts? They're engineered to:
# 1. Force JSON-only output (no markdown wrapper that breaks parsing)
# 2. Align risk_score with risk_level (consistency check later)
# 3. Require EXACTLY 3 key_concerns (prevents vague "see above" responses)
# 4. Demand actionable recommendations (not "consult a lawyer" generic advice)

SYSTEM_PROMPT = """You are a senior legal counsel with 20 years of experience in commercial contract review.
You specialize in identifying and assessing legal risks in contract clauses.

Your task is to analyze a contract clause and produce a structured risk assessment.

RULES:
1. Respond with ONLY valid JSON — no markdown, no backticks, no explanation before or after
2. The JSON must exactly match the schema provided
3. key_concerns must contain EXACTLY 3 items — the 3 most important risks
4. risk_score must align with risk_level: low (0.0-0.25), medium (0.26-0.50), high (0.51-0.75), critical (0.76-1.0)
5. recommendation must be specific and actionable — not vague like "review the clause"
6. confidence reflects how certain YOU are in your assessment (0.7+ for clear clauses, lower for ambiguous ones)
"""

USER_PROMPT_TEMPLATE = """Analyze this {clause_type} clause for legal risks:

CLAUSE:
{clause_text}

CONTEXT (surrounding text from the contract):
{context_window}

Respond with ONLY this JSON (no other text):
{{
    "clause_type": "{clause_type}",
    "risk_level": "low|medium|high|critical",
    "risk_score": 0.0-1.0,
    "key_concerns": ["concern_1", "concern_2", "concern_3"],
    "recommendation": "specific actionable recommendation",
    "confidence": 0.0-1.0
}}"""

SYNTHETIC_CLAUSE_SYSTEM_PROMPT = """You are a senior legal counsel who drafts and reviews commercial contracts.

Your task is to:
1. Generate a realistic contract clause of the specified type
2. Then analyze it for legal risks

RULES:
1. The clause must read like it came from a real commercial contract (use formal legal language)
2. Vary the complexity — some clauses should be simple, others complex with multiple conditions
3. Vary the risk level — some should be low risk, others high or critical
4. Respond with ONLY valid JSON — no markdown, no backticks, no explanation
"""

SYNTHETIC_USER_PROMPT = """Generate a realistic {clause_type} clause for a commercial contract and analyze its risks.

Difficulty level: {difficulty}
Risk tendency: {risk_tendency}

Respond with ONLY this JSON:
{{
    "generated_clause": "the full clause text you generated",
    "clause_type": "{clause_type}",
    "risk_level": "low|medium|high|critical",
    "risk_score": 0.0-1.0,
    "key_concerns": ["concern_1", "concern_2", "concern_3"],
    "recommendation": "specific actionable recommendation",
    "confidence": 0.0-1.0
}}"""


class SyntheticGenerator:
    """
    Manages synthetic risk assessment generation via Claude API.

    This class handles:
    - API calls with retry logic
    - Progress tracking and resume capability
    - Cost estimation
    - Two generation modes: CUAD-based and fully synthetic
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        output_dir: Path | str = "data/synthetic",
        logger: logging.Logger | None = None
    ):
        """
        Initialize the synthetic generator.

        Args:
            api_key: Anthropic API key
            model: Claude model to use (default: Sonnet 4)
            output_dir: Directory for output files
            logger: Optional logger instance
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or logging.getLogger(__name__)

        self.stats = {
            "total_attempted": 0,
            "successful": 0,
            "failed_api": 0,
            "failed_parse": 0,
            "retries": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

    def generate_single(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 2
    ) -> dict | None:
        """
        Send one clause to Claude and get back a risk assessment JSON.

        Why max_retries=2? API errors happen (rate limits, network issues).
        Two retries = 3 total attempts balances reliability vs latency.
        If it fails 3 times, the example is genuinely problematic.

        Args:
            system_prompt: System message for Claude
            user_prompt: User message with clause and instructions
            max_retries: Number of retry attempts on failure

        Returns:
            Parsed JSON dict, or None if all attempts failed
        """
        self.stats["total_attempted"] += 1

        for attempt in range(max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )

                self.stats["total_input_tokens"] += response.usage.input_tokens
                self.stats["total_output_tokens"] += response.usage.output_tokens

                text = response.content[0].text.strip()

                # Handle markdown code blocks (some models add ``` despite instructions)
                # This is defensive parsing - we tell Claude not to do this, but it sometimes does
                if text.startswith("```"):
                    # Remove opening ```json or ```
                    text = text.split("```", 1)[1]
                    if text.startswith("json"):
                        text = text[4:]
                    # Remove closing ```
                    if "```" in text:
                        text = text.split("```")[0]
                    text = text.strip()

                result = json.loads(text)
                self.stats["successful"] += 1
                return result

            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON parse failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                self.stats["failed_parse"] += 1

                if attempt < max_retries:
                    self.stats["retries"] += 1
                    time.sleep(1)  # Brief pause before retry
                    continue

                self.logger.error(f"All attempts failed for JSON parsing. Raw text: {text[:200]}...")
                return None

            except anthropic.APIError as e:
                self.logger.warning(f"API error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                self.stats["failed_api"] += 1

                if attempt < max_retries:
                    self.stats["retries"] += 1
                    # Exponential backoff: 5s, 10s, 20s
                    wait_time = 5 * (2 ** attempt)
                    self.logger.info(f"Backing off for {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                self.logger.error(f"All attempts failed for API call: {e}")
                return None

        return None

    def generate_cuad_based(
        self,
        clause: dict,
        rate_limit_delay: float = 0.5
    ) -> dict | None:
        """
        Generate risk assessment for a CUAD clause.

        Why rate_limit_delay=0.5s? Claude Sonnet has generous rate limits,
        but we're making 1,000+ calls. 0.5s = safe, won't hit limits.
        If you do hit limits, the retry logic will back off automatically.

        Args:
            clause: Dict from raw_clauses.jsonl with clause_text, clause_type, etc.
            rate_limit_delay: Seconds to wait between API calls

        Returns:
            Complete training example with input/output/metadata, or None if failed
        """
        time.sleep(rate_limit_delay)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            clause_type=clause["clause_type"],
            clause_text=clause["clause_text"],
            context_window=clause.get("context_window", "")
        )

        result = self.generate_single(SYSTEM_PROMPT, user_prompt)

        if result is None:
            return None

        return {
            "input": {
                "clause_text": clause["clause_text"],
                "clause_type": clause["clause_type"],
                "context_window": clause.get("context_window", ""),
                "source": "cuad",
                "source_doc": clause.get("source_doc", "unknown")
            },
            "output": result,
            "metadata": {
                "model": self.model,
                "template": "direct_risk_analysis",
                "generation_timestamp": datetime.now().isoformat(),
                "cuad_category": clause.get("cuad_category", "unknown")
            }
        }

    def generate_synthetic_clause(
        self,
        clause_type: str,
        difficulty: str,
        risk_tendency: str,
        rate_limit_delay: float = 0.5
    ) -> dict | None:
        """
        Generate a fully synthetic clause + risk assessment.

        This is used for the 2 missing categories: confidentiality, indemnification.
        Claude both creates the clause AND analyzes it in one API call.

        Why in one call? Ensures the analysis matches the clause. If we generated
        the clause first, then analyzed it separately, we'd need 2 API calls and
        risk inconsistency if the model "forgets" what it wrote.

        Args:
            clause_type: "confidentiality" or "indemnification"
            difficulty: "simple", "moderate", or "complex"
            risk_tendency: "low_risk", "moderate_risk", "high_risk", or "critical_risk"
            rate_limit_delay: Seconds to wait between API calls

        Returns:
            Complete training example with synthetic clause as input
        """
        time.sleep(rate_limit_delay)

        user_prompt = SYNTHETIC_USER_PROMPT.format(
            clause_type=clause_type,
            difficulty=difficulty,
            risk_tendency=risk_tendency
        )

        result = self.generate_single(SYNTHETIC_CLAUSE_SYSTEM_PROMPT, user_prompt)

        if result is None:
            return None

        generated_clause = result.pop("generated_clause", "")

        if not generated_clause:
            self.logger.error("Synthetic generation returned no clause text")
            return None

        return {
            "input": {
                "clause_text": generated_clause,
                "clause_type": clause_type,
                "context_window": "",  # No context for synthetic
                "source": "synthetic",
                "source_doc": "synthetic_generation"
            },
            "output": result,
            "metadata": {
                "model": self.model,
                "template": "synthetic_clause_generation",
                "generation_timestamp": datetime.now().isoformat(),
                "difficulty": difficulty,
                "risk_tendency": risk_tendency
            }
        }

    def estimate_cost(self) -> dict:
        """
        Estimate API cost based on token usage.

        Why track cost? Claude Sonnet is affordable but not free. At $3/1M input
        and $15/1M output tokens, 1,200 examples ≈ $3-5. Good to track.

        Claude Sonnet 4 pricing (as of Feb 2025):
        - Input: $3 per 1M tokens
        - Output: $15 per 1M tokens

        Returns:
            Dict with token counts and estimated cost
        """
        input_cost = (self.stats["total_input_tokens"] / 1_000_000) * 3.0
        output_cost = (self.stats["total_output_tokens"] / 1_000_000) * 15.0
        total_cost = input_cost + output_cost

        return {
            "input_tokens": self.stats["total_input_tokens"],
            "output_tokens": self.stats["total_output_tokens"],
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": total_cost
        }


def sample_clauses(
    clauses: list[dict],
    samples_per_category: dict[str, int],
    random_seed: int = 42
) -> list[dict]:
    """
    Sample clauses from CUAD data using stratified random sampling.

    Why stratified sampling? We want balanced representation across categories.
    If we just did random sampling, we might get 400 termination clauses and
    only 50 governing_law clauses, which would bias the model.

    Legal/Business Intent: A risk assessment model needs to perform well on
    ALL clause types, not just the most common ones. Stratified sampling ensures
    we train equally on rare but critical categories like governing_law.

    Args:
        clauses: List of clause dicts from raw_clauses.jsonl
        samples_per_category: Dict mapping clause_type -> number to sample
        random_seed: Random seed for reproducibility

    Returns:
        List of sampled clause dicts
    """
    random.seed(random_seed)

    # Group clauses by type
    by_type = {}
    for clause in clauses:
        clause_type = clause["clause_type"]
        if clause_type not in by_type:
            by_type[clause_type] = []
        by_type[clause_type].append(clause)

    # Sample from each category
    sampled = []
    for clause_type, num_samples in samples_per_category.items():
        available = by_type.get(clause_type, [])

        if len(available) < num_samples:
            # If we don't have enough, take all available
            sampled.extend(available)
        else:
            # Random sample without replacement
            sampled.extend(random.sample(available, num_samples))

    # Shuffle the final list to mix categories
    random.shuffle(sampled)

    return sampled
