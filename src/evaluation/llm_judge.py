"""
LLM-as-Judge evaluator using Claude Sonnet API.

Scores model predictions on 5 dimensions:
- Accuracy: Is the risk assessment factually correct?
- Completeness: Are all major risks identified?
- Legal Reasoning: Is the legal logic sound?
- Clarity: Is the analysis clear and well-structured?
- Actionability: Are recommendations specific and implementable?

Estimated cost: ~$0.0075 per call, ~$3.21 for 428 evaluations.
"""

import json
import time

from anthropic import Anthropic

from src.utils.config import get_anthropic_key


JUDGE_SYSTEM_PROMPT = """You are an expert legal evaluation specialist. Your job is to score AI-generated legal risk assessments by comparing them against expert reference analyses.

You will receive:
1. The original contract clause
2. A reference (expert) analysis
3. A model's predicted analysis

Score the model's prediction on these 5 dimensions (1-10 scale):

SCORING RUBRIC:

**Accuracy (1-10)**: Does the model correctly identify the risk level and risk score?
- 9-10: Risk level matches exactly, risk score within 0.1
- 7-8: Risk level off by one step (e.g., medium vs high), reasonable score
- 5-6: Risk level off by one step with poor score calibration
- 3-4: Risk level off by two steps (e.g., low vs high)
- 1-2: Completely wrong risk assessment

**Completeness (1-10)**: Does the model identify all the key legal risks?
- 9-10: All major concerns from reference are covered, plus valid additional ones
- 7-8: Most major concerns covered, one minor gap
- 5-6: Some concerns covered but missing one important risk
- 3-4: Only surface-level concerns, missing critical risks
- 1-2: Concerns are mostly irrelevant or generic

**Legal Reasoning (1-10)**: Does the analysis show genuine legal understanding?
- 9-10: Demonstrates sophisticated understanding of legal implications
- 7-8: Shows solid legal reasoning with minor gaps
- 5-6: Basic legal reasoning, somewhat generic
- 3-4: Shallow reasoning, could apply to any clause
- 1-2: No meaningful legal reasoning

**Clarity (1-10)**: Is the output clear, well-structured, and easy to understand?
- 9-10: Crystal clear, well-organized, professional quality
- 7-8: Clear and readable, minor structure issues
- 5-6: Understandable but verbose or poorly organized
- 3-4: Confusing or ambiguous language
- 1-2: Incoherent or contradictory

**Actionability (1-10)**: Are the recommendations specific enough to act on?
- 9-10: Specific negotiation points, concrete language suggestions
- 7-8: Good recommendations with some specificity
- 5-6: General direction is right but lacks specific action items
- 3-4: Vague advice like "consult a lawyer"
- 1-2: No actionable recommendation

Respond with ONLY this JSON format, no other text:
{"scores": {"accuracy": <int>, "completeness": <int>, "legal_reasoning": <int>, "clarity": <int>, "actionability": <int>}, "overall": <float>, "justification": "<2-3 sentences>"}"""


class LLMJudge:
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize the LLM Judge using Claude Sonnet.

        Args:
            model: Claude model to use as judge. Sonnet balances quality and cost.
                   Do NOT use Haiku (evaluation quality too low).
                   Do NOT use Opus (too expensive for 428 calls).
            max_retries: Retry count for API failures
            retry_delay: Base delay between retries (exponential backoff)
        """
        api_key = get_anthropic_key()
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.failed_calls = 0
        print(f"[JUDGE] Initialized with model: {model}")
        print(f"[JUDGE] Estimated cost: ~$3.21 for 428 evaluations")

    def evaluate_single(
        self,
        clause_text: str,
        clause_type: str,
        ground_truth: dict,
        prediction: dict,
    ) -> dict:
        """
        Evaluate a single model prediction against ground truth.

        Args:
            clause_text: The original contract clause
            clause_type: Type of clause (termination, liability, etc.)
            ground_truth: The reference/expert analysis
            prediction: The model's predicted analysis

        Returns:
            dict with 'scores', 'overall', 'justification', and metadata
        """
        user_message = self._build_judge_prompt(
            clause_text, clause_type, ground_truth, prediction
        )

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    system=JUDGE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )

                # Track token usage for cost estimation
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens

                raw_text = response.content[0].text.strip()
                result = self._parse_judge_response(raw_text)

                if result is not None:
                    return result
                else:
                    print(f"  [JUDGE] Parse failed (attempt {attempt+1}), retrying...")
                    print(f"  [JUDGE] Raw: {raw_text[:200]}")

            except Exception as e:
                wait_time = self.retry_delay * (2 ** attempt)
                error_msg = str(e).lower()

                # Handle rate limit
                if "429" in str(e) or "rate" in error_msg:
                    wait_time = max(wait_time, 30)
                    print(f"  [JUDGE] Rate limited. Waiting {wait_time:.0f}s...")
                # Handle overloaded
                elif "529" in str(e) or "overloaded" in error_msg:
                    wait_time = max(wait_time, 15)
                    print(f"  [JUDGE] API overloaded. Waiting {wait_time:.0f}s...")
                else:
                    print(f"  [JUDGE] API error (attempt {attempt+1}): {e}")
                    print(f"  [JUDGE] Retrying in {wait_time:.1f}s...")

                time.sleep(wait_time)

        # All retries failed
        self.failed_calls += 1
        print("  [JUDGE] All retries failed, returning default scores")
        return {
            "scores": {
                "accuracy": 0,
                "completeness": 0,
                "legal_reasoning": 0,
                "clarity": 0,
                "actionability": 0,
            },
            "overall": 0.0,
            "justification": "Evaluation failed after all retries",
            "eval_failed": True,
        }

    def _build_judge_prompt(
        self, clause_text: str, clause_type: str, ground_truth: dict, prediction: dict
    ) -> str:
        """Build the user message for the judge."""
        if len(clause_text) > 2000:
            clause_text = clause_text[:2000] + "... [truncated]"

        prompt = f"""Evaluate the following legal risk assessment:

CLAUSE TYPE: {clause_type}

ORIGINAL CLAUSE:
{clause_text}

EXPERT REFERENCE ANALYSIS:
{json.dumps(ground_truth, indent=2)}

MODEL'S PREDICTION:
{json.dumps(prediction, indent=2) if prediction else "FAILED TO PRODUCE VALID JSON OUTPUT"}

Score the model's prediction against the expert reference on all 5 dimensions.
If the model failed to produce valid JSON, score all dimensions as 1.
Respond with ONLY the JSON scores."""

        return prompt

    def _parse_judge_response(self, raw_text: str) -> dict:
        """Parse the judge's JSON response."""
        try:
            clean_text = raw_text

            # Handle potential markdown wrapping
            if clean_text.startswith("```"):
                clean_text = clean_text.split("```")[1]
                if clean_text.startswith("json"):
                    clean_text = clean_text[4:]
                clean_text = clean_text.strip()

            # If not starting with {, try to extract JSON
            if not clean_text.startswith("{"):
                start_idx = clean_text.find("{")
                end_idx = clean_text.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    clean_text = clean_text[start_idx:end_idx]

            result = json.loads(clean_text)

            if "scores" not in result:
                return None

            required_dims = [
                "accuracy", "completeness", "legal_reasoning", "clarity", "actionability"
            ]
            for dim in required_dims:
                if dim not in result["scores"]:
                    return None
                score = result["scores"][dim]
                if not isinstance(score, (int, float)):
                    return None
                result["scores"][dim] = max(1, min(10, int(score)))

            # Recompute overall with our weights (don't trust model math)
            s = result["scores"]
            result["overall"] = round(
                s["accuracy"] * 0.25 +
                s["completeness"] * 0.20 +
                s["legal_reasoning"] * 0.25 +
                s["clarity"] * 0.15 +
                s["actionability"] * 0.15,
                2
            )

            result["eval_failed"] = False
            return result

        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def get_cost_estimate(self) -> dict:
        """Estimate total API cost based on token usage."""
        # Claude Sonnet pricing
        input_cost_per_million = 3.00
        output_cost_per_million = 15.00

        input_cost = (self.total_input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (self.total_output_tokens / 1_000_000) * output_cost_per_million

        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
            "total_cost_usd": round(input_cost + output_cost, 4),
        }
