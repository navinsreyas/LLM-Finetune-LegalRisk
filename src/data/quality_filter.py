"""
Multi-Stage Quality Filter for Synthetic Risk Assessments.

Why filter synthetic data? LLMs are non-deterministic. Even with careful prompting,
Claude might occasionally:
- Return malformed JSON
- Give inconsistent risk scores
- Provide vague recommendations
- Generate duplicate assessments

This filter removes low-quality examples BEFORE fine-tuning. Teaching a model
on bad examples is worse than having fewer examples - it learns the wrong patterns.

LEGAL ENGINEERING PRINCIPLE: "Garbage in, garbage out"
A model trained on vague recommendations like "consult a lawyer" will produce
vague outputs. We need high-quality, specific, actionable training data.
"""

import json
import logging
from pathlib import Path
from collections import Counter, defaultdict


VALID_RISK_LEVELS = {"low", "medium", "high", "critical"}
VALID_CLAUSE_TYPES = {
    "liability",
    "termination",
    "non_compete",
    "ip",
    "governing_law",
    "confidentiality",
    "indemnification"
}

# Risk score ranges for each level (with slight overlap for flexibility)
# Why overlap? A risk_score of 0.28 could reasonably be "low" or "medium"
# depending on context. We allow slight overlap but reject wild inconsistencies.
RISK_SCORE_RANGES = {
    "low": (0.0, 0.30),
    "medium": (0.26, 0.55),
    "high": (0.46, 0.80),
    "critical": (0.71, 1.0)
}

# Generic phrases that indicate low-quality recommendations
# Why filter these? They're not actionable. A lawyer reading "consult a lawyer"
# learns nothing. We need specific next steps like "negotiate a 90-day cure period".
GENERIC_PHRASES = [
    "review the clause",
    "consult a lawyer",
    "consult an attorney",
    "seek legal advice",
    "have a lawyer review",
    "review with counsel"
]


class QualityFilter:
    """
    Five-stage quality filter for synthetic risk assessments.

    Each stage is designed to catch a specific type of quality issue:
    1. Format: Is it valid JSON with required fields?
    2. Range: Are values within expected ranges?
    3. Consistency: Do risk_score and risk_level align?
    4. Content: Are recommendations specific and meaningful?
    5. Deduplication: Remove exact and near-duplicates

    The order matters: We do cheap checks first (format, range) before
    expensive checks (content analysis, deduplication).
    """

    def __init__(self, logger: logging.Logger | None = None):
        """
        Initialize the quality filter.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        self.stats = {
            "total_input": 0,
            "stage_1_format": 0,
            "stage_2_range": 0,
            "stage_3_consistency": 0,
            "stage_4_content": 0,
            "stage_5_dedup": 0,
            "total_passed": 0
        }

        self.removal_reasons = defaultdict(int)

    def stage_1_format_validation(self, example: dict) -> tuple[bool, str]:
        """
        Stage 1: Check if the example has valid structure and required fields.

        Why this first? No point checking value ranges if fields don't exist.
        This catches API failures that returned partial JSON.

        Args:
            example: The generated example dict

        Returns:
            (is_valid, reason) tuple
        """
        if not isinstance(example, dict):
            return False, "not_a_dict"

        if "output" not in example:
            return False, "missing_output_field"

        output = example["output"]
        if not isinstance(output, dict):
            return False, "output_not_dict"

        required_fields = {
            "clause_type": str,
            "risk_level": str,
            "risk_score": (int, float),  # Allow both int and float
            "key_concerns": list,
            "recommendation": str,
            "confidence": (int, float)
        }

        for field, expected_type in required_fields.items():
            if field not in output:
                return False, f"missing_field_{field}"

            if not isinstance(output[field], expected_type):
                return False, f"wrong_type_{field}"

        return True, "passed"

    def stage_2_value_range_check(self, example: dict) -> tuple[bool, str]:
        """
        Stage 2: Check if values are within expected ranges.

        Why separate from format? Format checks structure, range checks semantics.
        A field can be a float (format ✓) but be -0.5 (range ✗).

        Args:
            example: The generated example dict

        Returns:
            (is_valid, reason) tuple
        """
        output = example["output"]

        # Check risk_level is valid
        if output["risk_level"] not in VALID_RISK_LEVELS:
            return False, f"invalid_risk_level_{output['risk_level']}"

        # Check clause_type is valid
        if output["clause_type"] not in VALID_CLAUSE_TYPES:
            return False, f"invalid_clause_type_{output['clause_type']}"

        # Check risk_score range
        risk_score = output["risk_score"]
        if not (0.0 <= risk_score <= 1.0):
            return False, f"risk_score_out_of_range_{risk_score}"

        # Check confidence range
        confidence = output["confidence"]
        if not (0.0 <= confidence <= 1.0):
            return False, f"confidence_out_of_range_{confidence}"

        # Check key_concerns has exactly 3 items
        concerns = output["key_concerns"]
        if len(concerns) != 3:
            return False, f"wrong_num_concerns_{len(concerns)}"

        # Check all concerns are non-empty strings
        for i, concern in enumerate(concerns):
            if not isinstance(concern, str):
                return False, f"concern_{i}_not_string"
            if not concern.strip():
                return False, f"concern_{i}_empty"

        # Check recommendation is non-empty
        if not output["recommendation"].strip():
            return False, "recommendation_empty"

        return True, "passed"

    def stage_3_consistency_check(self, example: dict) -> tuple[bool, str]:
        """
        Stage 3: Check if risk_score aligns with risk_level.

        Why this matters: If the model says risk_level="low" but risk_score=0.95,
        it's confused. This inconsistency will confuse the fine-tuned model.

        Legal/Business Intent: Risk scores and levels must align for the model
        to learn reliable patterns. If we train on inconsistent data, the model
        won't know whether to trust the score or the level.

        Args:
            example: The generated example dict

        Returns:
            (is_valid, reason) tuple
        """
        output = example["output"]
        risk_level = output["risk_level"]
        risk_score = output["risk_score"]

        min_score, max_score = RISK_SCORE_RANGES[risk_level]

        if not (min_score <= risk_score <= max_score):
            return False, f"score_level_mismatch_{risk_level}_{risk_score}"

        input_type = example.get("input", {}).get("clause_type", "")
        output_type = output["clause_type"]

        if input_type and input_type != output_type:
            return False, f"clause_type_mismatch_{input_type}_vs_{output_type}"

        return True, "passed"

    def stage_4_content_quality_check(self, example: dict) -> tuple[bool, str]:
        """
        Stage 4: Check if the content is high-quality and specific.

        Why this stage? Passing all prior checks doesn't mean the recommendation
        is useful. "Consult a lawyer" is technically valid but useless as training data.

        This is the most legally-informed filter. We check:
        - Are recommendations specific enough? (length check)
        - Are concerns meaningful? (length check)
        - Are we avoiding generic lawyerspeak? (phrase check)

        Args:
            example: The generated example dict

        Returns:
            (is_valid, reason) tuple
        """
        output = example["output"]

        # Check recommendation length
        recommendation = output["recommendation"]

        if len(recommendation) < 20:
            return False, f"recommendation_too_short_{len(recommendation)}"

        if len(recommendation) > 500:
            return False, f"recommendation_too_long_{len(recommendation)}"

        # Check for generic phrases (case-insensitive)
        recommendation_lower = recommendation.lower()
        for phrase in GENERIC_PHRASES:
            # If the ENTIRE recommendation is just the generic phrase, reject it
            # But allow it if it's part of a longer, more specific recommendation
            if recommendation_lower.strip() == phrase:
                return False, f"generic_phrase_only_{phrase.replace(' ', '_')}"

            # Also reject if the generic phrase appears and the recommendation is short
            if phrase in recommendation_lower and len(recommendation) < 50:
                return False, f"mostly_generic_{phrase.replace(' ', '_')}"

        # Check key_concerns quality
        for i, concern in enumerate(output["key_concerns"]):
            if len(concern) < 10:
                return False, f"concern_{i}_too_short_{len(concern)}"

            if len(concern) > 200:
                return False, f"concern_{i}_too_long_{len(concern)}"

        return True, "passed"

    def stage_5_deduplication(self, examples: list[dict]) -> list[dict]:
        """
        Stage 5: Remove exact and near-duplicate examples.

        Why deduplicate? Sometimes the API returns identical or nearly identical
        assessments for different clauses. This doesn't help training - it just
        teaches the model to memorize specific outputs.

        Deduplication strategy:
        1. Exact duplicates: Same output JSON (different input is OK)
        2. Near-duplicates: Same clause_type, risk_level, AND all 3 key_concerns

        Args:
            examples: List of examples that passed stages 1-4

        Returns:
            Deduplicated list of examples
        """
        seen_outputs = set()  # For exact duplicates
        seen_fingerprints = set()  # For near-duplicates
        deduplicated = []

        for example in examples:
            output = example["output"]

            # Exact duplicate check: serialize output to JSON string
            output_str = json.dumps(output, sort_keys=True)
            if output_str in seen_outputs:
                self.stats["stage_5_dedup"] += 1
                self.removal_reasons["exact_duplicate"] += 1
                continue

            # Near-duplicate check: create fingerprint
            # A fingerprint is: clause_type + risk_level + all 3 concerns (sorted)
            fingerprint = (
                output["clause_type"],
                output["risk_level"],
                tuple(sorted(output["key_concerns"]))
            )

            if fingerprint in seen_fingerprints:
                self.stats["stage_5_dedup"] += 1
                self.removal_reasons["near_duplicate"] += 1
                continue

            # Not a duplicate - keep it
            seen_outputs.add(output_str)
            seen_fingerprints.add(fingerprint)
            deduplicated.append(example)

        return deduplicated

    def filter_examples(self, examples: list[dict]) -> list[dict]:
        """
        Run all 5 filter stages on a list of examples.

        This is the main entry point. It processes examples through each stage
        sequentially, removing failures at each step.

        Args:
            examples: List of generated examples

        Returns:
            List of examples that passed all filters
        """
        self.stats["total_input"] = len(examples)
        self.logger.info(f"Starting quality filter on {len(examples)} examples")

        filtered = []

        for example in examples:
            is_valid, reason = self.stage_1_format_validation(example)
            if not is_valid:
                self.stats["stage_1_format"] += 1
                self.removal_reasons[f"stage1_{reason}"] += 1
                continue

            is_valid, reason = self.stage_2_value_range_check(example)
            if not is_valid:
                self.stats["stage_2_range"] += 1
                self.removal_reasons[f"stage2_{reason}"] += 1
                continue

            is_valid, reason = self.stage_3_consistency_check(example)
            if not is_valid:
                self.stats["stage_3_consistency"] += 1
                self.removal_reasons[f"stage3_{reason}"] += 1
                continue

            is_valid, reason = self.stage_4_content_quality_check(example)
            if not is_valid:
                self.stats["stage_4_content"] += 1
                self.removal_reasons[f"stage4_{reason}"] += 1
                continue

            filtered.append(example)

        self.logger.info(f"After stages 1-4: {len(filtered)} examples remain")

        filtered = self.stage_5_deduplication(filtered)

        self.stats["total_passed"] = len(filtered)
        self.logger.info(f"After deduplication: {len(filtered)} examples remain")

        return filtered

    def get_report(self, filtered_examples: list[dict]) -> dict:
        """
        Generate a detailed filter report.

        Why track this? We need to know what's failing and why. If 50% of examples
        fail stage 4 (content quality), our prompts need improvement.

        Args:
            filtered_examples: The examples that passed filtering

        Returns:
            Dict with statistics and distribution info
        """
        # Calculate distribution after filtering
        distribution = Counter()
        for example in filtered_examples:
            clause_type = example["output"]["clause_type"]
            distribution[clause_type] += 1

        # Calculate pass rate
        pass_rate = (
            self.stats["total_passed"] / self.stats["total_input"]
            if self.stats["total_input"] > 0
            else 0.0
        )

        report = {
            "total_input": self.stats["total_input"],
            "stage_1_removed": self.stats["stage_1_format"],
            "stage_2_removed": self.stats["stage_2_range"],
            "stage_3_removed": self.stats["stage_3_consistency"],
            "stage_4_removed": self.stats["stage_4_content"],
            "stage_5_removed": self.stats["stage_5_dedup"],
            "total_passed": self.stats["total_passed"],
            "pass_rate": round(pass_rate, 3),
            "distribution_after_filter": dict(distribution),
            "top_removal_reasons": dict(
                sorted(
                    self.removal_reasons.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # Top 10 reasons
            )
        }

        return report


def filter_and_save(
    input_path: Path | str,
    output_path: Path | str,
    report_path: Path | str,
    logger: logging.Logger | None = None
) -> dict:
    """
    Load examples, filter them, and save results.

    This is a convenience function that wraps the QualityFilter class
    for use in scripts.

    Args:
        input_path: Path to raw generated JSONL
        output_path: Path to save filtered JSONL
        report_path: Path to save filter report JSON
        logger: Optional logger instance

    Returns:
        Filter report dict
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    report_path = Path(report_path)

    logger = logger or logging.getLogger(__name__)

    # Load examples
    logger.info(f"Loading examples from {input_path}")
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))

    # Filter
    qf = QualityFilter(logger=logger)
    filtered = qf.filter_examples(examples)

    # Save filtered examples
    logger.info(f"Saving {len(filtered)} filtered examples to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in filtered:
            f.write(json.dumps(example) + '\n')

    # Generate and save report
    report = qf.get_report(filtered)
    logger.info(f"Saving filter report to {report_path}")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    return report
