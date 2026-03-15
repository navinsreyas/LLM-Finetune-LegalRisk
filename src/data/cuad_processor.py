"""
CUAD Dataset Processor for LegalRisk-LLM Pipeline.

Extracts individual clause spans (not full contracts) from the Contract
Understanding Atticus Dataset (CUAD) v1 with intelligent filtering.

KEY ARCHITECTURAL DECISION:
We extract at the ANSWER-SPAN level, not the context/paragraph level.
Why? Each answer is a specific labeled clause (e.g., "Indemnification" clause).
The full context is the entire contract, which would teach the model nothing
about clause-specific risk patterns.
"""

import json
from pathlib import Path
from collections import Counter
import pandas as pd
import re


# LEGAL ENGINEERING MAPPING
# Why these 5 categories? These represent the core risk dimensions in commercial contracts
# that drive business decisions. We collapsed CUAD's 41 granular categories into
# 5 high-level types because:
# 1. Many CUAD categories are legally related (e.g., "Cap on Liability" + "Uncapped Liability")
# 2. Fine-tuning needs ~500+ examples per category - 5 categories balances granularity with data needs
# 3. These 5 align with how in-house counsel actually think about contract risk
# 4. CUAD doesn't have "Confidentiality" or explicit "Indemnification" categories (limitation of dataset)
CUAD_TO_TARGET_CATEGORY = {
    # Liability: Who pays if something goes wrong?
    # NOTE: CUAD doesn't have generic "Liability" or "Indemnification" categories
    # We use Cap/Uncapped Liability + Insurance + Liquidated Damages as proxies
    "Cap On Liability": "liability",
    "Uncapped Liability": "liability",
    "Liquidated Damages": "liability",  # Pre-agreed damages = liability-related
    "Insurance": "liability",  # Insurance requirements often serve as liability caps

    # Termination: How and when can we exit this contract?
    "Termination For Convenience": "termination",
    "Rofr/Rofo/Rofn": "termination",  # Right of first refusal = exit-related
    "Expiration Date": "termination",
    "Renewal Term": "termination",
    "Notice Period To Terminate Renewal": "termination",
    "Post-Termination Services": "termination",
    "Change Of Control": "termination",  # Change of control often triggers termination rights

    # Non-compete: What can't we do during/after the contract?
    "Non-Compete": "non_compete",
    "Exclusivity": "non_compete",
    "No-Solicit Of Employees": "non_compete",
    "No-Solicit Of Customers": "non_compete",
    "Non-Disparagement": "non_compete",
    "Competitive Restriction Exception": "non_compete",  # Carveouts to non-compete = still relevant

    # Intellectual Property: Who owns what we create?
    "Ip Ownership Assignment": "ip",
    "Joint Ip Ownership": "ip",
    "License Grant": "ip",
    "Irrevocable Or Perpetual License": "ip",
    "Non-Transferable License": "ip",

    # Governing Law: Which court/law applies if we fight?
    "Governing Law": "governing_law",

    # NOTE: CUAD does NOT have a "Confidentiality" category, so we can't extract those clauses
    # If you need confidentiality clauses, you'll need a different dataset (e.g., EDGAR 10-Ks, NDAs)
}

# Filtering thresholds (explained)
MIN_CLAUSE_LENGTH = 20    # Below this = likely just a date or party name, not a clause
MAX_CLAUSE_LENGTH = 3000  # Above this = likely an extraction error or merged clauses
CONTEXT_WINDOW = 200      # Characters before/after clause for context
# Why 200 chars? Legal clauses often reference prior terms ("as defined in Section 3").
# 200 chars ≈ 40-50 words, enough to capture cross-references without bloating data.


def extract_context_window(
    full_text: str,
    start_pos: int,
    clause_length: int,
    window_size: int = CONTEXT_WINDOW
) -> str:
    """
    Extract surrounding context for a clause span.

    Why we need this: A clause like "termination requires 30 days notice" means
    different things if the preceding text says "either party" vs "only Company X".
    The context window teaches the model to attend to surrounding legal qualifiers.

    Args:
        full_text: The complete contract text
        start_pos: Character offset where clause begins
        clause_length: Length of the clause text
        window_size: Characters to grab before/after (default 200)

    Returns:
        Formatted string: "...context before... [CLAUSE] ...context after..."
    """
    end_pos = start_pos + clause_length

    # Get surrounding text, clamped to document boundaries
    before_start = max(0, start_pos - window_size)
    after_end = min(len(full_text), end_pos + window_size)

    before_text = full_text[before_start:start_pos].strip()
    clause_text = full_text[start_pos:end_pos].strip()
    after_text = full_text[end_pos:after_end].strip()

    # Format with markers (the [CLAUSE] marker helps the model distinguish the target from context)
    parts = []
    if before_text:
        parts.append(f"...{before_text}")
    parts.append(f"[CLAUSE: {clause_text}]")
    if after_text:
        parts.append(f"{after_text}...")

    return " ".join(parts)


def process_cuad_dataset(
    input_path: str | Path = "data/CUAD_v1.json",
    output_path: str | Path = "data/raw_clauses.jsonl",
) -> None:
    """
    Extract and filter individual clause spans from CUAD v1 dataset.

    CRITICAL CHANGE FROM V1: We now extract at the ANSWER level, not paragraph level.
    Each QA answer is a labeled clause span. This is why we go from ~500 entries
    (one per contract) to ~3000+ entries (many clauses per contract).

    Args:
        input_path: Path to CUAD_v1.json file
        output_path: Path for output JSONL file

    Raises:
        FileNotFoundError: If input dataset doesn't exist
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset with error handling
    try:
        print(f"[*] Loading CUAD dataset from: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"[ERROR] CUAD dataset not found at: {input_path.absolute()}\n"
            f"   Please download CUAD_v1.json and place it in the data/ directory."
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"[ERROR] Invalid JSON format in {input_path}: {e}")

    # Statistics tracking
    total_qa_pairs = 0
    matched_categories = 0
    filtered_too_short = 0
    filtered_too_long = 0
    filtered_empty = 0
    unmatched_categories = set()  # Track categories we're skipping

    clauses = []  # Will hold all extracted clause dicts

    for document in dataset.get('data', []):
        source_doc = document.get('title', 'Unknown')

        for paragraph in document.get('paragraphs', []):
            # This is the FULL contract text (thousands of words)
            full_context = paragraph.get('context', '')

            # Each QA pair represents a category of clauses (e.g., "Indemnification")
            for qa in paragraph.get('qas', []):
                total_qa_pairs += 1

                # The "question" contains the category name in quotes
                # Example: 'Highlight the parts (if any) of this contract related to "Cap On Liability" that...'
                question = qa.get('question', '').strip()

                # Extract category name from between quotes after "related to"
                match = re.search(r'related to "([^"]+)"', question)
                if not match:
                    continue

                cuad_category = match.group(1).strip()

                # Map to our 5 target categories (skip if not in mapping)
                target_category = CUAD_TO_TARGET_CATEGORY.get(cuad_category)
                if not target_category:
                    unmatched_categories.add(cuad_category)
                    continue  # Skip CUAD categories we don't care about

                matched_categories += 1

                for answer in qa.get('answers', []):
                    clause_text = answer.get('text', '').strip()
                    answer_start = answer.get('answer_start', 0)

                    if not clause_text:
                        filtered_empty += 1
                        continue

                    if len(clause_text) < MIN_CLAUSE_LENGTH:
                        filtered_too_short += 1
                        continue

                    if len(clause_text) > MAX_CLAUSE_LENGTH:
                        filtered_too_long += 1
                        continue

                    context_window = extract_context_window(
                        full_context,
                        answer_start,
                        len(clause_text),
                        CONTEXT_WINDOW
                    )

                    # Calculate token estimate (rough heuristic: 1 token ≈ 4 chars)
                    # Why estimate tokens? We'll use this later to batch clauses for LLM API calls
                    token_estimate = len(clause_text) // 4

                    clauses.append({
                        'clause_text': clause_text,
                        'clause_type': target_category,
                        'source_doc': source_doc,
                        'cuad_category': cuad_category,
                        'context_window': context_window,
                        'char_length': len(clause_text),
                        'token_estimate': token_estimate
                    })

    # Debug: Print sample unmatched categories
    if unmatched_categories:
        print(f"\n[DEBUG] Sample unmatched CUAD categories (first 10):")
        for cat in sorted(list(unmatched_categories))[:10]:
            print(f"  - {cat}")

    if not clauses:
        print(f"\n[ERROR] No clauses extracted! Check category mapping.")
        print(f"[DEBUG] Total QA pairs scanned: {total_qa_pairs:,}")
        print(f"[DEBUG] Matched categories: {matched_categories:,}")
        return

    initial_count = len(clauses)
    df = pd.DataFrame(clauses)
    df_unique = df.drop_duplicates(subset=['clause_text'], keep='first')
    final_count = len(df_unique)
    duplicates_removed = initial_count - final_count

    clause_type_counts = Counter(df_unique['clause_type'])
    lengths = df_unique['char_length']

    df_unique.to_json(
        output_path,
        orient='records',
        lines=True,
        force_ascii=False
    )

    print(f"\n{'='*70}")
    print(f"[SUCCESS] CUAD Clause Extraction Complete")
    print(f"{'='*70}")
    print(f"[STATS] Total QA pairs scanned:          {total_qa_pairs:,}")
    print(f"[STATS] Matched to target categories:    {matched_categories:,}")
    print(f"[FILTER] Filtered (empty text):          {filtered_empty:,}")
    print(f"[FILTER] Filtered (too short < {MIN_CLAUSE_LENGTH}):    {filtered_too_short:,}")
    print(f"[FILTER] Filtered (too long > {MAX_CLAUSE_LENGTH}):   {filtered_too_long:,}")
    print(f"[EXTRACT] Initial clauses extracted:     {initial_count:,}")
    print(f"[DEDUP] Duplicates removed:              {duplicates_removed:,}")
    print(f"[FINAL] Final clauses saved:             {final_count:,}")
    print(f"\n[DISTRIBUTION] Distribution by clause type:")
    for clause_type, count in clause_type_counts.most_common():
        percentage = (count / final_count) * 100
        print(f"   {clause_type:20s} {count:5,} ({percentage:5.1f}%)")
    print(f"\n[LENGTH] Clause length statistics:")
    print(f"   Min:  {lengths.min():,} chars")
    print(f"   Max:  {lengths.max():,} chars")
    print(f"   Mean: {int(lengths.mean()):,} chars")
    print(f"   Median: {int(lengths.median()):,} chars")
    print(f"\n[OUTPUT] Output location: {output_path.absolute()}")
    print(f"{'='*70}\n")


def verify_output(output_path: str | Path = "data/raw_clauses.jsonl") -> None:
    """
    Load and display the first 3 entries to visually confirm we extracted
    clause-level data (short paragraphs) not contract-level data (pages of text).

    Why this matters: If you see thousands of words per entry, something broke.
    Good clause extractions should be 50-500 words typically.
    """
    output_path = Path(output_path)

    if not output_path.exists():
        print(f"[ERROR] Output file not found: {output_path}")
        return

    print(f"\n{'='*70}")
    print(f"[VERIFY] VERIFICATION: First 3 Extracted Clauses")
    print(f"{'='*70}\n")

    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Only show first 3
                break

            entry = json.loads(line)

            print(f"{'-'*70}")
            print(f"Entry #{i+1}")
            print(f"{'-'*70}")
            print(f"Clause Type:     {entry['clause_type']}")
            print(f"CUAD Category:   {entry['cuad_category']}")
            print(f"Source Doc:      {entry['source_doc']}")
            print(f"Char Length:     {entry['char_length']:,}")
            print(f"Token Estimate:  {entry['token_estimate']:,}")
            print(f"\nClause Text (first 300 chars):")
            print(f"{entry['clause_text'][:300]}...")
            print(f"\nContext Window (first 400 chars):")
            print(f"{entry['context_window'][:400]}...")
            print()

    print(f"{'='*70}")
    print(f"[SUCCESS] If you see short clause paragraphs (not full contracts), extraction worked!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Execute processing pipeline
    process_cuad_dataset()

    # Verify output
    verify_output()
