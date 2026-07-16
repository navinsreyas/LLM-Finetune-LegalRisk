---
title: Legalrisk Analyzer
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: RAG-based legal contract clause risk analyzer
---

# LegalRisk-LLM: Benchmarking PEFT Fine-Tuning vs RAG for Legal Contract Risk Analysis

[![CI](https://github.com/navinsreyas/LLM-Finetune-LegalRisk/actions/workflows/ci.yml/badge.svg)](https://github.com/navinsreyas/LLM-Finetune-LegalRisk/actions/workflows/ci.yml)

A research system that compares three parameter-efficient fine-tuning methods (QLoRA, DoRA, IA3) against a RAG baseline on Llama-3.2-3B-Instruct for structured legal risk assessment. Built on real CUAD contract data augmented with Claude Sonnet synthetic examples, evaluated with an LLM judge, and subjected to full statistical significance testing.

Total compute: ~8 hours on an RTX 4060. Total API cost: ~$7.62 (estimated).

---

## TL;DR

- **What:** Fine-tuned Llama-3.2-3B-Instruct using 3 PEFT methods (QLoRA, DoRA, IA3) and built a RAG baseline. Each method reads a contract clause and outputs a structured JSON risk assessment (risk level, score, concerns, recommendation).
- **Data:** 855 training / 107 test examples across 7 clause types (termination, liability, non_compete, ip, governing_law, confidentiality, indemnification). Built from CUAD contracts + Claude Sonnet synthetic augmentation.
- **Honest finding:** No method achieves statistical significance over others at n=107 (Kruskal–Wallis H=1.77, p=0.62). Fine-tuning and RAG are complementary -- fine-tuning wins clarity/structure, RAG wins risk calibration.
- **Critical limitation:** Of the 7 Critical-risk test clauses, QLoRA, IA3, and RAG each misclassify 6 and DoRA misclassifies 5 -- usually predicting High (sometimes Low), rarely Critical. This is a safety concern that must be addressed before any production deployment.
- **Takeaway:** At 3B scale with ~855 examples, PEFT methods are statistically indistinguishable overall but differ meaningfully on specific dimensions.

---
## Architecture

```
CUAD Dataset (510 real contracts)
        |
        v
[Phase 1B] Clause Span Extraction
        -> 5,847 spans across 5 CUAD types
           (termination, liability, ip, non_compete, governing_law)
        |
        v
[Phase 1C] Synthetic Augmentation via Claude Sonnet
        -> confidentiality and indemnification introduced as synthetic types
           (CUAD has no such categories; Claude generates both clause + label)
        -> 855 train / 108 val / 107 test structured risk assessments
        |
        +---------------------------------------------+
        v                                             v
[Phase 2] PEFT Fine-Tuning                  [Phase 3A] RAG Baseline
  QLoRA / DoRA / IA3                          ChromaDB + sentence-transformers
  4-bit NF4, RTX 4060 8GB                     855 training clauses embedded
  r=16 LoRA rank for QLoRA/DoRA               5-shot retrieval per query
  IA3 rescales activations (0.29M params)     Base Llama-3.2-3B (no adapter)
        |                                             |
        +--------------------+------------------------+
                             v
              [Phase 3B] LLM-as-Judge
                Claude Sonnet scores each prediction
                5 dimensions scored across 107 test clauses x 4 methods = 428 judge evaluations
                             |
                             v
              [Phase 3C] Statistical Significance Testing
                Shapiro-Wilk (normality) . Kruskal-Wallis (global)
                Mann-Whitney U pairwise . Bonferroni . chi-square (JSON rates)
                             |
                             v
              [Phase 3D] Error Analysis
                Confusion matrices . Directional bias
                Cross-method failure overlap . Failure taxonomy
                             |
                             v
              [Phase 4] Visualizations
                10 publication-quality PNG charts
```

**Data pipeline:** CUAD JSON contracts are split into character-level clause spans. Claude Sonnet generates a structured JSON risk assessment for each span (risk level, score, concerns, recommendation). A quality filter removes short, malformed, or unbalanced examples. Final splits: 855 train / 108 val / 107 test.

**Training:** Three PEFT adapters trained on Llama-3.2-3B-Instruct with 4-bit NF4 quantization and gradient checkpointing to fit 8GB VRAM. QLoRA and DoRA use LoRA with r=16; IA3 rescales attention keys/values and FFN activations using ~85x fewer trainable parameters (0.29M vs 24.3M).

**RAG baseline:** ChromaDB vector store with all 855 training clauses embedded via all-MiniLM-L6-v2 (384-dim). At inference, retrieves the 5 most similar clauses by clause type and passes them as few-shot context to the base model (no adapter loaded).

**Evaluation:** 4-layer stack -- deterministic metrics -> LLM-as-Judge -> statistical testing -> error analysis. Each layer reveals something the previous one misses.

---

## Results

| Method | Accuracy | Judge Score | Clarity | Risk Bias | Trainable Params |
|--------|----------|-------------|---------|-----------|------------------|
| QLoRA  | 45.8%    | 6.67 / 10   | 8.04    | -0.421    | 24.3M            |
| DoRA   | 44.9%    | 6.69 / 10   | 8.09    | -0.374    | 25.1M            |
| IA3    | 47.7%    | 6.45 / 10   | 7.84    | -0.274    | 0.29M            |
| RAG    | 52.3%    | 6.49 / 10   | 7.45    | -0.089    | 0 (retrieval)    |

*Accuracy = exact risk level match (results/phase3d_error_analysis.json → confusion_matrices.&lt;method&gt;.accuracy, n=107). Risk Bias = mean(predicted_rank - true_rank) over risk levels (results/phase3d_error_analysis.json → risk_bias.&lt;method&gt;.mean_bias); negative = systematic under-prediction of severity. No pairwise comparison survived Bonferroni correction for the overall judge score (Kruskal–Wallis p=0.62; all six pairwise Mann–Whitney p=1.0 after Bonferroni).*

---

## Key Findings

**DoRA equals QLoRA at 3B scale with 855 examples.** DoRA's theoretical advantage -- decomposing LoRA weights into independent direction and magnitude components -- provides no measurable benefit here. The judge score gap is 0.015 points (6.69 vs 6.67), and the Mann–Whitney U test for QLoRA vs DoRA is not significant (p=0.98 raw, p=1.0 after Bonferroni; effect size negligible). This is a practical data point for practitioners: weight decomposition may not be worth the implementation complexity on small fine-tuning datasets.

**Fine-tuning and RAG are complementary, not competing.** Fine-tuning teaches the model how to format a legal analysis -- trained on 855 schema-consistent examples, it produces clean, well-structured JSON responses (clarity 8.04-8.09 vs 7.45 for RAG). RAG provides factual calibration -- it retrieves real examples before answering, which largely eliminates the systematic under-prediction bias that fine-tuned models exhibit (mean_bias -0.089 for RAG vs -0.27 to -0.42 for the fine-tuned methods). The ideal production system would combine both: a fine-tuned model for output structure with retrieval-augmented calibration for risk level accuracy.

**IA3 hits a capacity ceiling.** With ~85x fewer trainable parameters (0.29M vs 24.3M), IA3 has the lowest mean completeness (5.65) and legal_reasoning (6.23) of the four methods -- though at n=107 the difference is not statistically significant (legal_reasoning Kruskal–Wallis p=0.19). The IA3 mechanism -- scaling activations with learned vectors rather than adding low-rank weight deltas -- may be less able to reshape the model's output behavior when the legal domain requires genuine content changes, not just style shifts.

**The critical risk blindspot is a deployment blocker.** Of the seven Critical-risk clauses in the test set, QLoRA, IA3, and RAG each misclassify six and DoRA misclassifies five -- the models usually predict High, sometimes Low or Medium, but rarely Critical. Under-prediction is the more dangerous failure mode in legal contexts: a lawyer who follows a "Low" risk recommendation on a Critical clause could expose their client to catastrophic liability. Any production deployment needs a calibration layer, human review for High/Critical outputs, or a specialized classifier for edge-case detection.

**Statistical power is the real limitation.** At n=107 the global Kruskal–Wallis test is not significant (H=1.77, p=0.62), and none of the six pairwise Mann–Whitney comparisons survive Bonferroni correction (all adjusted p=1.0). Of the five judge dimensions, only clarity shows a globally significant difference across methods (Kruskal–Wallis p=0.015); accuracy, completeness, legal reasoning, and actionability do not. "Not significant" does not mean "no difference" -- it means the test set is too small to distinguish methods this close with confidence.

---

## Evaluation Methodology

This project uses a 4-layer evaluation stack where each layer adds something the previous one cannot provide.

**Layer 1 -- Deterministic Metrics.** Accuracy (exact risk level match), JSON parse success rate, per-class precision/recall/F1, and confusion matrices. Fully reproducible and zero cost. Catches gross failures but cannot distinguish between a wrong answer that was close in reasoning vs one that was entirely wrong.

**Layer 2 -- LLM-as-Judge.** Claude Sonnet scores each prediction on 5 dimensions (accuracy, completeness, legal reasoning, clarity, actionability) on a 1-10 scale. 428 total evaluations. The overall score is recomputed from dimension scores with fixed weights rather than trusting the model's own arithmetic. This reveals qualitative differences that accuracy alone misses -- specifically, that fine-tuning produces better-formatted responses even when the risk level classification is wrong.

**Layer 3 -- Statistical Significance Testing.** Shapiro-Wilk tests confirm the judge scores are not normally distributed, so non-parametric tests are used throughout. Kruskal–Wallis tests for a global difference across all four methods (overall and per dimension). Mann–Whitney U runs the six pairwise comparisons with Bonferroni correction (alpha = 0.05/6 ≈ 0.0083). A chi-square test compares JSON parse success rates across methods. Effect sizes (rank-biserial correlation, eta-squared) accompany the tests. Implemented in `outputs/phase3c_statistical_tests.py`.

**Layer 4 -- Error Analysis.** Per-class confusion matrices, directional bias (does each method systematically over- or under-predict severity?), cross-method failure overlap (which examples are universally hard vs method-specific failures), worst-case example mining, and judge critique taxonomy. This is where the critical risk blindspot and the systematic under-prediction bias were identified.

---

## Project Structure

```
LegalRisk-LLM/
|-- README.md
|-- requirements.txt
|-- .env.example
|-- .gitignore
|-- data/
|   |-- raw_clauses.jsonl            # 5,847 CUAD clause spans
|   +-- synthetic/                   # 855/108/107 train/val/test splits
|-- src/
|   |-- data/                        # CUAD processor, schema, quality filter
|   |-- training/                    # QLoRA, DoRA, IA3 trainers
|   |-- rag/                         # Embedder, vector store, RAG pipeline
|   +-- evaluation/                  # LLM judge, score aggregator
|-- scripts/                         # CLI entrypoints (train, inference, judge)
|-- outputs/                         # Phase 3C/3D/4 analysis scripts
|-- evaluation/                      # Prediction JSONL + judge score files
|   +-- judge_scores/
|-- models/                          # Trained PEFT adapters (gitignored)
|   |-- qlora/
|   |-- dora/
|   +-- ia3/
+-- results/
    |-- phase3c_statistical_results.json
    |-- phase3d_error_analysis.json
    +-- visualizations/              # 10 PNG charts
```

---

## Setup and Reproduction

**Prerequisites:** Python 3.10+, CUDA GPU with 8GB+ VRAM (training only), ANTHROPIC_API_KEY.

```bash
git clone https://github.com/navinsreyas/LLM-Finetune-LegalRisk.git
cd LLM-Finetune-LegalRisk
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

**Running each phase:**

```bash
# Prerequisite: download the CUAD v1 dataset (gitignored, not shipped in this repo).
# Get it from the Atticus Project (https://www.atticusprojectai.org/cuad); the release
# contains CUAD_v1.json. Place that file at:  data/CUAD_v1.json

# Phase 1B: Extract clause spans from CUAD
python src/data/cuad_processor.py

# Phase 1C: Generate synthetic risk assessments (~$5.44)
python scripts/generate_synthetic.py

# Phase 2: Train PEFT adapters (requires GPU)
python scripts/train.py --method qlora
python scripts/train.py --method dora
python scripts/train.py --method ia3

# Phase 2B: Fine-tuned inference on the test set
# (produces evaluation/<method>_predictions.jsonl -- the judge needs all four methods)
python scripts/run_finetuned_inference.py --method qlora
python scripts/run_finetuned_inference.py --method dora
python scripts/run_finetuned_inference.py --method ia3

# Phase 3A: RAG baseline -- build the vector index FIRST, then run inference
python scripts/build_rag_index.py
python scripts/run_rag_inference.py

# Phase 3B: LLM-as-Judge evaluation (~$2.18)
python scripts/run_judge.py

# Phase 3C: Statistical significance testing
python outputs/phase3c_statistical_tests.py

# Phase 3D: Error analysis
python outputs/phase3d_error_analysis.py

# Phase 4: Generate visualizations
python outputs/phase4_visualizations.py
```

Note: Adapter weights are gitignored due to size (QLoRA ~578MB, DoRA ~594MB, IA3 ~58MB -- IA3 is roughly 10x smaller). Phases 3C onward can be reproduced without training using the pre-generated JSONL files in evaluation/.

---

## Cost Breakdown

```
Synthetic data generation  (Claude Sonnet, Phase 1C):   $5.44  (code-estimated, not invoice)
LLM-as-Judge evaluation    (Claude Sonnet, Phase 3B):   $2.18  (code-estimated, not invoice)
Training and inference     (local RTX 4060 GPU):        $0.00
-------------------------------------------------------------
Total API spend:  ~$7.62 (estimated from token usage via the project's estimate_cost() function; not reconciled against invoice — a single shared API key made per-project separation impossible)
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Base Model | Llama-3.2-3B-Instruct (via Unsloth mirror) |
| Fine-Tuning | PyTorch, Transformers, PEFT, TRL |
| Quantization | bitsandbytes (NF4 4-bit) |
| RAG | ChromaDB, sentence-transformers (all-MiniLM-L6-v2) |
| LLM Judge | Anthropic API (Claude Sonnet) |
| Statistics | scipy, numpy, pandas, scikit-learn |
| Visualization | matplotlib, seaborn |
| Experiment Tracking | Weights and Biases |

---

## Experiment Tracking

**This is importing already-completed results into MLflow, not running new experiments.** `scripts/log_mlflow_runs.py` reads the existing result files (`results/phase3c_statistical_results.json`, `results/phase3d_error_analysis.json`, `results/trainable_params.json`) and logs one MLflow run per method (QLoRA, DoRA, IA3, RAG) so they can be browsed side by side. No training or inference happens when you run this script.

Runs were imported from completed experiments (dates: `results/phase3c_statistical_results.json` and `results/phase3d_error_analysis.json` both dated 2026-03-08, per file timestamps; `results/trainable_params.json` was computed later, on 2026-07-04, by counting parameters directly from the saved adapter weights), not logged live. The MLflow run start/end time for each method is backdated to 2026-03-08 to match, rather than showing the date the import script happened to be run -- each run is also tagged `run_type=imported_from_completed_experiment` so this is unambiguous inside the MLflow UI itself.

The 4 runs below were imported from those completed offline results, not logged live -- values read exactly from `mlflow_comparison.csv` (the exported MLflow comparison view):

| Method | Accuracy | Judge Score | Clarity | Risk Bias | Trainable Params |
|--------|----------|-------------|---------|-----------|-------------------|
| QLoRA  | 0.4579   | 6.6743      | 8.0374  | -0.4206   | 24313856          |
| DoRA   | 0.4486   | 6.6888      | 8.0935  | -0.3738   | 25088000          |
| IA3    | 0.4766   | 6.4467      | 7.8411  | -0.2736   | 286720            |
| RAG    | 0.5234   | 6.4888      | 7.4486  | -0.0891   | 0                 |

```bash
pip install mlflow   # local/dev only -- not part of the deployed app (see requirements-deploy.txt)
python scripts/log_mlflow_runs.py
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Then open http://127.0.0.1:5000 to compare `accuracy`, `judge_overall_score`, `clarity`, and `risk_bias_mean_bias` across the 4 methods, with `trainable_params` as a param and the three results JSON files attached as artifacts on every run.

---

## Monitoring

![Grafana dashboard](docs/grafana_dashboard.png)

The live HF Space is instrumented with Prometheus metrics, collected via Grafana Alloy and visualized in Grafana Cloud — request rate by endpoint, risk-level distribution, average classify latency (~0.9s), and average Groq API latency (~0.6s, isolated from retrieval/processing overhead).

---

## Limitations

**The LLM judge was not validated against human labels.** All quality scores come from Claude Sonnet acting as judge; there is no human-vs-judge agreement measurement in this project. A 30-example set (`data/human_anchor_30.jsonl`) was reserved for exactly this purpose, but no script in the repo scores it against the judge, so no correlation was computed. Judge reliability therefore rests only on the model's own internal consistency, not on any comparison to expert human assessment. Because Claude Sonnet also generated the training data, there is additionally a potential self-preference bias in the evaluation. Independent (ideally human) validation of the judge is required before treating these scores as ground truth.

**Underpowered test set.** At n=107 the study cannot statistically distinguish methods that are this close (Kruskal–Wallis p=0.62; no pairwise comparison survives Bonferroni correction). Reported differences are descriptive, not proven.

---

## Future Work

1. **Hybrid architecture:** Route each query through the fine-tuned model for output structure, then apply RAG-based retrieval to calibrate the risk level prediction. Combines fine-tuning's formatting consistency with RAG's calibration accuracy -- no additional training required.

2. **Larger test set (n=500+):** Adequate statistical power to confirm all observed effects. The current n=107 is underpowered for most pairwise comparisons. A stratified test set with balanced Critical-risk examples would also address the blindspot analysis.

3. **Risk calibration layer:** Train a lightweight post-hoc classifier on the residual error patterns from Phase 3D. Fine-tuned models systematically under-predict risk; the calibration layer would learn to correct this bias without retraining the base adapter.

---

*Portfolio project comparing PEFT fine-tuning and RAG on a realistic legal NLP task. All evaluation scripts are reproducible from the included code. Adapter weights available on request.*
