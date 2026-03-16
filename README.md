Benchmarking QLoRA, DoRA, IA³ vs RAG on Llama-3.2-3B for legal risk analysis. 4-layer evaluation framework

---

## TL;DR

- **What:** Fine-tuned Llama-3.2-3B-Instruct using 3 PEFT methods (QLoRA, DoRA, IA3) and built a RAG baseline. Each method reads a contract clause and outputs a structured JSON risk assessment (risk level, score, concerns, recommendation).
- **Data:** 855 training / 107 test examples across 7 clause types (Termination, Liability Cap, IP Ownership, Indemnification, Governing Law, Confidentiality, Auto-Renewal). Built from CUAD contracts + Claude Sonnet synthetic augmentation.
- **Honest finding:** No method achieves statistical significance over others at n=107 (Friedman p=0.17). Fine-tuning and RAG are complementary -- fine-tuning wins clarity/structure, RAG wins risk calibration.
- **Critical limitation:** All methods miss ~86% of Critical-risk clauses. This is a safety concern that must be addressed before any production deployment.
- **Takeaway:** At 3B scale with ~855 examples, PEFT methods are statistically indistinguishable overall but differ meaningfully on specific dimensions.

---
## Architecture

```
CUAD Dataset (510 real contracts)
        |
        v
[Phase 1B] Clause Span Extraction
        -> 5,847 clause spans across 7 types
        |
        v
[Phase 1C] Synthetic Augmentation via Claude Sonnet
        -> 855 train / 108 val / 107 test structured risk assessments
        |
        +---------------------------------------------+
        v                                             v
[Phase 2] PEFT Fine-Tuning                  [Phase 3A] RAG Baseline
  QLoRA / DoRA / IA3                          ChromaDB + sentence-transformers
  4-bit NF4, RTX 4060 8GB                     855 training clauses embedded
  r=16 LoRA rank for QLoRA/DoRA               5-shot retrieval per query
  IA3 rescales activations (0.03M params)     Base Llama-3.2-3B (no adapter)
        |                                             |
        +--------------------+------------------------+
                             v
              [Phase 3B] LLM-as-Judge
                Claude Sonnet scores each prediction
                5 dimensions x 107 test x 4 methods = 428 evaluations
                             |
                             v
              [Phase 3C] Statistical Significance Testing
                Bootstrap CIs (10K iterations) . Wilcoxon signed-rank
                Friedman . McNemar's . Bonferroni correction . Power analysis
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

**Training:** Three PEFT adapters trained on Llama-3.2-3B-Instruct with 4-bit NF4 quantization and gradient checkpointing to fit 8GB VRAM. QLoRA and DoRA use LoRA with r=16; IA3 rescales attention keys/values and FFN activations using 100x fewer trainable parameters (0.03M vs 2.6M).

**RAG baseline:** ChromaDB vector store with all 855 training clauses embedded via all-MiniLM-L6-v2 (384-dim). At inference, retrieves the 5 most similar clauses by clause type and passes them as few-shot context to the base model (no adapter loaded).

**Evaluation:** 4-layer stack -- deterministic metrics -> LLM-as-Judge -> statistical testing -> error analysis. Each layer reveals something the previous one misses.

---

## Results

| Method | Accuracy | MAE   | Judge Score | Clarity | Risk Bias | Trainable Params |
|--------|----------|-------|-------------|---------|-----------|------------------|
| QLoRA  | 45.8%    | 0.152 | 6.67 / 10   | 8.04    | -0.090    | 2.6M             |
| DoRA   | 44.9%    | 0.156 | 6.69 / 10   | 8.09    | -0.084    | 2.6M             |
| IA3    | 47.7%    | 0.161 | 6.45 / 10   | 7.84    | -0.029    | 0.03M            |
| RAG    | 50.5%    | 0.124 | 6.49 / 10   | 7.45    | -0.002    | 0 (retrieval)    |

*Accuracy = exact risk level match. MAE = mean absolute error on 0-1 normalized risk scores. Risk Bias = mean(predicted_rank - true_rank); negative = systematic under-prediction of severity. No pairwise comparison survived Bonferroni correction for the overall judge score.*

---

## Key Findings

**DoRA equals QLoRA at 3B scale with 855 examples.** DoRA's theoretical advantage -- decomposing LoRA weights into independent direction and magnitude components -- provides no measurable benefit here. The judge score gap is 0.015 points (6.69 vs 6.67). To confirm this difference statistically, you would need roughly n=29,000 test examples, not 107. This is a practical data point for practitioners: weight decomposition may not be worth the implementation complexity on small fine-tuning datasets.

**Fine-tuning and RAG are complementary, not competing.** Fine-tuning teaches the model how to format a legal analysis -- trained on 855 schema-consistent examples, it produces clean, well-structured JSON responses (clarity 8.04-8.09 vs 7.45 for RAG). RAG provides factual calibration -- it retrieves real examples before answering, which largely eliminates the systematic under-prediction bias that fine-tuned models exhibit (bias -0.002 vs -0.084 to -0.090). The ideal production system would combine both: a fine-tuned model for output structure with retrieval-augmented calibration for risk level accuracy.

**IA3 hits a capacity ceiling.** With 100x fewer trainable parameters (0.03M vs 2.6M), IA3 scores measurably lower on completeness and legal reasoning. The Wilcoxon signed-rank test for IA3 vs QLoRA on legal_reasoning survives Bonferroni correction (p=0.002). The IA3 mechanism -- scaling activations with learned vectors rather than adding low-rank weight deltas -- cannot fully reshape the model's output behavior when the legal domain requires genuine content changes, not just style shifts.

**The critical risk blindspot is a deployment blocker.** Six out of seven Critical-risk clauses in the test set are misclassified by every method -- the models default to predicting High or Medium. Under-prediction is the more dangerous failure mode in legal contexts: a lawyer who follows a "Medium" risk recommendation on a Critical clause could expose their client to catastrophic liability. Any production deployment needs a calibration layer, human review for High/Critical outputs, or a specialized classifier for edge-case detection.

**Statistical power matters more than most practitioners realize.** n=107 is underpowered for the comparisons being made. To confirm the observed DoRA vs QLoRA difference, we would need ~29,000 examples. To confirm DoRA vs IA3, ~237 examples. "Not significant" does not mean "no difference" -- it means the test set is not large enough to distinguish methods this close with confidence. Two effects do survive Bonferroni correction: clarity differences between QLoRA and RAG (p=0.005), and legal reasoning differences between IA3 and QLoRA (p=0.002).

---

## Evaluation Methodology

This project uses a 4-layer evaluation stack where each layer adds something the previous one cannot provide.

**Layer 1 -- Deterministic Metrics.** Accuracy (exact risk level match), MAE on 0-1 normalized risk scores, JSON parse success rate, and per-class F1. Fully reproducible and zero cost. Catches gross failures but cannot distinguish between a wrong answer that was close in reasoning vs one that was entirely wrong.

**Layer 2 -- LLM-as-Judge.** Claude Sonnet scores each prediction on 5 dimensions (accuracy, completeness, legal reasoning, clarity, actionability) on a 1-10 scale. 428 total evaluations. The overall score is recomputed from dimension scores with fixed weights rather than trusting the model's own arithmetic. This reveals qualitative differences that accuracy alone misses -- specifically, that fine-tuning produces better-formatted responses even when the risk level classification is wrong.

**Layer 3 -- Statistical Significance Testing.** Bootstrap CIs (10,000 iterations) for uncertainty quantification. Wilcoxon signed-rank for pairwise judge score comparisons (non-parametric, no normality assumption required). Friedman test for global significance across all methods simultaneously. McNemar's test for pairwise accuracy differences on paired examples. Bonferroni correction applied to all 6 pairwise comparisons. Power analysis to quantify the minimum sample size needed to confirm each observed effect.

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
# Phase 1B: Extract clause spans from CUAD
python scripts/extract_cuad.py

# Phase 1C: Generate synthetic risk assessments (~$5.44)
python scripts/generate_synthetic.py

# Phase 2: Train PEFT adapters (requires GPU)
python scripts/train.py --method qlora
python scripts/train.py --method dora
python scripts/train.py --method ia3

# Phase 3A: RAG baseline inference
python scripts/inference_rag.py

# Phase 3B: LLM-as-Judge evaluation (~$2.18)
python scripts/run_judge.py

# Phase 3C: Statistical significance testing
python outputs/phase3c_statistical_tests.py

# Phase 3D: Error analysis
python outputs/phase3d_error_analysis.py

# Phase 4: Generate visualizations
python outputs/phase4_visualizations.py
```

Note: Adapter weights are gitignored due to size (~500MB each). Phases 3C onward can be reproduced without training using the pre-generated JSONL files in evaluation/.

---

## Cost Breakdown

```
Synthetic data generation  (Claude Sonnet, Phase 1C):   $5.44
LLM-as-Judge evaluation    (Claude Sonnet, Phase 3B):   $2.18
Training and inference     (local RTX 4060 GPU):        $0.00
-------------------------------------------------------------
Total API spend:                                         $7.62
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

## Future Work

1. **Hybrid architecture:** Route each query through the fine-tuned model for output structure, then apply RAG-based retrieval to calibrate the risk level prediction. Combines fine-tuning's formatting consistency with RAG's calibration accuracy -- no additional training required.

2. **Larger test set (n=500+):** Adequate statistical power to confirm all observed effects. The current n=107 is underpowered for most pairwise comparisons. A stratified test set with balanced Critical-risk examples would also address the blindspot analysis.

3. **Risk calibration layer:** Train a lightweight post-hoc classifier on the residual error patterns from Phase 3D. Fine-tuned models systematically under-predict risk; the calibration layer would learn to correct this bias without retraining the base adapter.

---

*Portfolio project comparing PEFT fine-tuning and RAG on a realistic legal NLP task. All evaluation scripts are reproducible from the included code. Adapter weights available on request.*

