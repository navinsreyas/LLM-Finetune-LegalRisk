"""
Import completed LegalRisk-LLM results into MLflow for side-by-side comparison.

*** THIS IS AN IMPORT, NOT A NEW EXPERIMENT. ***
No training or inference runs here. This script reads the already-computed result
files (results/phase3c_statistical_results.json, results/phase3d_error_analysis.json,
results/trainable_params.json) and logs one MLflow run per method (qlora, dora, ia3,
rag) so they can be browsed/compared in the MLflow UI.

Each run's start/end time is backdated to the real mtime of the source result files
(not "now") and tagged run_type=imported_from_completed_experiment, so the MLflow UI
itself does not misrepresent these as live runs.

Usage:
    pip install mlflow   # not in requirements-deploy.txt -- local/dev only
    python scripts/log_mlflow_runs.py
    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# Add project root to path (consistent with the other scripts/ entrypoints)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

METHODS = ["qlora", "dora", "ia3", "rag"]

PHASE3C_PATH = PROJECT_ROOT / "results" / "phase3c_statistical_results.json"
PHASE3D_PATH = PROJECT_ROOT / "results" / "phase3d_error_analysis.json"
TRAINABLE_PARAMS_PATH = PROJECT_ROOT / "results" / "trainable_params.json"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
TRACKING_URI = f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').as_posix()}"
ARTIFACT_LOCATION = (MLRUNS_DIR / "artifacts").resolve().as_uri()
EXPERIMENT_NAME = "legalrisk-llm-comparison"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required results file not found: {path}\n"
            f"Run Phase 3C/3D (outputs/phase3c_statistical_tests.py, "
            f"outputs/phase3d_error_analysis.py) first, or use the pre-generated "
            f"JSONL/JSON files already committed to this repo."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_trainable_params(trainable_params_data: dict, method: str) -> int:
    """RAG has no trained adapter (retrieval only) -- 0 trainable params."""
    if method == "rag":
        return 0
    return trainable_params_data["methods"][method]["trainable_params"]


def get_source_date(*paths: Path) -> datetime:
    """
    Real completion date of the imported experiment, taken from the source result
    files' mtimes (NOT invented, NOT "now"). Used to backdate the MLflow run so the
    UI shows when the comparison actually finished, not when this script was run.
    """
    latest = max(p.stat().st_mtime for p in paths)
    return datetime.fromtimestamp(latest, tz=timezone.utc)


def setup_experiment(client: MlflowClient) -> str:
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    (MLRUNS_DIR / "artifacts").mkdir(parents=True, exist_ok=True)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION
        )
        print(f"[MLFLOW] Created experiment '{EXPERIMENT_NAME}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"[MLFLOW] Using existing experiment '{EXPERIMENT_NAME}' (id={experiment_id})")

    return experiment_id


def log_method_run(
    client: MlflowClient,
    experiment_id: str,
    method: str,
    phase3c: dict,
    phase3d: dict,
    trainable_params_data: dict,
    source_date: datetime,
):
    """One MLflow run per method, backdated to the real result-file date."""
    start_time_ms = int(source_date.timestamp() * 1000)

    run = client.create_run(
        experiment_id,
        run_name=method,
        start_time=start_time_ms,
        tags={
            "run_type": "imported_from_completed_experiment",
            "mlflow.note.content": (
                "Imported from completed offline results, not a live training/"
                "inference run. See results/phase3c_statistical_results.json and "
                "results/phase3d_error_analysis.json for the source computation."
            ),
        },
    )
    run_id = run.info.run_id

    # --- Params ---
    client.log_param(run_id, "method", method)
    client.log_param(
        run_id, "trainable_params", get_trainable_params(trainable_params_data, method)
    )

    # --- Metrics (real values, read directly from the results JSON) ---
    accuracy = phase3d["confusion_matrices"][method]["accuracy"]
    judge_overall = phase3c["descriptive_stats"][method]["overall"]["mean"]
    clarity = phase3c["descriptive_stats"][method]["clarity"]["mean"]
    risk_bias = phase3d["risk_bias"][method]["mean_bias"]

    client.log_metric(run_id, "accuracy", accuracy)
    client.log_metric(run_id, "judge_overall_score", judge_overall)
    client.log_metric(run_id, "clarity", clarity)
    client.log_metric(run_id, "risk_bias_mean_bias", risk_bias)

    # --- Artifacts: the results JSON files themselves ---
    for path in (PHASE3C_PATH, PHASE3D_PATH, TRAINABLE_PARAMS_PATH):
        client.log_artifact(run_id, str(path))

    client.set_terminated(run_id, status="FINISHED", end_time=start_time_ms)

    print(
        f"[MLFLOW] Logged {method}: accuracy={accuracy}, judge_overall={judge_overall}, "
        f"clarity={clarity}, risk_bias={risk_bias} (run_id={run_id})"
    )


def main():
    print("=" * 70)
    print("  Importing completed LegalRisk-LLM results into MLflow")
    print("  (This logs pre-computed results -- no training/inference runs here)")
    print("=" * 70)

    phase3c = load_json(PHASE3C_PATH)
    phase3d = load_json(PHASE3D_PATH)
    trainable_params_data = load_json(TRAINABLE_PARAMS_PATH)

    source_date = get_source_date(PHASE3C_PATH, PHASE3D_PATH)
    print(f"[MLFLOW] Source results date (from file mtime): {source_date.date()}")

    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"[MLFLOW] Tracking URI: {TRACKING_URI}")

    client = MlflowClient()
    experiment_id = setup_experiment(client)

    for method in METHODS:
        log_method_run(
            client, experiment_id, method, phase3c, phase3d, trainable_params_data, source_date
        )

    print("\n" + "=" * 70)
    print("  Done. View the comparison with:")
    print(f"    mlflow ui --backend-store-uri {TRACKING_URI}")
    print("=" * 70)


if __name__ == "__main__":
    main()
