"""
FastAPI web layer around the existing RAG pipeline.

This is ONLY a web wrapper -- it does not change any pipeline logic. It builds a single
RAGPipeline at startup and reuses it for every request.

Run locally (from the project root):
    uvicorn app.main:app --reload
"""

# Load .env FIRST -- before importing rag_pipeline or anything that reads env vars,
# so LLM_PROVIDER / GROQ_API_KEY / GROQ_MODEL are picked up without a manual `set`
# (identical to scripts/run_rag_inference.py).
from dotenv import load_dotenv
load_dotenv()

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Observability:
#   - Instrumentator auto-records per-request metrics (count, latency, status) and
#     exposes them at GET /metrics in Prometheus text format.
#   - Counter is a raw prometheus_client metric we drive ourselves for the one custom
#     signal below (successful classifications per risk level).
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

# Make the project root importable so `from src.rag...` works under `uvicorn app.main:app`.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.rag_pipeline import RAGPipeline  # noqa: E402  (after sys.path + load_dotenv)

STATIC_DIR = Path(__file__).parent / "static"

# The 7 clause types the pipeline supports (matches src/data/quality_filter.py VALID_CLAUSE_TYPES).
CLAUSE_TYPES = (
    "termination",
    "liability",
    "non_compete",
    "ip",
    "governing_law",
    "confidentiality",
    "indemnification",
)

# Single shared pipeline instance, built once at startup.
_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build ONE RAGPipeline at startup and reuse it for all requests."""
    global _pipeline
    print("[APP] Building RAGPipeline (one instance, reused per request)...")
    _pipeline = RAGPipeline(
        # Fall back to the pipeline's own defaults; overridable via env for deployment.
        persist_dir=os.getenv("RAG_PERSIST_DIR", "data/rag/chroma_db"),
        n_examples=int(os.getenv("RAG_N_EXAMPLES", "5")),
        device=os.getenv("RAG_DEVICE", "cuda"),
    )
    print("[APP] RAGPipeline ready.")
    yield
    # No explicit teardown needed.


app = FastAPI(title="LegalRisk-LLM RAG API", lifespan=lifespan)

# Attach the Prometheus instrumentator:
#   .instrument(app) adds middleware that times every request and records the standard
#     HTTP metrics (request count, latency histogram, in-progress gauge) labeled by
#     method / path / status code.
#   .expose(app) registers the GET /metrics endpoint that serves those metrics (plus any
#     custom metrics, like the counter below) in Prometheus text format.
Instrumentator().instrument(app).expose(app)

# ONE custom metric: how many /classify calls succeeded, broken down by predicted
# risk level. `risk_level` is a low-cardinality label (Low/Medium/High/Critical, plus
# "unknown" when the model didn't return a parseable level), which is safe for Prometheus.
CLASSIFY_RISK_LEVEL = Counter(
    "classify_risk_level_total",
    "Successful /classify responses, labeled by the predicted risk_level.",
    ["risk_level"],
)


class ClassifyRequest(BaseModel):
    """Request body for POST /classify, with validation."""
    clause_text: str = Field(..., min_length=1, description="The contract clause to analyze")
    clause_type: Literal[
        "termination",
        "liability",
        "non_compete",
        "ip",
        "governing_law",
        "confidentiality",
        "indemnification",
    ] = Field(..., description="One of the 7 supported clause types")


@app.get("/", include_in_schema=False)
def index():
    """Serve the demo page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    """Deployment health check."""
    return {"status": "ok"}


@app.post("/classify")
def classify(req: ClassifyRequest):
    """
    Run the RAG pipeline on one clause and return its full result dict
    (output = risk assessment JSON, plus raw_text, latency_ms, json_parse_success, etc.).
    """
    if _pipeline is None:
        # Startup failed or not finished -- don't crash, report clearly.
        raise HTTPException(status_code=503, detail="Pipeline not ready. Check server startup logs.")

    try:
        result = _pipeline.generate(
            clause_text=req.clause_text,
            clause_type=req.clause_type,
        )
    except Exception as e:
        # Any pipeline failure -> HTTP 500 with a clear message; server keeps running.
        raise HTTPException(status_code=500, detail=f"Pipeline error: {type(e).__name__}: {e}")

    # Observability: count this successful classification by its predicted risk level.
    # Safe extraction so metrics never affect the response: if output is None (parse
    # failure) or has no risk_level, label it "unknown".
    output = result.get("output") if isinstance(result, dict) else None
    risk_level = (output or {}).get("risk_level") or "unknown"
    CLASSIFY_RISK_LEVEL.labels(risk_level=str(risk_level)).inc()

    return result


if __name__ == "__main__":
    # Allow `python app/main.py` too. Reads PORT from env (Hugging Face Spaces default
    # is 7860); falls back to 7860 if unset.
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
