"""
Tests for the FastAPI app (app/main.py).

No real Groq API or model is called: the RAGPipeline is patched with a fake whose
generate() returns a fixed valid result dict. Because the patch replaces
`app.main.RAGPipeline` BEFORE the app's startup (lifespan) builds the pipeline, the
real pipeline -- and the `groq` import inside its __init__ -- never runs.

Run:
    pytest tests/ -v
"""

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Make the project root importable so `import app.main` and `from src...` resolve.
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test-only shim: importing app.main pulls in the RAG stack, whose top-level imports
# touch sentence_transformers / chromadb. We fully mock RAGPipeline, so those heavy
# libs are never actually used -- stub them ONLY if they aren't installed, so this
# web-layer test runs without the full ML environment. (No-op when they ARE present.)
_STUBS = {
    "sentence_transformers": {"SentenceTransformer": object},
    "chromadb": {},
    "chromadb.config": {"Settings": object},
}
for _name, _attrs in _STUBS.items():
    try:
        importlib.import_module(_name)
    except Exception:
        _mod = types.ModuleType(_name)
        for _a, _v in _attrs.items():
            setattr(_mod, _a, _v)
        sys.modules[_name] = _mod

import app.main as app_main  # noqa: E402  (after sys.path + stubs)


# Fixed result shaped exactly like RAGPipeline.generate() returns -- the risk
# assessment lives under "output".
FIXED_RESULT = {
    "output": {
        "clause_type": "termination",
        "risk_level": "Medium",
        "risk_score": 0.45,
        "key_concerns": [
            "30-day notice may be too short for transition",
            "No cause required for termination",
            "Potential for abrupt service disruption",
        ],
        "recommendation": "Negotiate a 90-day notice period and a transition-assistance clause.",
        "confidence": 0.82,
    },
    "raw_text": '{"risk_level": "Medium", ...}',
    "retrieved_examples": [],
    "n_retrieved": 5,
    "input_tokens": 321,
    "latency_ms": 42.0,
    "json_parse_success": True,
}


class _FakePipeline:
    """Stand-in for RAGPipeline: construction is a no-op, generate() is deterministic."""

    def __init__(self, *args, **kwargs):
        # No model load, no Groq client, no network.
        pass

    def generate(self, clause_text, clause_type, **kwargs):
        return FIXED_RESULT


@pytest.fixture
def client():
    """
    Patch RAGPipeline with the fake, then start the app via TestClient as a context
    manager so the lifespan (startup) builds the *fake* pipeline. Yields a live client.
    """
    with patch.object(app_main, "RAGPipeline", _FakePipeline):
        with TestClient(app_main.app) as c:
            yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_classify_valid_returns_expected_keys(client):
    resp = client.post(
        "/classify",
        json={
            "clause_text": "Either party may terminate this Agreement for convenience on 30 days' notice.",
            "clause_type": "termination",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    # The risk assessment is under "output".
    assert "output" in data
    out = data["output"]
    for key in ("risk_level", "risk_score", "key_concerns", "recommendation"):
        assert key in out, f"missing key: {key}"

    # Values come from our fake (proves the mocked pipeline was used, not a real call).
    assert out["risk_level"] == "Medium"
    assert isinstance(out["key_concerns"], list) and len(out["key_concerns"]) == 3
    assert data["json_parse_success"] is True


def test_classify_empty_clause_text_is_422(client):
    resp = client.post(
        "/classify",
        json={"clause_text": "", "clause_type": "termination"},
    )
    assert resp.status_code == 422  # Pydantic min_length=1 validation


def test_classify_invalid_clause_type_is_422(client):
    resp = client.post(
        "/classify",
        json={"clause_text": "Some valid clause text.", "clause_type": "not_a_real_type"},
    )
    assert resp.status_code == 422  # Literal[...] type validation
