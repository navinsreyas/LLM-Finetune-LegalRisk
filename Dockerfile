# ============================================================
# LegalRisk-LLM -- Hugging Face Spaces (Docker SDK), Groq mode
# ============================================================
# Small CPU-only image: the local Llama model is NEVER loaded (LLM_PROVIDER=groq),
# so no CUDA torch, no transformers, no bitsandbytes/unsloth. Retrieval still embeds
# the query clause locally (sentence-transformers on CPU), so a CPU build of torch is
# installed first (a few hundred MB) instead of the multi-GB CUDA build.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_PROVIDER=groq \
    RAG_DEVICE=cpu \
    PORT=7860 \
    HF_HOME=/tmp/hf \
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf

WORKDIR /app

# 1) Install a CPU-only torch FIRST so sentence-transformers reuses it instead of
#    pulling the default (CUDA) build. Then install the slim deploy requirements
#    (NO torch/transformers/bitsandbytes/unsloth in that file).
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-deploy.txt

# 2) Pre-download the embedding model so cold starts don't hit the network.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# 3) App code + the PREBUILT ChromaDB index. The index is gitignored in the repo,
#    but it MUST be present in the image (RAG retrieval needs it). For Hugging Face
#    Spaces you must also force-add it to the Space repo -- see the note at the bottom.
COPY app/ ./app/
COPY src/ ./src/
COPY data/rag/chroma_db/ ./data/rag/chroma_db/

EXPOSE 7860

# Honors $PORT (HF Spaces), defaults to 7860.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

# ------------------------------------------------------------
# Hugging Face Spaces notes:
#  - Set GROQ_API_KEY as a Space *secret* (Settings -> Variables and secrets).
#  - LLM_PROVIDER=groq and RAG_DEVICE=cpu are already set above; override GROQ_MODEL
#    via a Space variable if you want a model other than llama-3.1-8b-instant.
#  - data/rag/chroma_db is gitignored: run `git add -f data/rag/chroma_db` in the
#    Space repo (or add a .gitignore negation) so it is pushed and available to build.
# ------------------------------------------------------------
