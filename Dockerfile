# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Yardi Finance Ops Assistant -- self-contained API image.
#
# WHY python:3.11-slim (not the default or 3.12+):
#   - 3.11 matches the version the app was built and tested on (avoids the
#     ML-package compatibility issues newer CPython sometimes has).
#   - "slim" = a smaller Debian base without build extras we don't need at
#     runtime, keeping the image leaner.
#
# WHY models are downloaded AT BUILD TIME (see the warm-up step):
#   The embedder/reranker weights (~100MB) would otherwise be fetched from
#   HuggingFace on first request. Baking them into the image means containers
#   start fast and work with no internet at runtime.
#
# WHY the data (qdrant_store + DB) is COPIED IN:
#   This is a self-contained demo image -- it runs anywhere with no external
#   services. In PRODUCTION, Qdrant and the SQL DB would be separate persistent
#   services and the app container would stay stateless (connect over network).
#   That change is a config swap (QdrantClient path-> url), not a rewrite.
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# system deps: build-essential is occasionally needed by wheels that compile;
# we remove apt lists afterward to keep the layer small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- dependency layer (cached unless requirements.txt changes) ---
# Copying requirements first, before the rest of the code, means Docker can
# REUSE this layer on rebuilds when only app code changed -- a big speedup.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- app code + data ---
COPY app/ ./app/
COPY data/ ./data/

# --- bake the models into the image at build time ---
# Instantiating the embedder and reranker forces fastembed to download and
# cache the ONNX weights now, during build, so runtime needs no network.
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
    TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); \
    TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2'); \
    print('models cached into image')"

# the API listens on 8000
EXPOSE 8000

# NOTE: ANTHROPIC_API_KEY is NOT baked in (never bake secrets into an image).
# It is passed at run time via -e ANTHROPIC_API_KEY=... or an env file.

# uvicorn serves the FastAPI app. No --reload in production images.
# --host 0.0.0.0 is required so the server is reachable from OUTSIDE the
# container (127.0.0.1 inside a container is only reachable from inside it).
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]