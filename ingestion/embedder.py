"""
embedder.py
-----------
Phase 3, Step A: embed every chunk with bge-small-en-v1.5 (via fastembed) and
load the vectors + metadata into a local Qdrant collection.

WHY fastembed (not sentence-transformers directly):
fastembed runs bge-small through ONNX runtime, so it does NOT need a separate
torch inference path. On Windows this sidesteps the exact torch/DLL class of
problems we just fought through during parsing. Same embedding model
(bge-small-en-v1.5, 384-dim), lighter runtime.

WHY bge-small-en-v1.5 (recap):
384-dim, retrieval-tuned, runs fast on CPU, open weights. Right tradeoff for a
local project. The SAME model must embed both the chunks (here) and the user's
question (at query time) -- vectors from two different models aren't comparable.

WHAT GOES IN THE QDRANT PAYLOAD (the important part):
Each vector carries its full metadata as payload: text, doc_id, access_roles,
status, superseded_by, version, category, chunk_type, section_title. This is
what lets retrieval later filter by role (RBAC) and prefer current-over-
superseded versions WITHOUT a second database lookup -- the metadata is already
sitting next to the vector. This is why we attached it at chunk time.

WHY LOCAL (on-disk) QDRANT:
qdrant-client can run an embedded, file-backed instance with no Docker/server
needed -- perfect for development. We point it at a local path; it persists to
disk so you don't re-embed every run. (Phase 6 deployment can swap this for a
Qdrant server by changing one line.)

Run with:
    python embedder.py
"""

import json
import os

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(HERE, "..", "data", "chunks", "chunks.json")
QDRANT_PATH = os.path.join(HERE, "..", "data", "qdrant_store")  # on-disk persistence

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_DIM = 384            # bge-small-en-v1.5 output dimension
COLLECTION = "yardi_sops"
DISTANCE = Distance.COSINE  # cosine: compares direction (meaning), not magnitude


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(chunks)} chunks. Embedding with {EMBED_MODEL} ...")
    print("(first run downloads the model -- a one-time wait)")

    # Embed all chunk texts. fastembed returns a generator of numpy vectors.
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    vectors = list(embedder.embed(texts))
    print(f"Generated {len(vectors)} vectors of dim {len(vectors[0])}.")

    # Connect to a local on-disk Qdrant (no server needed).
    client = QdrantClient(path=QDRANT_PATH)

    # Create the collection fresh each run so re-running is idempotent.
    # (Newer qdrant-client deprecates recreate_collection in favour of an
    # explicit exists-check + create.)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=DISTANCE),
    )

    # Build points: vector + full metadata payload. We use an integer id per
    # point but keep the human-readable chunk_id in the payload too.
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        payload = {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "doc_id": chunk["doc_id"],
            "title": chunk.get("title"),
            "category": chunk.get("category"),
            "chunk_type": chunk["chunk_type"],
            "section_title": chunk.get("section_title"),
            # --- RBAC ---
            "access_roles": chunk.get("access_roles", []),
            # --- version handling ---
            "status": chunk.get("status", "CURRENT"),
            "superseded_by": chunk.get("superseded_by"),
            "version": chunk.get("version"),
            "effective_date": chunk.get("effective_date"),
        }
        points.append(PointStruct(id=i, vector=vec.tolist(), payload=payload))

    client.upsert(collection_name=COLLECTION, points=points)

    count = client.count(collection_name=COLLECTION).count
    print(f"\nUpserted {count} points into Qdrant collection '{COLLECTION}'.")
    print(f"Persisted to {QDRANT_PATH}")

    # Quick smoke test: embed a question and see what comes back (no filtering
    # yet -- that's Phase 2). Just confirming the store works end to end.
    print("\nSmoke test query: 'who approves a large invoice?'")
    q_vec = list(embedder.embed(["who approves a large invoice?"]))[0]
    # Newer qdrant-client uses query_points (returns an object with .points)
    # instead of the older search() method.
    result = client.query_points(
        collection_name=COLLECTION,
        query=q_vec.tolist(),
        limit=3,
    )
    for h in result.points:
        print(f"  score={h.score:.3f}  [{h.payload['doc_id']} / {h.payload['chunk_type']}]  "
              f"{h.payload['text'][:80]}")


if __name__ == "__main__":
    main()