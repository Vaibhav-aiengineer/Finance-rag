"""
retrieval.py
------------
Phase 2: the hybrid document retriever that the `retrieve_docs` graph node
calls. This is the most feature-dense part of the project.

PIPELINE (all inside one logical retrieve() call):
  1. DENSE   -- embed the question (bge-small), search Qdrant. RBAC role filter
                applied HERE, inside Qdrant (efficient: disallowed chunks are
                never even scored).
  2. BM25    -- keyword search over an in-memory BM25 index built from the same
                chunks. BM25 knows nothing about Qdrant's filter, so RBAC is
                RE-APPLIED here as a post-filter. Dual-layer RBAC = defense in
                depth: a chunk must pass BOTH paths to survive.
  3. MERGE   -- union the two candidate sets, dedup on chunk_id.
  4. RERANK  -- cross-encoder (ms-marco-MiniLM-L-6-v2) scores each candidate
                against the question. Dense & BM25 scores aren't comparable;
                the reranker re-scores everything on one consistent scale.
  5. VERSION -- soft-demote SUPERSEDED chunks: subtract a penalty from their
                rerank score so the CURRENT version wins for "what's the current
                rule" questions, while SUPERSEDED stays reachable for "what was
                the old rule" questions (not hard-excluded).

WHY A SEPARATE MODULE (not inline in the graph node):
The graph node should stay a thin wrapper that reads/writes state. The actual
retrieval logic lives here so it's independently testable -- you can call
retrieve() directly in a script without spinning up the whole graph.
"""

import json
import os
from typing import List, Dict, Any

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from rank_bm25 import BM25Okapi

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(HERE, "..", "data", "chunks", "chunks.json")
QDRANT_PATH = os.path.join(HERE, "..", "data", "qdrant_store")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
COLLECTION = "yardi_sops"

DENSE_TOP_K = 15           # candidates pulled from dense search
BM25_TOP_K = 15            # candidates pulled from keyword search
FINAL_TOP_K = 5            # chunks returned to the answer generator
SUPERSEDED_PENALTY = 2.0   # subtracted from a SUPERSEDED chunk's rerank score


class HybridRetriever:
    """Loads everything once (models, Qdrant client, BM25 index) and exposes
    retrieve(). Instantiated a single time and reused across queries -- loading
    the models per query would be far too slow."""

    def __init__(self):
        # chunks are loaded so we can build the BM25 index and map ids->payload
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks: List[Dict[str, Any]] = json.load(f)

        # quick lookup from chunk_id -> chunk dict
        self.by_id = {c["chunk_id"]: c for c in self.chunks}

        # models (loaded once)
        self.embedder = TextEmbedding(model_name=EMBED_MODEL)
        self.reranker = TextCrossEncoder(model_name=RERANK_MODEL)

        # qdrant (the persisted store from embedder.py)
        self.client = QdrantClient(path=QDRANT_PATH)

        # build the in-memory BM25 index over chunk texts.
        # tokenization here is deliberately simple (lowercase split); BM25 is a
        # keyword method, so simple whitespace tokenization is fine and fast.
        self.bm25_corpus_ids = [c["chunk_id"] for c in self.chunks]
        tokenized = [self._tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def close(self):
        """Explicitly close the Qdrant client to avoid a noisy (harmless)
        traceback from its destructor firing during interpreter shutdown."""
        try:
            self.client.close()
        except Exception:
            pass

    # ---- step 1: dense, with RBAC filter inside Qdrant ----
    def _dense_search(self, question: str, user_role: str) -> List[str]:
        q_vec = list(self.embedder.embed([question]))[0]
        role_filter = Filter(
            must=[FieldCondition(key="access_roles", match=MatchAny(any=[user_role]))]
        )
        result = self.client.query_points(
            collection_name=COLLECTION,
            query=q_vec.tolist(),
            query_filter=role_filter,   # RBAC enforced in the vector search
            limit=DENSE_TOP_K,
        )
        return [h.payload["chunk_id"] for h in result.points]

    # ---- step 2: BM25, with RBAC re-applied as a post-filter ----
    def _bm25_search(self, question: str, user_role: str) -> List[str]:
        scores = self.bm25.get_scores(self._tokenize(question))
        ranked = sorted(zip(self.bm25_corpus_ids, scores), key=lambda x: x[1], reverse=True)
        out = []
        for chunk_id, score in ranked:
            if score <= 0:
                continue
            chunk = self.by_id[chunk_id]
            # RBAC post-filter: BM25 index doesn't know about Qdrant's filter,
            # so we enforce role membership again here.
            if user_role in chunk.get("access_roles", []):
                out.append(chunk_id)
            if len(out) >= BM25_TOP_K:
                break
        return out

    # ---- steps 3-5: merge, rerank, version soft-demote ----
    def retrieve(self, question: str, user_role: str) -> List[Dict[str, Any]]:
        dense_ids = self._dense_search(question, user_role)
        bm25_ids = self._bm25_search(question, user_role)

        # merge + dedup (preserve first-seen order, dense first)
        candidate_ids = list(dict.fromkeys(dense_ids + bm25_ids))
        if not candidate_ids:
            return []

        candidates = [self.by_id[cid] for cid in candidate_ids]

        # rerank: cross-encoder scores question vs each candidate text
        rerank_scores = list(self.reranker.rerank(question, [c["text"] for c in candidates]))

        scored = []
        for chunk, score in zip(candidates, rerank_scores):
            adjusted = score
            # version soft-demote: SUPERSEDED chunks get a penalty so the
            # CURRENT version outranks them, but they remain in the list.
            if chunk.get("status") == "SUPERSEDED":
                adjusted -= SUPERSEDED_PENALTY
            scored.append((chunk, score, adjusted))

        # sort by adjusted score, return the top-k (with both scores for tracing)
        scored.sort(key=lambda x: x[2], reverse=True)
        results = []
        for chunk, raw, adjusted in scored[:FINAL_TOP_K]:
            results.append({
                **chunk,
                "rerank_score": float(raw),
                "adjusted_score": float(adjusted),
            })
        return results


# module-level singleton so the graph node reuses one loaded instance
_retriever = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


if __name__ == "__main__":
    # standalone test of the retriever, independent of the graph
    r = get_retriever()
    tests = [
        ("What is the current approval threshold for a journal entry over 40000?", "staff_accountant"),
        ("How are vendor invoices approved?", "ap_clerk"),
        ("What is the journal entry approval threshold?", "ap_clerk"),
    ]
    for q, role in tests:
        print(f"\nQ ({role}): {q}")
        for hit in r.retrieve(q, role):
            flag = "SUPERSEDED" if hit.get("status") == "SUPERSEDED" else "current"
            print(f"  [{hit['doc_id']:12s} {flag:10s}] adj={hit['adjusted_score']:.3f} "
                  f"raw={hit['rerank_score']:.3f}  {hit['text'][:70]}")
    r.close()