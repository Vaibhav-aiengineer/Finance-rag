"""
api.py
------
Phase 6, Step A: expose the LangGraph pipeline as an HTTP API.

THE CORE PATTERN -- LOAD ONCE, SERVE MANY:
The graph carries heavy state (embedder, reranker, Qdrant client) that takes
~20s to load. A server handles many requests over its lifetime, so we build the
graph EXACTLY ONCE at startup (in the lifespan handler) and reuse it for every
request. Reloading per request would make each call slow and blow up memory.

This also fixes the Qdrant shutdown traceback: the client now lives for the
whole server lifetime and is closed cleanly at shutdown, instead of being
garbage-collected during interpreter exit.

Run locally:
    uvicorn app.api:app --reload
Then POST to http://127.0.0.1:8000/ask  (or open /docs for the Swagger UI).
"""

from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.graph import build_graph

# holds the singletons built at startup
_state = {"graph": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup: build the graph once (loads models, connects Qdrant) ----
    print("Starting up: building graph and loading models (one-time)...")
    _state["graph"] = build_graph()
    # warm the retriever so the FIRST real request isn't the one paying the
    # model-load cost. get_retriever() is a singleton, so this primes it.
    try:
        from app.retrieval import get_retriever
        get_retriever()
    except Exception as e:
        print(f"(retriever warm-up skipped: {e})")
    print("Startup complete. Ready to serve.")

    yield  # <-- app serves requests here

    # ---- shutdown: close the Qdrant client cleanly ----
    try:
        from app.retrieval import get_retriever
        get_retriever().close()
    except Exception:
        pass
    print("Shutdown complete.")


app = FastAPI(
    title="Yardi Finance Ops Assistant",
    description="RAG + SQL co-pilot for property-management finance operations.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---- request / response schemas (pydantic = automatic validation + docs) ----

class AskRequest(BaseModel):
    question: str = Field(..., description="The user's question.")
    user_role: str = Field(
        ...,
        description="Caller's role for RBAC: ap_clerk, staff_accountant, "
                    "property_controller, regional_manager, or admin.",
    )


class AskResponse(BaseModel):
    answer: str
    route: Optional[str] = None
    route_reason: Optional[str] = None
    citations: List[str] = []
    confident: Optional[bool] = None
    trace: List[str] = []


@app.get("/health")
def health():
    """Liveness check -- used by Docker/CI and load balancers."""
    return {"status": "ok", "graph_loaded": _state["graph"] is not None}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Answer a question through the full graph (router -> retrieve/sql/fusion
    -> generate -> confidence gate)."""
    graph = _state["graph"]
    result = graph.invoke({"question": req.question, "user_role": req.user_role})
    return AskResponse(
        answer=result.get("answer", ""),
        route=result.get("route"),
        route_reason=result.get("route_reason"),
        citations=result.get("citations") or [],
        confident=result.get("confident"),
        trace=result.get("trace") or [],
    )