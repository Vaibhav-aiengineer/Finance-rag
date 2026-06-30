"""
graph.py
--------
The query-time pipeline as a LangGraph state machine.

THIS IS A SKELETON: every node is a placeholder that just stamps the state and
logs itself in `trace`. No real retrieval, SQL, or LLM calls yet. The point is
to get the GRAPH STRUCTURE -- nodes, edges, and the conditional router branch --
running end to end, so we can fill in one node at a time afterward and test each
in isolation.

THE GRAPH SHAPE:

                          +--> retrieve_docs --+
    START -> route -------+--> generate_sql ---+--> generate_answer
              (conditional)+--> fusion ---------+         |
                                                          v
                                                 confidence_check -> END

The conditional edge after `route` is the heart of why we use LangGraph: it
reads state["route"] and dispatches to one of three branches. A plain script
would do this with if/else; LangGraph makes it an explicit, inspectable graph
with retry/loop capability we'll use later (SQL validation retry, low-confidence
query rewrite).

Run with:
    python graph.py
"""

from langgraph.graph import StateGraph, START, END

try:
    from app.state import GraphState
except ModuleNotFoundError:
    from state import GraphState


# ---------------------------------------------------------------------------
# Node functions. Each takes the current state and returns a PARTIAL dict of
# fields to update. LangGraph merges that into the running state.
# ---------------------------------------------------------------------------

def _log(state: GraphState, name: str) -> list:
    """Return this node's own trace increment as a single-element list.
    Because `trace` in GraphState uses an operator.add reducer, LangGraph
    CONCATENATES each node's increment onto the running trace automatically.
    So each node returns ONLY its own name here -- not the whole accumulated
    list -- otherwise the reducer would double-count earlier entries."""
    return [name]


def route_node(state: GraphState) -> dict:
    """Classify the question into doc_rag / sql_rag / fusion via the router
    (LLM classifier for now; keyword fast-path can slot in later). Delegates to
    router.classify so the node stays a thin state wrapper."""
    try:
        from app.router import classify
    except ModuleNotFoundError:
        from router import classify
    decision = classify(state["question"])
    return {
        "route": decision["route"],
        "route_reason": decision["route_reason"],
        "trace": _log(state, "router"),
    }


def retrieve_docs_node(state: GraphState) -> dict:
    """Phase 2 doc retrieval: dense + BM25 + rerank against Qdrant, with
    dual-layer RBAC filtering (Qdrant filter for dense, post-filter for BM25)
    and version soft-demote (SUPERSEDED chunks down-ranked so CURRENT wins,
    but kept reachable). Delegates the actual work to HybridRetriever so the
    node stays a thin state wrapper."""
    # import here (not at top) so the heavy models only load when this node
    # actually runs. Try package-relative first, fall back to flat import so
    # it works whether run as `python app/graph.py` or `python -m app.graph`.
    try:
        from app.retrieval import get_retriever
    except ModuleNotFoundError:
        from retrieval import get_retriever
    retriever = get_retriever()
    chunks = retriever.retrieve(state["question"], state["user_role"])
    return {
        "retrieved_chunks": chunks,
        "trace": _log(state, "retrieve_docs"),
    }


def generate_sql_node(state: GraphState) -> dict:
    """SQL RAG path: generate a SELECT, validate it (SELECT-only, table
    whitelist, per-role access), retry once on a fixable failure, then execute.
    Delegates to sql_rag.run_sql_rag so the node stays a thin state wrapper."""
    try:
        from app.sql_rag import run_sql_rag
    except ModuleNotFoundError:
        from sql_rag import run_sql_rag
    result = run_sql_rag(state["question"], state["user_role"])
    return {
        "generated_sql": result["generated_sql"],
        "sql_rows": result["sql_rows"],
        "sql_error": result["sql_error"],
        "trace": _log(state, "generate_sql"),
    }


def fusion_decompose_node(state: GraphState) -> dict:
    """Split a fusion question into a focused SQL sub-question (the number) and
    doc sub-question (the rule), so each parallel branch gets a clean,
    single-purpose question. Writes both sub-questions to state; the two
    parallel branches read them."""
    try:
        from app.fusion import decompose
    except ModuleNotFoundError:
        from fusion import decompose
    parts = decompose(state["question"])
    return {
        "sql_subquestion": parts["sql_subquestion"],
        "doc_subquestion": parts["doc_subquestion"],
        "trace": _log(state, "fusion_decompose"),
    }


def fusion_retrieve_docs_node(state: GraphState) -> dict:
    """Parallel branch 1: retrieve the RULE half. Uses the doc_subquestion.
    Writes ONLY retrieved_chunks (+ its trace increment) so it never collides
    with the parallel SQL branch's writes."""
    try:
        from app.retrieval import get_retriever
    except ModuleNotFoundError:
        from retrieval import get_retriever
    retriever = get_retriever()
    chunks = retriever.retrieve(state["doc_subquestion"], state["user_role"])
    return {
        "retrieved_chunks": chunks,
        "trace": _log(state, "fusion_retrieve_docs"),
    }


def fusion_generate_sql_node(state: GraphState) -> dict:
    """Parallel branch 2: get the NUMBER half. Uses the sql_subquestion.
    Writes ONLY sql_* fields (+ its trace increment) so it never collides with
    the parallel doc branch's writes."""
    try:
        from app.sql_rag import run_sql_rag
    except ModuleNotFoundError:
        from sql_rag import run_sql_rag
    result = run_sql_rag(state["sql_subquestion"], state["user_role"])
    return {
        "generated_sql": result["generated_sql"],
        "sql_rows": result["sql_rows"],
        "sql_error": result["sql_error"],
        "trace": _log(state, "fusion_generate_sql"),
    }


def fusion_join_node(state: GraphState) -> dict:
    """The JOIN: both parallel branches have completed, so state now has both
    retrieved_chunks AND sql_rows. Synthesize them into one combined answer.
    Sets `answer` and `citations` directly (so fusion skips the normal
    generate_answer node and goes straight to confidence_check)."""
    try:
        from app.fusion import synthesize
    except ModuleNotFoundError:
        from fusion import synthesize
    result = synthesize(
        state["question"],
        state.get("sql_rows") or [],
        state.get("retrieved_chunks") or [],
        state.get("sql_error"),
    )
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "trace": _log(state, "fusion_join"),
    }


def generate_answer_node(state: GraphState) -> dict:
    """Produce the final answer. Branches on what the upstream path populated:
    - SQL path (route sql_rag): format sql_rows (or a sql_error) into an answer.
    - Doc path (route doc_rag): generate a grounded, cited answer from chunks.
    (Fusion handled later.) Keeps the node a thin wrapper over generation.py."""
    try:
        from app.generation import generate_answer, generate_sql_answer
    except ModuleNotFoundError:
        from generation import generate_answer, generate_sql_answer

    route = state.get("route")

    if route == "sql_rag":
        result = generate_sql_answer(
            state["question"],
            state.get("generated_sql"),
            state.get("sql_rows") or [],
            state.get("sql_error"),
        )
    else:  # doc_rag (and, for now, anything else) -> document grounding
        result = generate_answer(state["question"], state.get("retrieved_chunks") or [])

    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "trace": _log(state, "generate_answer"),
    }


def confidence_check_node(state: GraphState) -> dict:
    """The honesty gate. Decides whether the produced answer is trustworthy,
    and if not, replaces it with an honest 'I don't have that' message.

    Three signals (Option 3), with one critical distinction baked in:
    a confident REFUSAL (access denial, or a deliberate 'no records') is NOT
    low confidence -- it's a correct, authoritative answer that passes through
    untouched. We only flag genuine 'couldn't find it' failures.

      1. SQL access denial / error  -> authoritative, pass through (confident).
      2. Doc path score threshold   -> if the best retrieved chunk scored below
         RELEVANCE_FLOOR, nothing relevant was found -> not confident.
      3. Generator self-report      -> if the answer text itself says it doesn't
         have the information, honor that -> not confident.
    """
    RELEVANCE_FLOOR = 0.0   # rerank scores below this mean "no real match"
    route = state.get("route")
    answer = state.get("answer") or ""

    # --- signal 1: SQL denials / errors are authoritative refusals ---
    # A role/scope denial is a CORRECT answer the system is certain about, so it
    # stays as-is and counts as confident. (We don't overwrite a denial.)
    if route == "sql_rag" and state.get("sql_error"):
        return {"confident": True, "trace": _log(state, "confidence_check")}

    # --- signal 3: generator self-report of "I don't have that" ---
    # cheap text check across all routes. Phrasings the prompts tend to produce.
    low_conf_markers = [
        "don't have information", "don't have that information",
        "do not have information", "not have that in the available",
        "isn't available", "is not available", "no matching records",
        "not authorized",  # treated below: authorized denial vs. genuine gap
    ]
    said_cannot = any(m in answer.lower() for m in low_conf_markers)

    # "not authorized" is an authoritative denial, not a low-confidence gap.
    is_denial = "not authorized" in answer.lower()

    # --- signal 2: doc-path retrieval score floor ---
    weak_retrieval = False
    if route in ("doc_rag", "fusion"):
        chunks = state.get("retrieved_chunks") or []
        if not chunks:
            weak_retrieval = True
        else:
            best = max((c.get("adjusted_score", c.get("rerank_score", 0.0))
                        for c in chunks), default=0.0)
            if best < RELEVANCE_FLOOR:
                weak_retrieval = True

    # --- decide ---
    if is_denial:
        # authoritative refusal -> confident, leave answer alone
        return {"confident": True, "trace": _log(state, "confidence_check")}

    if weak_retrieval or said_cannot:
        honest = ("I don't have information on that in the available finance "
                  "procedures or data. Please rephrase, or check whether this "
                  "falls outside the scope of the SOPs and finance database.")
        return {
            "confident": False,
            "answer": honest,
            "trace": _log(state, "confidence_check"),
        }

    return {"confident": True, "trace": _log(state, "confidence_check")}


# ---------------------------------------------------------------------------
# The conditional edge function: reads state["route"] and returns the NAME of
# the next node to run. This is what makes the router a real branch.
# ---------------------------------------------------------------------------

def route_decider(state: GraphState) -> str:
    return state["route"]   # "doc_rag" | "sql_rag" | "fusion"


# ---------------------------------------------------------------------------
# Build the graph.
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(GraphState)

    # register nodes
    g.add_node("router", route_node)
    g.add_node("retrieve_docs", retrieve_docs_node)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("generate_answer", generate_answer_node)
    g.add_node("confidence_check", confidence_check_node)

    # fusion sub-graph nodes
    g.add_node("fusion_decompose", fusion_decompose_node)
    g.add_node("fusion_retrieve_docs", fusion_retrieve_docs_node)
    g.add_node("fusion_generate_sql", fusion_generate_sql_node)
    g.add_node("fusion_join", fusion_join_node)

    # entry: START -> router
    g.add_edge(START, "router")

    # conditional branch out of router.
    g.add_conditional_edges(
        "router",
        route_decider,
        {
            "doc_rag": "retrieve_docs",
            "sql_rag": "generate_sql",
            "fusion": "fusion_decompose",
        },
    )

    # doc and sql single-path branches converge on generate_answer
    g.add_edge("retrieve_docs", "generate_answer")
    g.add_edge("generate_sql", "generate_answer")
    g.add_edge("generate_answer", "confidence_check")

    # ---- fusion sub-graph ----
    # decompose then FAN OUT to both parallel branches: adding two edges from
    # one node makes LangGraph run both next nodes concurrently.
    g.add_edge("fusion_decompose", "fusion_retrieve_docs")
    g.add_edge("fusion_decompose", "fusion_generate_sql")
    # both parallel branches FAN IN to the join. fusion_join only runs once BOTH
    # of its predecessors have completed -- that's the synchronization barrier.
    g.add_edge("fusion_retrieve_docs", "fusion_join")
    g.add_edge("fusion_generate_sql", "fusion_join")
    # fusion produces its own answer in the join, so it goes straight to the
    # confidence gate (skipping the doc/sql generate_answer node).
    g.add_edge("fusion_join", "confidence_check")

    g.add_edge("confidence_check", END)

    return g.compile()


if __name__ == "__main__":
    graph = build_graph()

    # Exercise the different routes end to end now that doc_rag and sql_rag
    # both produce real answers. (role varies to show RBAC in action.)
    cases = [
        ("How are vendor invoices approved?", "staff_accountant"),
        ("How many invoices are pending?", "staff_accountant"),
        ("What is the total amount of all gl_transactions?", "ap_clerk"),  # denied
        ("What were Riverside Industrial Park's repair costs in 2024, and are "
         "those costs eligible to be recovered through CAM?", "staff_accountant"),  # fusion
        ("What is the capital of France?", "staff_accountant"),  # out of scope (Q20)
    ]
    for q, role in cases:
        result = graph.invoke({"question": q, "user_role": role})
        print(f"\nQ ({role}): {q}")
        print(f"  route:   {result['route']}  ({result['route_reason']})")
        print(f"  trace:   {' -> '.join(result['trace'])}")
        print(f"  answer:  {result['answer'][:300]}")