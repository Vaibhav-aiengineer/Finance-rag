"""
state.py
--------
The shared state object that flows through every node of the query-time graph.

HOW LANGGRAPH STATE WORKS:
Nodes don't call each other directly. Each node is a function that receives the
current state, does its work, and returns a partial dict of fields to UPDATE.
LangGraph merges that partial dict back into the running state and passes it to
the next node. So state is the ONLY communication channel between nodes --
anything a downstream node needs, an upstream node must have written here.

WHY A TypedDict:
LangGraph uses the type annotations to know what fields exist. It's a plain
dict at runtime (fast, simple), but the annotations document the contract and
give you editor autocomplete. Every field is Optional because early nodes
haven't filled the later fields yet -- e.g. `answer` is None until the
generate node runs.
"""

from typing import Optional, List, Dict, Any, TypedDict, Annotated
import operator


class GraphState(TypedDict, total=False):
    # ---- input ----
    question: str                     # the user's raw question
    user_role: str                    # the caller's role, for RBAC filtering

    # ---- set by the router node ----
    route: Optional[str]              # "doc_rag" | "sql_rag" | "fusion"
    route_reason: Optional[str]       # why the router chose this path (for tracing)

    # ---- set by the fusion decompose node ----
    doc_subquestion: Optional[str]    # the "rule" half of a fusion question
    sql_subquestion: Optional[str]    # the "number" half of a fusion question

    # ---- set by the doc-retrieval node ----
    retrieved_chunks: Optional[List[Dict[str, Any]]]  # chunks from Qdrant (post-filter)

    # ---- set by the SQL node ----
    generated_sql: Optional[str]      # the SELECT the LLM produced
    sql_rows: Optional[List[Dict[str, Any]]]          # rows returned by executing it
    sql_error: Optional[str]          # validation/execution error, if any (drives retry)

    # ---- set by the generate-answer node ----
    answer: Optional[str]             # the final natural-language answer
    citations: Optional[List[str]]    # doc_ids / sources backing the answer

    # ---- set by the confidence-check node ----
    confident: Optional[bool]         # did anything cross the relevance threshold?

    # ---- bookkeeping / tracing ----
    # `trace` uses an `operator.add` reducer: when two PARALLEL nodes both write
    # trace (as the fusion fan-out does), LangGraph CONCATENATES their lists
    # instead of raising a concurrent-write conflict. Without this reducer, two
    # parallel nodes writing the same key would error.
    trace: Annotated[List[str], operator.add]