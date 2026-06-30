"""
router.py
---------
Phase 3: the router that classifies each question into one of three routes:

  doc_rag  -- answerable from SOP documents ("how do I process an invoice?")
  sql_rag  -- answerable from the finance database ("how many invoices pending?")
  fusion   -- needs BOTH: a number from SQL AND a rule from docs
              ("what were CHI-455's repair costs, and are they CAM-recoverable?")

WHY AN LLM CLASSIFIER (not keyword matching):
Keyword routing ("how many" -> sql) can't detect FUSION at all -- there's no
keyword for "this needs both a number and a rule." Since fusion is a headline
feature of this project, the router must be able to reach it, and only an
LLM that understands intent can. The cost is one small, fast classification
call per query -- negligible for a non-real-time assistant.

DESIGNED FOR A LATER KEYWORD FAST-PATH:
classify() is the public entry point. Right now it always calls the LLM. Later
we can add a keyword_prefilter() that returns a route instantly for obvious
cases and only falls through to the LLM when ambiguous -- without changing how
the graph node calls this. The seam is already here (see the TODO).
"""

import os
from typing import Optional

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

MODEL = "claude-sonnet-4-6"
VALID_ROUTES = {"doc_rag", "sql_rag", "fusion"}

_client = None

def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


CLASSIFIER_SYSTEM = """You are a query router for a property-management finance \
assistant. Classify the user's question into EXACTLY ONE route. Respond with \
ONLY the route word, nothing else.

Routes:
- doc_rag: The answer is a procedure, rule, policy, definition, or threshold \
described in an SOP document. Examples: "How do I process a vendor invoice?", \
"What's the month-end close checklist?", "What is the journal entry approval \
threshold?", "Which accounts are CAM-eligible?"
- sql_rag: The answer is a number, count, list, or aggregate from the finance \
database, OR the question asks whether the user can SEE/ACCESS specific \
database data (transactions, GL, invoices, etc.). Examples: "How many invoices \
are pending?", "What's the total of pending AP?", "How many properties are in \
the West region?", "Which property had the largest CAM variance?", "Can I see \
the individual GL transactions for a property?", "Am I allowed to view \
transaction-level data?" (these access questions go to sql_rag so the database \
role-gate can authoritatively allow or deny them).
- fusion: The question needs BOTH a value from the database AND a rule or \
definition from a document to fully answer. Example: "What were Riverside's \
repair costs in 2024, and are those costs CAM-recoverable?" (the cost is a \
database number; the recoverability is a document rule).

Output exactly one of: doc_rag, sql_rag, fusion"""


def keyword_prefilter(question: str) -> Optional[str]:
    """TODO (later optimization): return a route instantly for obvious cases,
    or None to fall through to the LLM. Disabled for now -- we validate the LLM
    classifier first, then add this in front once we know which questions are
    'obvious' enough to skip the LLM call. Returning None always = LLM decides."""
    return None


def _llm_classify(question: str) -> str:
    client = _get_client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=10,                      # we only need one word back
        system=CLASSIFIER_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    raw = resp.content[0].text.strip().lower()

    # be defensive: the model should return exactly one route word, but if it
    # wraps it in punctuation or extra text, extract the first valid route.
    for route in VALID_ROUTES:
        if route in raw:
            return route
    # safe default: if classification is unclear, treat as a document question
    # (doc_rag is the most common and least risky -- it never touches the DB).
    return "doc_rag"


def classify(question: str) -> dict:
    """Public entry point. Returns {route, route_reason}."""
    kw = keyword_prefilter(question)
    if kw is not None:
        return {"route": kw, "route_reason": "keyword fast-path"}

    route = _llm_classify(question)
    return {"route": route, "route_reason": "llm classifier"}


if __name__ == "__main__":
    # standalone test against representative questions from the eval set
    tests = [
        "How are vendor invoices approved?",                 # doc_rag
        "What is the journal entry approval threshold?",     # doc_rag
        "How many invoices are pending?",                    # sql_rag
        "How many properties are in the West region?",       # sql_rag
        "Which property had the largest CAM variance in 2024?",  # sql_rag
        "What were Riverside Industrial Park's repair costs in 2024, "
        "and are those costs eligible to be recovered through CAM?",  # fusion
    ]
    for q in tests:
        out = classify(q)
        print(f"  {out['route']:8s} <- {q[:65]}")