"""
fusion.py
---------
Phase 4: the fusion path's helpers. Fusion answers a question that needs BOTH a
database number AND a document rule, e.g.:

  "What were Riverside's repair costs in 2024, and are those costs eligible to
   be recovered through CAM?"
   -> number: repair costs (SQL over gl_transactions)
   -> rule:   CAM-recoverability (docs: GL coding / CAM SOPs)

Two helpers here:
  1. decompose() -- splits the fusion question into a focused sql_subquestion
     (the "number" part) and doc_subquestion (the "rule" part). We chose to
     decompose (Option B) so each parallel sub-pipeline receives a clean,
     single-purpose question in its proven sweet spot, rather than the messy
     combined question that could pollute SQL generation with the rule clause.
  2. synthesize() -- the join step: takes the SQL rows AND the retrieved chunks
     and asks the LLM to produce ONE answer combining the number and the rule,
     grounded and cited.
"""

import json
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

MODEL = "claude-sonnet-4-6"
_client = None

def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


DECOMPOSE_SYSTEM = """You split a finance question that has TWO parts -- a data \
part (answerable from a database: amounts, counts, totals) and a rule part \
(answerable from a procedure document: policies, eligibility, definitions) -- \
into those two sub-questions.

Respond with ONLY a JSON object, no other text:
{"sql_subquestion": "<the data question>", "doc_subquestion": "<the rule question>"}

The sql_subquestion should be a clean standalone data question with no rule \
language. The doc_subquestion should be a clean standalone rule question with \
no specific numbers. Preserve key entities (property names, years, account \
types) in whichever sub-question needs them."""


def decompose(question: str) -> Dict[str, str]:
    """Split a fusion question into sql_subquestion and doc_subquestion.
    Falls back to using the full question for both if parsing fails -- safer to
    over-ask each pipeline than to drop part of the question."""
    resp = _get_client().messages.create(
        model=MODEL,
        max_tokens=300,
        system=DECOMPOSE_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    raw = resp.content[0].text.strip()
    # strip any accidental markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(raw)
        sqlq = obj.get("sql_subquestion") or question
        docq = obj.get("doc_subquestion") or question
        return {"sql_subquestion": sqlq, "doc_subquestion": docq}
    except json.JSONDecodeError:
        # fallback: hand the full question to both halves
        return {"sql_subquestion": question, "doc_subquestion": question}


SYNTH_SYSTEM = """You answer a finance question that has two parts: a data part \
(answered by the SQL results) and a rule part (answered by the document \
context). Combine BOTH into one clear answer.

Rules:
- State the number/figure exactly as given in the SQL results. Never invent \
figures.
- State the rule/eligibility exactly as supported by the document context. Do \
not use outside knowledge.
- If either part is missing, answer the part you can and say the other part \
isn't available.
- Prefer CURRENT documents over SUPERSEDED ones.
- End with a "Sources:" line citing the document doc_id(s) used, and note the \
figure came from the finance database."""


def synthesize(question: str, sql_rows: List[Dict[str, Any]],
               chunks: List[Dict[str, Any]],
               sql_error: str = None) -> Dict[str, Any]:
    """The join: combine SQL number + document rule into one grounded answer."""
    rows_text = ("(no data returned)" if not sql_rows
                 else "\n".join(str(r) for r in sql_rows[:50]))
    if sql_error:
        rows_text = f"(database access issue: {sql_error})"

    if chunks:
        ctx = "\n\n".join(
            f"[{c['doc_id']} status={c.get('status', 'CURRENT')}] {c['text']}"
            for c in chunks
        )
    else:
        ctx = "(no document context retrieved)"

    user_msg = (
        f"Question: {question}\n\n"
        f"SQL results (the data part):\n{rows_text}\n\n"
        f"Document context (the rule part):\n{ctx}\n\n"
        f"Give one combined answer."
    )
    resp = _get_client().messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYNTH_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    answer_text = resp.content[0].text
    # only count doc_ids the answer actually referenced (+ the db marker if a
    # figure was used). Prevents listing a retrieved-but-unused doc as a source.
    available = list(dict.fromkeys(c["doc_id"] for c in chunks))
    citations = [d for d in available if d in answer_text]
    if sql_rows and not sql_error:
        citations.append("finance_db")
    return {"answer": answer_text, "citations": citations}