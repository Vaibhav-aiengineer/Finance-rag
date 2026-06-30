"""
generation.py
-------------
The answer-generation step the `generate_answer` graph node calls. Takes the
retrieved chunks and asks Claude to write an answer that is:

  1. GROUNDED -- uses ONLY the provided chunks, never the model's own knowledge.
     In a finance/compliance domain an ungrounded answer is worse than none.
  2. CITED -- names the doc_id(s) it relied on, so a user can verify against the
     real SOP. This accountability layer is what MediBot lacked.
  3. HONEST -- if the chunks don't contain the answer, it says so rather than
     inventing one.

WHY A SEPARATE MODULE (same reasoning as retrieval.py):
The graph node stays a thin state wrapper; the prompt + API logic lives here so
it's independently testable and swappable.
"""

import os
import re
from typing import List, Dict, Any


def _parse_cited_docs(answer: str, available_doc_ids: List[str]) -> List[str]:
    """Extract the doc_ids the answer ACTUALLY cited, rather than every doc_id
    fed into context. We look at the answer text (which the prompt instructs to
    end with a 'Sources:' line) and keep only the available doc_ids that appear
    in it. This fixes citation over-reporting: a SUPERSEDED doc that was
    retrieved but NOT used by the model won't be falsely listed as a source."""
    cited = [d for d in available_doc_ids if d in answer]
    # if the model didn't name any (unusual), fall back to nothing rather than
    # over-claiming -- an empty citation list is more honest than a wrong one.
    return list(dict.fromkeys(cited))

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()  # reads .env into environment

MODEL = "claude-sonnet-4-6"   # fast + capable; good default for grounded RAG answers
MAX_TOKENS = 700

_client = None

def _get_client() -> Anthropic:
    global _client
    if _client is None:
        # Anthropic() reads ANTHROPIC_API_KEY from the environment automatically.
        _client = Anthropic()
    return _client


SYSTEM_PROMPT = """You are a finance operations assistant for a property \
management company. You answer questions about accounting SOPs using ONLY the \
provided context chunks.

Rules:
- Answer ONLY from the provided context. Do not use outside knowledge.
- If the context does not contain the answer, say you don't have that \
information in the available procedures. Do not guess.
- Be concise and specific. When the answer involves thresholds, amounts, or \
roles, state them exactly as written in the context.
- End your answer with a "Sources:" line listing the doc_id(s) you used.
- If multiple versions of a procedure appear, use the one marked CURRENT and \
ignore SUPERSEDED ones unless the user explicitly asks what the old rule was."""


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """Render retrieved chunks into a labeled context block the model can cite.
    Each chunk shows its doc_id, status, and text so the model can both ground
    and cite, and can see which version is CURRENT vs SUPERSEDED."""
    lines = []
    for i, c in enumerate(chunks, 1):
        status = c.get("status", "CURRENT")
        lines.append(
            f"[Chunk {i}] doc_id={c['doc_id']} status={status} "
            f"section={c.get('section_title')}\n{c['text']}"
        )
    return "\n\n".join(lines)


def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce a grounded, cited answer. Returns {answer, citations}."""
    if not chunks:
        return {
            "answer": "I don't have information on that in the available "
                      "procedures, or you may not have access to the relevant "
                      "documents.",
            "citations": [],
        }

    context = _format_context(chunks)
    user_msg = (
        f"Context chunks:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}\n\n"
        f"Answer using only the context above, and cite the doc_id(s) you used."
    )

    client = _get_client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    answer_text = resp.content[0].text

    # citations = the doc_ids the answer ACTUALLY referenced (parsed from the
    # answer's Sources line), not every doc fed into context. This prevents a
    # retrieved-but-unused SUPERSEDED doc from being falsely listed as a source.
    available = list(dict.fromkeys(c["doc_id"] for c in chunks))
    citations = _parse_cited_docs(answer_text, available)

    return {"answer": answer_text, "citations": citations}


def generate_sql_answer(question: str, sql: str, rows: List[Dict[str, Any]],
                        sql_error: str = None) -> Dict[str, Any]:
    """Turn SQL results (or a SQL error) into a natural-language answer.

    Three cases:
      - sql_error present  -> return the error message as the answer (e.g. an
        access denial). We do NOT call the LLM; the denial is authoritative.
      - rows empty         -> say no matching data was found.
      - rows present       -> ask the LLM to phrase the rows as a concise answer
        to the question (it only formats given data; it does not invent values).
    """
    if sql_error:
        return {"answer": sql_error, "citations": []}

    if not rows:
        return {
            "answer": "No matching records were found in the finance database "
                      "for that question.",
            "citations": ["finance_db"],
        }

    # Render rows compactly for the model. For small result sets this is plenty.
    rows_text = "\n".join(str(r) for r in rows[:50])
    system = (
        "You turn SQL query results into a short, direct natural-language answer "
        "to the user's question. Use ONLY the values in the results -- never "
        "invent or estimate numbers. Be concise. State figures exactly as given. "
        "Do not mention SQL or that this came from a database query."
    )
    user_msg = (
        f"Question: {question}\n\n"
        f"SQL results:\n{rows_text}\n\n"
        f"Answer the question using only these results."
    )
    resp = _get_client().messages.create(
        model=MODEL,
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return {"answer": resp.content[0].text, "citations": ["finance_db"]}


if __name__ == "__main__":
    # standalone smoke test -- requires a real ANTHROPIC_API_KEY in .env
    fake_chunks = [
        {
            "doc_id": "SOP-JE-002", "status": "CURRENT",
            "section_title": "Approval Thresholds",
            "text": "[Approval Thresholds] Journal Entry Amount (USD): 5,001 to "
                    "50,000; Required Approver: Property Controller",
        },
        {
            "doc_id": "SOP-JE-001", "status": "SUPERSEDED",
            "section_title": "Approval Thresholds",
            "text": "[Approval Thresholds] Journal Entry Amount (USD): 2,501 to "
                    "10,000; Required Approver: Property Controller",
        },
    ]
    out = generate_answer(
        "What is the current approval threshold for a $40,000 journal entry?",
        fake_chunks,
    )
    print("ANSWER:\n", out["answer"])
    print("\nCITATIONS:", out["citations"])