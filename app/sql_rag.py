"""
sql_rag.py
----------
The SQL RAG path the `generate_sql` graph node calls. Orchestrates:

  generate (LLM) -> validate -> [retry on validation failure] -> execute

WHY A RETRY LOOP:
If the LLM generates SQL that fails validation (wrong table, not a SELECT, etc.),
we feed the validator's rejection reason back into a second generation attempt.
This is a real, bounded loop -- one of the reasons this project uses LangGraph:
the SQL node has retry behavior, not just a straight line. We cap attempts so a
persistently-bad question can't loop forever; after the cap we return the error
honestly instead of executing anything unsafe.

WHY ACCESS DENIAL IS RETURNED, NOT RETRIED:
If validation fails because the ROLE isn't allowed (vs. a malformed query), we
do NOT retry -- regenerating won't grant access. We surface the denial reason
straight to the user. Only "fixable" failures (bad SQL shape) are retried.
"""

import os
import re
import sqlite3
from typing import Dict, Any, List

from dotenv import load_dotenv
from anthropic import Anthropic

from app.sql_config import SCHEMA_DESCRIPTION, ROLE_TABLE_ACCESS, SQL_ALLOWED_ROLES
from app.sql_validator import validate_sql, check_question_scope

load_dotenv()

MODEL = "claude-sonnet-4-6"
DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "database", "yardi_finance.db")
)
MAX_ATTEMPTS = 2          # initial try + one retry on a fixable validation failure
MAX_ROWS = 100            # safety cap on returned rows

_client = None

def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


def _gen_system(user_role: str) -> str:
    allowed = sorted(ROLE_TABLE_ACCESS.get(user_role, set()))
    return (
        "You write a single SQLite SELECT query to answer the user's question, "
        "using ONLY the schema and tables provided. Output ONLY the SQL query -- "
        "no explanation, no markdown fences, no semicolon.\n\n"
        f"{SCHEMA_DESCRIPTION}\n"
        f"This user's role is '{user_role}'. You may ONLY query these tables: "
        f"{', '.join(allowed)}. Do not reference any other table.\n"
        "Rules: SELECT only. No INSERT/UPDATE/DELETE/DDL. One statement."
    )


def _strip_sql(text: str) -> str:
    """Clean the model output to bare SQL: drop markdown fences and stray prose."""
    t = text.strip()
    t = re.sub(r"^```(?:sql)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    return t.rstrip(";").strip()


def _generate_sql(question: str, user_role: str, prior_error: str = "") -> str:
    user_msg = question
    if prior_error:
        user_msg = (
            f"{question}\n\n(Your previous query was rejected: {prior_error} "
            f"Generate a corrected query that obeys the table and SELECT-only rules.)"
        )
    resp = _get_client().messages.create(
        model=MODEL,
        max_tokens=300,
        system=_gen_system(user_role),
        messages=[{"role": "user", "content": user_msg}],
    )
    return _strip_sql(resp.content[0].text)


def _execute(sql: str) -> List[Dict[str, Any]]:
    """Run the validated SELECT and return rows as dicts (column-name keyed)."""
    if not os.path.exists(DB_PATH):
        raise sqlite3.Error(
            f"Database not found at {DB_PATH}. Run data/database/build_db.py first, "
            f"or check the path."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchmany(MAX_ROWS)]
        return rows
    finally:
        conn.close()

def run_sql_rag(question: str, user_role: str) -> Dict[str, Any]:
    """Full SQL path. Returns a dict with generated_sql, sql_rows, sql_error."""

    # Hard gate: role not allowed to use SQL at all -> deny immediately, no LLM.
    if user_role not in SQL_ALLOWED_ROLES:
        return {
            "generated_sql": None,
            "sql_rows": [],
            "sql_error": (f"Role '{user_role}' is not authorized to query the "
                          f"finance database. This data is available to finance "
                          f"roles only."),
        }

    # PRE-GENERATION scope gate: deny early if the question is clearly about a
    # table this role can't access, before the LLM can silently substitute an
    # allowed table and return a confidently-wrong answer.
    in_scope, scope_reason = check_question_scope(question, user_role)
    if not in_scope:
        return {
            "generated_sql": None,
            "sql_rows": [],
            "sql_error": scope_reason,
        }

    prior_error = ""
    for attempt in range(MAX_ATTEMPTS):
        sql = _generate_sql(question, user_role, prior_error)
        ok, reason = validate_sql(sql, user_role)

        if ok:
            try:
                rows = _execute(sql)
                return {"generated_sql": sql, "sql_rows": rows, "sql_error": None}
            except sqlite3.Error as e:
                # execution-time DB error (e.g. bad column) -> retry with reason
                prior_error = f"SQL execution error: {e}"
                continue

        # validation failed. If it's an access-denial, do NOT retry -- a retry
        # can't grant access. Surface it immediately.
        if "not authorized" in reason.lower():
            return {"generated_sql": sql, "sql_rows": [], "sql_error": reason}

        # otherwise it's a fixable shape problem -> feed reason back and retry
        prior_error = reason

    # exhausted attempts without a valid, executable query
    return {
        "generated_sql": None,
        "sql_rows": [],
        "sql_error": f"Could not produce a valid query after {MAX_ATTEMPTS} attempts. "
                     f"Last issue: {prior_error}",
    }


if __name__ == "__main__":
    tests = [
        ("How many invoices are pending?", "staff_accountant"),
        ("How many properties are in the West region?", "ap_clerk"),
        ("Show me all general ledger transactions", "regional_manager"),  # denied role
        ("What is the total amount of all gl_transactions?", "ap_clerk"),  # disallowed table for clerk
    ]
    for q, role in tests:
        print(f"\nQ ({role}): {q}")
        out = run_sql_rag(q, role)
        print(f"  SQL:   {out['generated_sql']}")
        print(f"  rows:  {out['sql_rows'][:3]}")
        print(f"  error: {out['sql_error']}")