"""
sql_validator.py
----------------
The safety layer for the SQL RAG path. It re-checks the LLM-generated SQL
BEFORE execution. This is ENFORCEMENT (the prompt is only guidance) -- the
validator does not trust that the prompt successfully constrained the model.

CHECKS (in order, fail closed):
  1. Non-empty, single statement (no stacked queries via ';').
  2. SELECT-only -- must start with SELECT, and must not contain any data- or
     schema-modifying keyword (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/...).
  3. Table whitelist -- every table the SQL references must be in the asking
     role's allowed set (per-role RBAC at the data layer).

WHY TEXT-TO-SQL NEEDS THIS WHERE DOC RAG DOESN'T:
Doc RAG only ever produces text. SQL RAG produces executable code that runs
against a real database. An unvalidated DELETE or a query against an
unauthorized table is a real harm, so the generated SQL is treated as
untrusted input and checked before it touches the DB.
"""

import re
from typing import Tuple, Set

from app.sql_config import (
    ROLE_TABLE_ACCESS, ALL_TABLES, SQL_ALLOWED_ROLES, TABLE_CONCEPTS,
)

# keywords that must never appear -- their presence means the SQL is not a
# pure read, so we reject regardless of context.
FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "create",
    "truncate", "replace", "attach", "detach", "pragma", "vacuum",
    "grant", "revoke",
]


def _referenced_tables(sql: str) -> Set[str]:
    """Extract table names that appear after FROM or JOIN. Deliberately simple
    and conservative: we match the token after FROM/JOIN and intersect with the
    set of tables we actually know about. Anything we can't account for makes
    validation fail elsewhere (whitelist check), so this errs safe."""
    tokens = re.findall(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, re.IGNORECASE)
    return {t.lower() for t in tokens}


def check_question_scope(question: str, user_role: str) -> Tuple[bool, str]:
    """PRE-GENERATION intent gate. Returns (allowed, reason).

    Detects (heuristically, via the concept map) which tables a question seems
    to be ABOUT, and denies early if any of those tables is one the role can't
    access. This stops 'silent scope substitution' -- where the LLM, told it may
    only use the role's allowed tables, quietly answers a GL question from the
    AP table instead and returns a confidently-wrong number.

    Deliberately conservative and transparent. It is the EARLY gate; the SQL
    validator is still the authoritative check on the generated query. A miss
    here (unusual phrasing) is backstopped there."""
    q = question.lower()
    allowed_tables = ROLE_TABLE_ACCESS.get(user_role, set())

    # which tables does the question appear to be about?
    targeted = set()
    for table, concepts in TABLE_CONCEPTS.items():
        if any(concept in q for concept in concepts):
            targeted.add(table)

    # only block on tables that are real AND outside the role's allowed set
    blocked = {t for t in (targeted - allowed_tables) if t in ALL_TABLES}
    if blocked:
        return False, (
            f"Your question appears to ask about {', '.join(sorted(blocked))} "
            f"data, which the '{user_role}' role is not authorized to access."
        )

    return True, ""


def validate_sql(sql: str, user_role: str) -> Tuple[bool, str]:
    """Return (is_valid, reason). reason is '' when valid, else why it failed.
    The reason string is fed back to the generator on failure to drive a retry,
    and surfaced to the user when access is denied."""

    # 0. role must be allowed to use SQL at all
    if user_role not in SQL_ALLOWED_ROLES:
        return False, (f"Role '{user_role}' is not authorized to query the "
                       f"finance database directly.")

    if not sql or not sql.strip():
        return False, "Empty SQL."

    s = sql.strip().rstrip(";").strip()

    # 1. single statement only -- no stacked queries
    if ";" in s:
        return False, "Multiple SQL statements are not allowed."

    # 2. must be a SELECT
    if not s.lower().startswith("select"):
        return False, "Only SELECT queries are allowed."

    # 3. no forbidden (write/DDL) keywords anywhere
    lowered = s.lower()
    for kw in FORBIDDEN_KEYWORDS:
        # word-boundary match so e.g. 'created_at' wouldn't trip 'create'
        if re.search(rf"\b{kw}\b", lowered):
            return False, f"Forbidden keyword '{kw}' present; only read queries allowed."

    # 4. per-role table whitelist
    allowed = ROLE_TABLE_ACCESS.get(user_role, set())
    referenced = _referenced_tables(s)

    # any referenced table that isn't a real table at all -> reject
    unknown = referenced - ALL_TABLES
    if unknown:
        return False, f"Query references unknown table(s): {', '.join(sorted(unknown))}."

    # any referenced table not in this role's allowed set -> access denied
    disallowed = referenced - allowed
    if disallowed:
        return False, (f"Role '{user_role}' is not authorized to access "
                       f"table(s): {', '.join(sorted(disallowed))}.")

    return True, ""