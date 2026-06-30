"""
sql_config.py
-------------
Configuration for the SQL RAG path: which roles may use SQL at all, which
tables each role may query, and the schema description handed to the LLM.

WHY ROLE -> TABLE MAPPING LIVES HERE (not in the prompt):
The prompt TELLS the model which tables it may use, but this map is what the
validator ENFORCES against the generated SQL. Keeping it as data (not prose in
a prompt) means the enforcement is testable and can't be talked around by a
cleverly worded question. Mirrors the RBAC role matrix in the project README.
"""

# Roles allowed to use the SQL path at all. A role not listed here gets every
# analytical question blocked before any SQL is even generated.
SQL_ALLOWED_ROLES = {"ap_clerk", "staff_accountant", "property_controller", "admin"}
# (regional_manager is deliberately NOT here for transaction-level access -- see
#  golden Q15: they get portfolio summaries via docs, not raw GL/table queries.)

# Per-role table whitelist. The validator rejects any generated SQL that
# references a table outside the asking role's allowed set.
ROLE_TABLE_ACCESS = {
    "ap_clerk": {
        "ap_invoices", "properties", "chart_of_accounts",
    },
    "staff_accountant": {
        "ap_invoices", "gl_transactions", "leases",
        "cam_reconciliations", "properties", "chart_of_accounts",
    },
    "property_controller": {
        "ap_invoices", "gl_transactions", "leases", "cam_reconciliations",
        "properties", "chart_of_accounts",
    },
    "admin": {
        "ap_invoices", "gl_transactions", "leases", "cam_reconciliations",
        "properties", "chart_of_accounts",
    },
    # regional_manager intentionally absent -> no direct table access
}

# All physically real tables (the master whitelist). Even an admin can't query
# a table that isn't here -- prevents access to anything not explicitly modeled.
ALL_TABLES = {
    "properties", "leases", "ap_invoices",
    "gl_transactions", "cam_reconciliations", "chart_of_accounts",
}

# Schema description given to the LLM so it can write correct SQL. Kept concise
# and accurate -- the model writes better SQL from a clear schema than from raw
# CREATE statements.
SCHEMA_DESCRIPTION = """Database schema (SQLite):

properties(property_code PK, property_name, region, property_type, city, state, total_sqft)
leases(lease_id PK, property_code FK, tenant_name, unit, monthly_rent, escalation_pct, lease_start, lease_end, rentable_sqft)
ap_invoices(invoice_id PK, vendor_code, vendor_name, property_code FK, invoice_date, due_date, amount, gl_account FK, approval_status, approved_by, description)
  -- approval_status is one of: 'Pending', 'Approved', 'Paid', 'Rejected'
gl_transactions(transaction_id PK, transaction_date, post_month, property_code FK, gl_account FK, gl_account_name, debit, credit, amount, description)
cam_reconciliations(recon_id PK, property_code FK, recon_year, budgeted_cam, actual_cam, variance, tenant_share_pct, status)
chart_of_accounts(gl_account PK, gl_account_name, account_type, normal_balance)

Notes:
- region values include 'Northeast', 'Midwest', 'South', 'West'
- property_type values include 'Office', 'Retail', 'Industrial', 'Mixed-Use'
- For "CAM variance" use cam_reconciliations.variance (actual_cam - budgeted_cam)
"""

# ---------------------------------------------------------------------------
# Concept map for the PRE-GENERATION scope gate (intent detection).
# Maps each table to the words/phrases that signal a question is *about* that
# table's data. Used BEFORE SQL is generated: if a question clearly targets a
# table the asking role can't access, we deny early instead of letting the LLM
# silently substitute an allowed table and return a confidently-wrong answer
# ("silent scope substitution").
#
# This is a deterministic, auditable heuristic -- intentionally simple. It is
# the EARLY gate; the SQL validator remains the AUTHORITATIVE enforcement on the
# generated query. Two layers, different strengths.
# ---------------------------------------------------------------------------
TABLE_CONCEPTS = {
    "gl_transactions": [
        "gl transaction", "gl transactions", "gl_transaction", "gl_transactions",
        "general ledger", "ledger", "journal entry posting", "gl posting",
        "posted to the gl", "transaction", "transactions", "debit", "credit",
    ],
    "ap_invoices": [
        "invoice", "invoices", "accounts payable", "ap ", "vendor payment",
        "payable", "pending invoice", "approval status",
    ],
    "leases": [
        "lease", "leases", "tenant", "rent", "rentable", "escalation",
    ],
    "cam_reconciliations": [
        "cam", "common area maintenance", "reconciliation", "true-up", "true up",
        "cam variance", "budgeted cam", "actual cam",
    ],
    "properties": [
        "property", "properties", "portfolio", "region", "square feet", "sqft",
    ],
    "chart_of_accounts": [
        "chart of accounts", "gl account name", "account type",
    ],
}