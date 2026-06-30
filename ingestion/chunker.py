"""
chunker.py
----------
Phase 1, Step B: turn each parsed SOP markdown file into retrieval-ready chunks
with full metadata attached.

TWO CHUNK TYPES PER DOCUMENT:
  1. SECTION chunks -- the prose under each "## heading", with any table
     REMOVED so the prose embedding isn't diluted by raw table text.
  2. ROW chunks -- one per table row, with the column headers WOVEN IN so each
     row reads like a natural sentence ("Invoice amounts Above 25,000 USD
     require approval by Chief Financial Officer"). This makes each row
     self-describing and embeds far better than a bare "Above 25,000 | CFO".

WHY METADATA IS ATTACHED HERE (the new-vs-MediBot part):
Every chunk carries access_roles (RBAC), plus version/effective_date/status/
superseded_by (version handling). Retrieval can only filter by who's allowed to
see a chunk, and only prefer current-over-superseded versions, if that metadata
is present ON the chunk. Attaching it now is what makes Phase 2's version-aware
retrieval and RBAC filtering POSSIBLE at all -- skip it here and there's
nothing downstream to filter on.

WHY THIS IS STILL A SCRIPT, NOT A LANGGRAPH NODE:
Like parsing, chunking is a single linear path with no branching or retries.
LangGraph is for the query-time decision logic, not batch ingestion.

Run with:
    python chunker.py
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PARSED_DIR = os.path.join(HERE, "..", "data", "parsed")
PDF_DIR = os.path.join(HERE, "..", "data", "sops_pdf")
OUT_DIR = os.path.join(HERE, "..", "data", "chunks")
METADATA_PATH = os.path.join(PDF_DIR, "metadata.json")

os.makedirs(OUT_DIR, exist_ok=True)


def load_metadata() -> dict:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def split_into_sections(markdown_text: str):
    """Split markdown into (section_title, section_body) pairs on '## ' headings.
    Text before the first '## ' (the document title block) is grouped under a
    synthetic 'Overview' section so nothing is lost."""
    sections = []
    current_title = "Overview"
    current_lines = []

    for line in markdown_text.splitlines():
        if line.startswith("## "):
            # flush the previous section before starting a new one
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return sections


def extract_table(section_body: str):
    """Return (prose_without_table, table_rows) for a section.
    table_rows is a list of dicts mapping column_header -> cell_value, or [] if
    the section has no pipe table. Prose is the section text with the table
    lines stripped out."""
    lines = section_body.splitlines()
    table_lines = [l for l in lines if l.strip().startswith("|")]
    prose_lines = [l for l in lines if not l.strip().startswith("|")]

    rows = []
    if len(table_lines) >= 2:
        # first row = headers; second row = separator (|---|---|); rest = data
        headers = [c.strip() for c in table_lines[0].strip().strip("|").split("|")]
        for data_line in table_lines[2:]:
            cells = [c.strip() for c in data_line.strip().strip("|").split("|")]
            if len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))

    prose = "\n".join(prose_lines).strip()
    return prose, headers if rows else [], rows


def weave_row(section_title: str, headers: list, row: dict) -> str:
    """Turn one table row into a self-describing sentence by weaving in the
    column headers. e.g. headers ['Invoice Amount (USD)', 'Required Approver']
    + row {...: 'Above 25,000', ...: 'CFO'} becomes:
      '[Approval Routing and Thresholds] Invoice Amount (USD): Above 25,000;
       Required Approver: Chief Financial Officer'
    Prose-like and header-anchored, so it embeds on its own merits."""
    parts = [f"{h}: {row[h]}" for h in headers]
    return f"[{section_title}] " + "; ".join(parts)


def build_chunks_for_doc(doc_id: str, markdown_text: str, meta: dict):
    """Produce the list of chunk dicts for a single document."""
    chunks = []

    # The metadata bundle attached to EVERY chunk from this document.
    base_meta = {
        "doc_id": doc_id,
        "title": meta.get("title"),
        "category": meta.get("category"),
        "version": meta.get("version"),
        "effective_date": meta.get("effective_date"),
        "status": meta.get("status", "CURRENT"),
        "superseded_by": meta.get("superseded_by"),
        "access_roles": meta.get("access_roles", []),
        "source_pdf": meta.get("source_pdf"),
    }

    for section_title, section_body in split_into_sections(markdown_text):
        prose, headers, rows = extract_table(section_body)

        # 1) SECTION chunk: prose only (table stripped out)
        if prose:
            chunks.append({
                "chunk_type": "section",
                "section_title": section_title,
                "text": f"{section_title}\n\n{prose}",
                **base_meta,
            })

        # 2) ROW chunks: one per table row, headers woven in
        for row in rows:
            chunks.append({
                "chunk_type": "table_row",
                "section_title": section_title,
                "text": weave_row(section_title, headers, row),
                **base_meta,
            })

    return chunks


def chunk_all():
    metadata = load_metadata()
    md_files = sorted(f for f in os.listdir(PARSED_DIR) if f.endswith(".md"))

    all_chunks = []
    for fname in md_files:
        doc_id = fname.split("_")[0]
        with open(os.path.join(PARSED_DIR, fname), "r", encoding="utf-8") as f:
            markdown_text = f.read()

        meta = metadata.get(doc_id, {})
        doc_chunks = build_chunks_for_doc(doc_id, markdown_text, meta)
        all_chunks.extend(doc_chunks)

        n_section = sum(1 for c in doc_chunks if c["chunk_type"] == "section")
        n_row = sum(1 for c in doc_chunks if c["chunk_type"] == "table_row")
        print(f"{doc_id:14s}  {n_section:2d} section + {n_row:2d} row = {len(doc_chunks):2d} chunks "
              f"[{meta.get('status', 'CURRENT')}]")

    # assign a stable id to each chunk (doc_id + running index within doc)
    counters = {}
    for c in all_chunks:
        i = counters.get(c["doc_id"], 0)
        c["chunk_id"] = f"{c['doc_id']}::{i}"
        counters[c["doc_id"]] = i + 1

    out_path = os.path.join(OUT_DIR, "chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print("-" * 60)
    print(f"Total: {len(all_chunks)} chunks across {len(md_files)} documents")
    print(f"Written to {out_path}")
    print("\nSpot-check a woven table row and a superseded-doc chunk:")
    for c in all_chunks:
        if c["chunk_type"] == "table_row" and c["doc_id"] == "SOP-AP-001":
            print("  ROW :", c["text"][:90])
            break
    for c in all_chunks:
        if c["doc_id"] == "SOP-JE-001":
            print(f"  JE-001 status -> {c['status']} (superseded_by: {c['superseded_by']})")
            break


if __name__ == "__main__":
    chunk_all()