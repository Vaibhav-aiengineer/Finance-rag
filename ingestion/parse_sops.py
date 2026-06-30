"""
parse_sops.py
-------------
Phase 1, Step A: parse every SOP PDF into Markdown using Docling, and join in
the metadata.json (version, access_roles, etc.) that couldn't survive the PDF
conversion. This is a ONE-TIME BATCH job — it does not run per query.

WHY DOCLING HERE (not pypdf/pdfplumber):
Several of these SOPs have real tables (approval thresholds, GL account
ranges, aging buckets). A naive text extractor would flatten those into
unstructured text and destroy row/column relationships. Docling's layout +
table models reconstruct that structure so the downstream chunker can later
split text by heading and tables by row, the same pattern used in MediBot.

WHY THIS IS A SCRIPT, NOT A LANGGRAPH NODE:
There's no branching or retry logic here — every PDF goes through the exact
same single path: load -> convert -> export to markdown -> save. A graph
needs decision points to be worth using; this has none. LangGraph enters the
project later, at the query-time router.

Run with:
    python parse_sops.py
"""

import json
import os

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(HERE, "..", "data", "sops_pdf")
OUT_DIR = os.path.join(HERE, "..", "data", "parsed")
METADATA_PATH = os.path.join(PDF_DIR, "metadata.json")

os.makedirs(OUT_DIR, exist_ok=True)


def load_metadata() -> dict:
    """Load the doc_id -> {version, access_roles, ...} mapping that was pulled
    out of the markdown front-matter when the PDFs were generated. This is
    what we re-attach after parsing, since Docling has no way to know a
    document's version or who's allowed to see it -- that's business
    metadata, not something present in the PDF's visual layout."""
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_all_pdfs():
    metadata = load_metadata()

    # WHY DISABLE OCR: these SOP PDFs are digitally generated -- the text is
    # real, selectable text embedded in the file, not pixels in a scanned
    # image. OCR only does useful work on image-based/scanned pages. Leaving
    # it on makes Docling load and run character-recognition models over every
    # page looking for image text that doesn't exist -- pure wasted compute
    # and slower parsing. We keep table-structure detection ON (that's what
    # gives us the clean pipe tables); we only turn OFF the OCR stage.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))
    print(f"Found {len(pdf_files)} PDFs to parse.\n")

    manifest = []  # tracks what was parsed, for a quick sanity check after

    for fname in pdf_files:
        pdf_path = os.path.join(PDF_DIR, fname)
        doc_id = fname.split("_")[0]  # e.g. "SOP-AP-001" from the filename prefix

        print(f"Parsing {fname} ...")
        result = converter.convert(pdf_path)
        markdown_text = result.document.export_to_markdown()

        # Save the raw parsed markdown -- this is the inspectable intermediate
        # output mentioned above. If chunking misbehaves later, look here
        # first to see if Docling already got the structure wrong.
        out_path = os.path.join(OUT_DIR, fname.replace(".pdf", ".md"))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        doc_meta = metadata.get(doc_id, {})
        manifest.append({
            "doc_id": doc_id,
            "source_pdf": fname,
            "parsed_markdown": os.path.basename(out_path),
            "version": doc_meta.get("version"),
            "status": doc_meta.get("status", "CURRENT"),
            "access_roles": doc_meta.get("access_roles", []),
            "char_count": len(markdown_text),
        })
        print(f"  -> {len(markdown_text)} chars written to {os.path.basename(out_path)}")

    manifest_path = os.path.join(OUT_DIR, "parse_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(manifest)} documents parsed.")
    print(f"Manifest written to {manifest_path}")
 #   print("\nSanity check -- inspect one parsed file before moving to chunking:")
#  print(f"  {os.path.join(OUT_DIR, manifest[0]['parsed_markdown'])}")


if __name__ == "__main__":
    parse_all_pdfs()