"""CLI entry point for ingesting and indexing research papers.

Usage:
    python src/ingest_cli.py --paper research-data/2604.01733v1.pdf
    python src/ingest_cli.py --all
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import Dict, List, Optional, Any
import pandas as pd

from src.config import (
    BASE_DIR,
    CATALOGUE_PATH,
    RESEARCH_DATA_DIR,
    INDEX_DIR,
    FIGURES_DIR,
    sanitize_error,
)
from src.ingestion.validator import validate_pdf
from src.ingestion.parser import parse_pdf_document
from src.ingestion.visual_extractor import extract_visual_elements
from src.ingestion.chunker import chunk_document
from src.retrieval.embeddings import EmbeddingProvider
from src.retrieval.vector_store import VectorIndex
from src.retrieval.bm25_index import BM25Index


def load_catalog_map() -> Dict[str, Dict[str, str]]:
    """Loads catalogue CSV mapping filename -> metadata."""
    if not CATALOGUE_PATH.exists():
        return {}
    df = pd.read_csv(CATALOGUE_PATH)
    res = {}
    for _, row in df.iterrows():
        fname = str(row.get("filename", "")).strip()
        if fname:
            res[fname] = {
                "paper_id": str(row.get("paper_id", "")).strip(),
                "title": str(row.get("title", "")).strip(),
            }
    return res


def print_summary_table(rows: List[Dict[str, Any]]):
    """Prints a clean ASCII summary table in console/PowerShell."""
    if not rows:
        print("No papers processed.")
        return

    headers = ["Paper ID", "Title", "Pages", "Text Chunks", "Figures", "Tables", "Captions Linked", "Warnings"]
    col_widths = [10, 36, 7, 13, 9, 8, 17, 15]

    def fmt_cell(val, width):
        s = str(val)
        if len(s) > width:
            return s[: width - 3] + "..."
        return s.ljust(width)

    border = "+-" + "-+-".join(["-" * w for w in col_widths]) + "-+"
    header_line = "| " + " | ".join([fmt_cell(h, col_widths[i]) for i, h in enumerate(headers)]) + " |"

    print("\n" + border)
    print(header_line)
    print(border)

    for r in rows:
        line = "| " + " | ".join([
            fmt_cell(r.get("paper_id", "-"), col_widths[0]),
            fmt_cell(r.get("title", "-"), col_widths[1]),
            fmt_cell(r.get("pages", 0), col_widths[2]),
            fmt_cell(r.get("text_chunks", 0), col_widths[3]),
            fmt_cell(r.get("figures", 0), col_widths[4]),
            fmt_cell(r.get("tables", 0), col_widths[5]),
            fmt_cell(r.get("captions_linked", 0), col_widths[6]),
            fmt_cell(r.get("warnings", "None"), col_widths[7]),
        ]) + " |"
        print(line)

    print(border + "\n")


def ingest_single_paper(
    pdf_path: Path,
    catalog_map: Dict[str, Dict[str, str]],
    embed_provider: EmbeddingProvider,
    vec_index: VectorIndex,
    bm25_idx: BM25Index,
) -> Optional[Dict[str, Any]]:
    """Runs ingestion, visual extraction, chunking, and incremental indexing on one PDF."""
    path = Path(pdf_path)
    filename = path.name

    # 1. Validation
    is_valid, err_msg = validate_pdf(path)
    if not is_valid:
        print(f"[SKIP] {filename}: {err_msg}")
        return None

    # Resolve metadata from catalogue
    meta = catalog_map.get(filename, {})
    paper_id = meta.get("paper_id") or path.stem[:8].upper()
    paper_title = meta.get("title") or path.stem

    print(f"\n---> Processing [{paper_id}] {filename} ...")

    # 2. Parsing
    parsed = parse_pdf_document(path, paper_id=paper_id, paper_title=paper_title)

    # 3. Visual & Table Extraction
    figures, tables = extract_visual_elements(path, paper_id=paper_id, paper_title=paper_title)

    # 4. Chunking
    chunks = chunk_document(parsed, figures=figures, tables=tables)

    # Count captions linked
    captions_linked = len(figures) + len(tables)

    # 5. Embedding & Incremental Indexing
    texts_to_embed = [c.text for c in chunks]
    print(f"     Embedding {len(chunks)} chunks with local MiniLM model...")
    embeddings = embed_provider.embed_texts(texts_to_embed)

    vec_index.add_chunks(chunks, embeddings)
    bm25_idx.add_chunks(chunks)

    # Persist after each paper
    vec_index.save(INDEX_DIR)
    bm25_idx.save(INDEX_DIR)

    warn_str = ", ".join(parsed.get("warnings", [])) if parsed.get("warnings") else "None"

    return {
        "paper_id": paper_id,
        "title": paper_title,
        "pages": parsed["total_pages"],
        "text_chunks": len([c for c in chunks if c.chunk_type == "text"]),
        "figures": len(figures),
        "tables": len(tables),
        "captions_linked": captions_linked,
        "warnings": warn_str,
        "sample_figures": figures[:3],
    }


def main():
    parser = argparse.ArgumentParser(description="ResearchIQ PDF Ingestion & Indexing Pipeline")
    parser.add_argument("--paper", type=str, help="Path to single PDF to ingest")
    parser.add_argument("--all", action="store_true", help="Ingest all PDFs in research-data/")
    args = parser.parse_args()

    if not args.paper and not args.all:
        parser.print_help()
        sys.exit(1)

    catalog_map = load_catalog_map()
    embed_provider = EmbeddingProvider()

    # Load existing indices if present (to support incremental additions)
    vec_index = VectorIndex()
    vec_index.load(INDEX_DIR)
    bm25_idx = BM25Index()
    bm25_idx.load(INDEX_DIR)

    processed_summaries = []

    if args.paper:
        target_path = Path(args.paper)
        if not target_path.is_absolute():
            target_path = (BASE_DIR / target_path).resolve()
        summary = ingest_single_paper(target_path, catalog_map, embed_provider, vec_index, bm25_idx)
        if summary:
            processed_summaries.append(summary)

    elif args.all:
        pdf_files = sorted(RESEARCH_DATA_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {RESEARCH_DATA_DIR}")
            sys.exit(1)

        print(f"Discovered {len(pdf_files)} research papers in research-data/.")
        for pdf_path in pdf_files:
            summary = ingest_single_paper(pdf_path, catalog_map, embed_provider, vec_index, bm25_idx)
            if summary:
                processed_summaries.append(summary)

    print("\n================== INGESTION & INDEXING SUMMARY ==================")
    print_summary_table(processed_summaries)
    print(f"Indices persisted to: {INDEX_DIR}")
    print(f"Total indexed chunks: {len(vec_index.chunks)}")


if __name__ == "__main__":
    main()
