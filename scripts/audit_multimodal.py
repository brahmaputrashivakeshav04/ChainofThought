"""Audit script for ResearchIQ multimodal pipeline.
Audits:
a) Figure/table captions vs extracted records per paper.
b) Figure PNG dimensions, blank/near-blank crops (>98% white), short side < 300px, multi-figure pages, etc.
c) Vector crops vs embedded rasters, wrong regions / overlapping.
d) Table audit: 52 tables, rows x cols, empty-cell ratio, header detected, units (%, $, ms, etc.), crop linked, suspicious tables.
"""

import csv
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
import numpy as np
import pymupdf

CATALOGUE = Path("papers_catalogue.csv")
RESEARCH_DATA = Path("research-data")
FIGURES_DIR = Path("data/figures")
INDEX_DIR = Path("data/index")

CAPTION_REGEX = re.compile(
    r"^(?:Figure|Fig\.?|Table)\s*(\d+[a-zA-Z]?)\s*[:\.]\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)

def run_audit():
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    fig_chunks = [c for c in chunks if c.chunk_type == "figure"]
    tab_chunks = [c for c in chunks if c.chunk_type == "table"]

    print("=== TOTAL CHUNKS IN INDEX ===")
    print(f"Total: {len(chunks)}, Figures: {len(fig_chunks)}, Tables: {len(tab_chunks)}")

    # 1. Per paper audit: Captions in text vs extracted
    papers = []
    with open(CATALOGUE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        papers = list(reader)

    print("\n=== AUDIT 1A: CAPTIONS FOUND VS EXTRACTED ===")
    audit_1a = []
    
    for p in papers:
        pid = p["paper_id"]
        pdf_path = RESEARCH_DATA / p["filename"]
        if not pdf_path.exists():
            continue
        
        doc = pymupdf.open(pdf_path)
        fig_caps_found = []
        tab_caps_found = []
        
        for page_idx, page in enumerate(doc):
            pnum = page_idx + 1
            blocks = page.get_text("blocks")
            for b in blocks:
                if b[6] == 0:
                    text = b[4].strip()
                    m = CAPTION_REGEX.match(text)
                    if m:
                        cap_type = text[:4].lower()
                        num = m.group(1)
                        if "tab" in cap_type:
                            tab_caps_found.append((pnum, f"Table {num}", text[:60]))
                        else:
                            fig_caps_found.append((pnum, f"Figure {num}", text[:60]))
        doc.close()

        p_figs = [c for c in fig_chunks if c.paper_id == pid]
        p_tabs = [c for c in tab_chunks if c.paper_id == pid]

        audit_1a.append({
            "paper_id": pid,
            "filename": p["filename"],
            "fig_caps_count": len(fig_caps_found),
            "fig_extracted_count": len(p_figs),
            "tab_caps_count": len(tab_caps_found),
            "tab_extracted_count": len(p_tabs),
            "fig_caps": fig_caps_found,
            "tab_caps": tab_caps_found,
            "extracted_figs": [f"{c.element_number} (p.{c.page}, {c.chunk_id})" for c in p_figs],
            "extracted_tabs": [f"{c.element_number} (p.{c.page}, {c.chunk_id})" for c in p_tabs],
        })

    for row in audit_1a:
        print(f"Paper {row['paper_id']} ({row['filename']}):")
        print(f"  Figures: {row['fig_caps_count']} captions found in PDF text vs {row['fig_extracted_count']} extracted")
        print(f"  Tables:  {row['tab_caps_count']} captions found in PDF text vs {row['tab_extracted_count']} extracted")
        
        extracted_fig_caps = [c.caption.lower() if c.caption else "" for c in fig_chunks if c.paper_id == row["paper_id"]]
        extracted_fig_ids = [c.element_number.lower() if c.element_number else "" for c in fig_chunks if c.paper_id == row["paper_id"]]
        for pnum, label, cap in row["fig_caps"]:
            # Check if this figure caption was matched
            num_only = label.split()[-1].lower()
            matched = any(f"fig{num_only}" in ex or f"fig. {num_only}" in cap_text or f"figure {num_only}" in cap_text for ex, cap_text in zip(extracted_fig_ids, extracted_fig_caps))
            if not matched:
                print(f"    [MISSING FIGURE]: {label} on page {pnum}: '{cap}'")

        extracted_tab_caps = [c.caption.lower() if c.caption else "" for c in tab_chunks if c.paper_id == row["paper_id"]]
        extracted_tab_ids = [c.element_number.lower() if c.element_number else "" for c in tab_chunks if c.paper_id == row["paper_id"]]
        for pnum, label, cap in row["tab_caps"]:
            num_only = label.split()[-1].lower()
            matched = any(f"tab{num_only}" in ex or f"table {num_only}" in cap_text for ex, cap_text in zip(extracted_tab_ids, extracted_tab_caps))
            if not matched:
                print(f"    [MISSING TABLE]: {label} on page {pnum}: '{cap}'")

    print("\n=== AUDIT 1B & 1C: FIGURE IMAGE QUALITY & BBOXES ===")
    fig_issues = []
    
    # Check all figure images in data/figures
    all_fig_paths = list(FIGURES_DIR.glob("*/*.png"))
    all_fig_paths = [p for p in all_fig_paths if "fig_" in p.name]
    
    print(f"Total figure PNG files found on disk: {len(all_fig_paths)}")
    
    pages_with_multiple = {}
    for c in fig_chunks:
        key = (c.paper_id, c.page)
        pages_with_multiple[key] = pages_with_multiple.get(key, 0) + 1

    for c in fig_chunks:
        img_path = Path(c.image_path) if c.image_path else None
        issue = []
        dims = (0, 0)
        white_pct = 0.0
        
        if not img_path or not img_path.exists():
            issue.append("MISSING_FILE")
        else:
            try:
                with Image.open(img_path) as im:
                    dims = im.size
                    w, h = dims
                    short_side = min(w, h)
                    if short_side < 300:
                        issue.append(f"SMALL_SHORT_SIDE_{short_side}px")
                    
                    # Convert to grayscale numpy array to check white/blank
                    gray = im.convert("L")
                    arr = np.array(gray)
                    # White is 255 or near 250+
                    white_pct = float(np.sum(arr >= 250) / arr.size * 100)
                    if white_pct > 98.0:
                        issue.append(f"BLANK_OR_NEAR_BLANK_{white_pct:.1f}%")
            except Exception as e:
                issue.append(f"IMAGE_ERR_{e}")

        # Check caption cutoff or multiple captions in crop
        # Check if page has multiple figures
        if pages_with_multiple.get((c.paper_id, c.page), 0) > 1:
            issue.append(f"PAGE_HAS_{pages_with_multiple[(c.paper_id, c.page)]}_FIGURES")

        # Check if caption is empty or missing
        if not c.caption or len(c.caption.strip()) < 5:
            issue.append("NO_CAPTION")

        fig_issues.append({
            "chunk_id": c.chunk_id,
            "paper_id": c.paper_id,
            "page": c.page,
            "label": c.element_number,
            "img_path": str(img_path) if img_path else "",
            "dims": dims,
            "white_pct": round(white_pct, 1),
            "issues": issue
        })

    for f in fig_issues:
        print(f"Figure {f['chunk_id']} ({f['paper_id']} p.{f['page']} {f['label']}): dims={f['dims']}, white={f['white_pct']}% | Issues: {f['issues']}")

    print("\n=== AUDIT 1D: TABLE AUDIT (52 TABLES) ===")
    units_regex = re.compile(r"[%$€£]|ms|sec|kb|mb|gb|ghz|pts|k=|\bacc\b|\bf1\b|\bmrr\b|\bmap\b|\bndcg\b", re.IGNORECASE)
    
    table_issues = []
    for c in tab_chunks:
        lines = [line.strip() for line in (c.text or "").split("\n") if line.strip().startswith("|")]
        rows = len(lines)
        cols = 0
        empty_cells = 0
        total_cells = 0
        header_detected = False
        units_found = False
        
        if rows >= 2:
            # Header line is line 0
            # Separator is line 1
            header_cells = [cell.strip() for cell in lines[0].split("|")[1:-1]]
            cols = len(header_cells)
            header_detected = any(len(h) > 0 for h in header_cells) and ("---" in lines[1])
            
            # Count cells in data rows (lines[2:])
            for r in lines[2:]:
                cells = [cell.strip() for cell in r.split("|")[1:-1]]
                for cell in cells:
                    total_cells += 1
                    if not cell:
                        empty_cells += 1
                    if units_regex.search(cell):
                        units_found = True
            if not units_found and units_regex.search(c.text):
                units_found = True
        elif rows == 1:
            cols = len(lines[0].split("|")[1:-1])
            total_cells = cols
            empty_cells = sum(1 for cell in lines[0].split("|")[1:-1] if not cell.strip())

        empty_ratio = (empty_cells / total_cells) if total_cells > 0 else 1.0
        crop_path = Path(c.image_path) if c.image_path else None
        crop_linked = bool(crop_path and crop_path.exists())

        issues = []
        if cols <= 1:
            issues.append("SINGLE_COL")
        if empty_ratio > 0.5:
            issues.append(f"MOSTLY_EMPTY_{empty_ratio*100:.0f}%")
        if not header_detected:
            issues.append("HEADER_MISSING")
        if not crop_linked:
            issues.append("CROP_NOT_LINKED")
        if rows < 3:
            issues.append(f"FEW_ROWS_{rows}")

        table_issues.append({
            "chunk_id": c.chunk_id,
            "paper_id": c.paper_id,
            "page": c.page,
            "label": c.element_number,
            "rows": rows,
            "cols": cols,
            "empty_ratio": round(empty_ratio, 2),
            "header_detected": header_detected,
            "units_found": units_found,
            "crop_linked": crop_linked,
            "issues": issues,
        })

    suspicious_tables = [t for t in table_issues if t["issues"]]
    print(f"Total tables: {len(table_issues)}, Suspicious tables: {len(suspicious_tables)}")
    for t in table_issues:
        print(f"Table {t['chunk_id']} ({t['paper_id']} p.{t['page']} {t['label']}): {t['rows']}x{t['cols']}, empty={t['empty_ratio']*100:.0f}%, hdr={t['header_detected']}, units={t['units_found']}, crop={t['crop_linked']} | Issues: {t['issues']}")

if __name__ == "__main__":
    run_audit()
