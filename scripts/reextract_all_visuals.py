"""Re-extracts figures and tables with the corrected 200 DPI and proximity logic."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pymupdf
from PIL import Image
import numpy as np

from src.ingestion.visual_extractor import extract_visual_elements
from src.config import FIGURES_DIR, RESEARCH_DATA_DIR, CATALOGUE_PATH

with open(CATALOGUE_PATH, "r", encoding="utf-8") as f:
    papers = list(csv.DictReader(f))

all_figs = []
all_tabs = []

for p in papers:
    pid = p["paper_id"]
    pdf_p = RESEARCH_DATA_DIR / p["filename"]
    print(f"Extracting visuals for {pid} ({p['filename']})...")
    figs, tabs = extract_visual_elements(
        file_path=pdf_p,
        paper_id=pid,
        paper_title=p["title"],
        output_figures_dir=FIGURES_DIR,
    )
    all_figs.extend(figs)
    all_tabs.extend(tabs)
    print(f"  -> {len(figs)} figures, {len(tabs)} tables")

print(f"\nTOTAL: {len(all_figs)} figures, {len(all_tabs)} tables extracted.")

# Verify image dimensions
small_short_side = 0
for f in all_figs:
    p = Path(f.image_path)
    if p.exists():
        with Image.open(p) as im:
            w, h = im.size
            if min(w, h) < 300:
                small_short_side += 1
                # print small ones
                print(f"  Short side < 300px: {f.figure_id} dims=({w}, {h})")

print(f"Figures with short side < 300px: {small_short_side} / {len(all_figs)}")
