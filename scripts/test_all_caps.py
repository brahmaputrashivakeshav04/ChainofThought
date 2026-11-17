import re
import csv
from pathlib import Path
import pymupdf

CATALOGUE = Path("papers_catalogue.csv")
RESEARCH_DATA = Path("research-data")

CAPTION_REGEX = re.compile(
    r"^(?P<type>Figure|Fig\.?|Table)\s*[\xa0\u2002\s]*(?P<num>[0-9IVX]+[a-zA-Z]?)(?:\s*[:\.]|\s+[\u2002\xa0]|\s{2,}|\s+(?=[A-Z]))\s*(?P<caption>.*)",
    re.IGNORECASE | re.DOTALL,
)

with open(CATALOGUE, "r", encoding="utf-8") as f:
    papers = list(csv.DictReader(f))

for p in papers:
    pid = p["paper_id"]
    pdf_path = RESEARCH_DATA / p["filename"]
    doc = pymupdf.open(pdf_path)
    fig_caps = []
    tab_caps = []
    for page_idx, page in enumerate(doc):
        blocks = page.get_text("blocks")
        for b in blocks:
            if b[6] == 0:
                text = b[4].strip()
                m = CAPTION_REGEX.match(text)
                if m:
                    t = m.group("type").lower()
                    num = m.group("num")
                    cap = m.group("caption").replace("\n", " ")[:40]
                    # Filter out dialogue references like Question:... Response:...
                    if "question:" in text.lower() or "response:" in text.lower() or len(cap) < 3:
                        continue
                    if "tab" in t:
                        tab_caps.append((page_idx + 1, num, cap))
                    else:
                        fig_caps.append((page_idx + 1, num, cap))
    doc.close()
    print(f"{pid} ({p['filename']}): {len(fig_caps)} figures, {len(tab_caps)} tables")
    for f in fig_caps:
        print(f"   Fig {f[1]} (p.{f[0]}): {f[2]}")
    for t in tab_caps:
        print(f"   Tab {t[1]} (p.{t[0]}): {t[2]}")
