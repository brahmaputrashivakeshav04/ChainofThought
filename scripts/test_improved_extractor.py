"""Test script for improved visual extractor logic."""
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import pymupdf
from PIL import Image

CAPTION_REGEX = re.compile(
    r"^(?P<type>Figure|Fig\.?|Table)\s*[\xa0\u2002\s]*(?P<num>[0-9IVX]+[a-zA-Z]?)(?:[:\.\u2002]|\s*[\n\r]+|\s{2,}|\s*[:\.]\s*)(?P<caption>.*)",
    re.IGNORECASE | re.DOTALL,
)

def test_extraction_on_papers():
    papers = [
        ("P01", "research-data/2026.acl-long.204.pdf"),
        ("P02", "research-data/2601.11144v3.pdf"),
        ("P03", "research-data/2602.10271v3.pdf"),
        ("P04", "research-data/2603.24925v3.pdf"),
        ("P05", "research-data/2604.01733v1.pdf"),
        ("P06", "research-data/2604.12352v1.pdf"),
        ("P07", "research-data/2606.18075v1.pdf"),
        ("P08", "research-data/s44163-026-02023-3.pdf"),
    ]

    total_figs = 0
    total_tabs = 0

    for pid, pdf_path in papers:
        doc = pymupdf.open(pdf_path)
        paper_figs = 0
        paper_tabs = 0
        for pidx, page in enumerate(doc):
            pnum = pidx + 1
            page_rect = page.rect
            blocks = page.get_text("blocks")
            text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

            # Table finder
            tf = page.find_tables()
            page_tables = tf.tables if tf else []

            # Find table captions
            tab_caps = []
            fig_caps = []
            for b in text_blocks:
                t = b[4].strip()
                m = CAPTION_REGEX.match(t)
                if not m:
                    continue
                if "question:" in t.lower() or "response:" in t.lower() or len(m.group("caption").strip()) < 3:
                    continue
                elem_type = m.group("type").lower()
                num = m.group("num")
                cap_text = t.replace("\n", " ").strip()
                if "tab" in elem_type:
                    tab_caps.append((b, f"Table {num}", cap_text))
                else:
                    fig_caps.append((b, f"Figure {num}", cap_text))

            # Table count
            paper_tabs += max(len(page_tables), len(tab_caps))

            # Figure count and bounding boxes
            img_info_list = page.get_image_info(xrefs=True)
            for cap_b, elem_label, cap_text in fig_caps:
                paper_figs += 1
                cap_x0, cap_y0, cap_x1, cap_y1 = cap_b[:4]

                # Match nearest image or images
                matched_imgs = []
                for img_info in img_info_list:
                    ibbox = img_info.get("bbox")
                    if not ibbox:
                        continue
                    w = ibbox[2] - ibbox[0]
                    h = ibbox[3] - ibbox[1]
                    if w < 40 or h < 40:
                        continue
                    # Check vertical proximity
                    vert_dist_above = cap_y0 - ibbox[3]
                    vert_dist_below = ibbox[1] - cap_y1
                    # Column overlap
                    horiz_overlap = max(0, min(cap_x1 + 30, ibbox[2]) - max(cap_x0 - 30, ibbox[0]))
                    if (0 <= vert_dist_above < 380 or 0 <= vert_dist_below < 380) and (horiz_overlap > 20 or abs(ibbox[0] - cap_x0) < 150):
                        matched_imgs.append((ibbox, vert_dist_above if vert_dist_above >= 0 else vert_dist_below))

                if matched_imgs:
                    # Sort by distance
                    matched_imgs.sort(key=lambda x: x[1])
                    # If multiple subpanels form the figure, take their union if they are close
                    primary_bbox = matched_imgs[0][0]
                    union_bbox = list(primary_bbox)
                    for other_bbox, dist in matched_imgs[1:]:
                        if abs(other_bbox[1] - primary_bbox[1]) < 80 or abs(other_bbox[3] - primary_bbox[3]) < 80:
                            union_bbox[0] = min(union_bbox[0], other_bbox[0])
                            union_bbox[1] = min(union_bbox[1], other_bbox[1])
                            union_bbox[2] = max(union_bbox[2], other_bbox[2])
                            union_bbox[3] = max(union_bbox[3], other_bbox[3])
                    chosen_bbox = tuple(union_bbox)
                else:
                    # Vector crop
                    is_two_col = page_rect.width > 500 and (cap_x1 - cap_x0 < page_rect.width * 0.6)
                    if is_two_col:
                        col_x0 = max(page_rect.x0 + 25, cap_x0 - 15)
                        col_x1 = min(page_rect.x1 - 25, cap_x1 + 15)
                    else:
                        col_x0 = page_rect.x0 + 35
                        col_x1 = page_rect.x1 - 35

                    if cap_y0 > 160:
                        top_y = max(page_rect.y0 + 35, cap_y0 - 280)
                        chosen_bbox = (col_x0, top_y, col_x1, cap_y0 - 5)
                    else:
                        chosen_bbox = (col_x0, cap_y1 + 5, col_x1, min(page_rect.y1 - 35, cap_y1 + 280))

        doc.close()
        print(f"Paper {pid}: {paper_figs} figures, {paper_tabs} tables extracted")
        total_figs += paper_figs
        total_tabs += paper_tabs

    print(f"\nTOTAL: {total_figs} figures, {total_tabs} tables across all 8 papers")

if __name__ == '__main__':
    test_extraction_on_papers()
