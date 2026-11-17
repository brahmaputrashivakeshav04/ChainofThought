"""Visual and tabular element extractor for multimodal research assistance.

Extracts figures, architecture diagrams, charts, and tables from PDFs,
preserving bounding boxes, captions, page numbers, and adjacent text links.
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import pymupdf

from ..schemas import FigureRecord, TableRecord, VisualElementType, BoundingBox
from ..config import FIGURES_DIR, sanitize_error

# Regex to detect figure and table captions
# Matches: "Figure 1:", "Fig. 2.", "Table 1:", "Fig 3:", "TABLE I", "Fig. 1 <en-space>", "Table 16:"
CAPTION_REGEX = re.compile(
    r"^(?P<type>Figure|Fig\.?|Table)\s*[\xa0\u2002\s]*(?P<num>[0-9IVX]+[a-zA-Z]?)(?:[:\.\u2002]|\s*[\n\r]+|\s{2,}|\s*[:\.]\s*)(?P<caption>.*)",
    re.IGNORECASE | re.DOTALL,
)

# Reference regex to find mentions in body text, e.g. "as shown in Figure 2" or "see Table 1"
REF_REGEX = re.compile(
    r"\b(?:as shown in|see|in|from|refer to|illustrated in)\s+(?:figure|fig\.?|table)\s*\d+[a-zA-Z]?\b|"
    r"\b(?:figure|fig\.?|table)\s*\d+[a-zA-Z]?\b",
    re.IGNORECASE,
)


def extract_table_markdown(table_data: List[List[Optional[str]]]) -> str:
    """Converts 2D extracted table cells into clean markdown table text."""
    if not table_data or len(table_data) < 1:
        return ""
    cleaned = [
        [(cell or "").replace("\n", " ").replace("|", "\\|").strip() for cell in row]
        for row in table_data
    ]
    header = cleaned[0]
    if not any(header):
        return ""
    num_cols = len(header)
    md_lines = ["| " + " | ".join(header) + " |"]
    md_lines.append("| " + " | ".join(["---"] * num_cols) + " |")
    for row in cleaned[1:]:
        padded = (row + [""] * num_cols)[:num_cols]
        md_lines.append("| " + " | ".join(padded) + " |")
    return "\n".join(md_lines)


def find_body_references(all_pages_text: List[str], label: str) -> List[str]:
    """Finds sentences across the document that reference a specific Figure or Table."""
    sentences_found = []
    # Search for e.g. "Figure 1" or "Fig. 1"
    parts = label.split()
    if len(parts) >= 2:
        num = parts[1]
        pattern = re.compile(rf"([^.?!;\n]*\b(?:figure|fig\.?|table)\s*{re.escape(num)}\b[^.?!;\n]*)", re.IGNORECASE)
        for page_text in all_pages_text:
            for match in pattern.finditer(page_text):
                sent = match.group(1).strip()
                if sent and len(sent) > 20 and sent not in sentences_found:
                    sentences_found.append(sent)
    return sentences_found[:4]


def extract_visual_elements(
    file_path: str | Path,
    paper_id: str,
    paper_title: str,
    output_figures_dir: Path = FIGURES_DIR,
) -> Tuple[List[FigureRecord], List[TableRecord]]:
    """Extracts all figures, diagrams, charts, and tables from the PDF.

    Saves images under data/figures/{paper_id}/.
    Extracts tables as markdown and renders visual crops.
    Preserves caption links, page numbers, bboxes, and nearby text.
    """
    path = Path(file_path)
    paper_figures_dir = output_figures_dir / paper_id
    paper_figures_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(path)
    all_pages_text = [page.get_text("text") for page in doc]

    figures: List[FigureRecord] = []
    tables: List[TableRecord] = []

    fig_counter = 0
    tab_counter = 0

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        page_rect = page.rect
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

        # Collect table and figure caption candidates on this page
        page_tab_caps = []
        page_fig_caps = []
        for b_idx, b in enumerate(text_blocks):
            t = b[4].strip()
            match = CAPTION_REGEX.match(t)
            if not match:
                continue
            # Filter out dialogue references or non-caption text
            if "question:" in t.lower() or "response:" in t.lower() or len(match.group("caption").strip()) < 3:
                continue
            elem_type = match.group("type").lower()
            num_str = match.group("num")
            caption_text = t.replace("\n", " ").strip()
            if "tab" in elem_type:
                page_tab_caps.append((b_idx, b, f"Table {num_str}", caption_text))
            else:
                page_fig_caps.append((b_idx, b, f"Figure {num_str}", caption_text))

        # 1. Detect and Extract Tables
        table_finder = page.find_tables()
        extracted_tables = table_finder.tables if table_finder else []
        matched_tab_cap_indices = set()

        for tab_obj in extracted_tables:
            tab_counter += 1
            t_bbox = tab_obj.bbox  # (x0, y0, x1, y1)
            raw_tab_data = tab_obj.extract()
            md_content = extract_table_markdown(raw_tab_data)

            # Match nearest unassigned table caption by vertical proximity
            best_caption = f"Table {tab_counter}"
            element_label = f"Table {tab_counter}"
            best_dist = float("inf")
            best_cap_idx = None

            for c_idx, (_, cap_b, cap_label, cap_txt) in enumerate(page_tab_caps):
                if c_idx in matched_tab_cap_indices:
                    continue
                # Vertical distance between table and caption
                dist = min(abs(t_bbox[3] - cap_b[1]), abs(t_bbox[1] - cap_b[3]))
                if dist < best_dist:
                    best_dist = dist
                    best_caption = cap_txt
                    element_label = cap_label
                    best_cap_idx = c_idx

            if best_cap_idx is not None:
                matched_tab_cap_indices.add(best_cap_idx)

            # Render table crop at 200 DPI
            tab_rect = pymupdf.Rect(t_bbox)
            padded_rect = (tab_rect + (-8, -8, 8, 8)) & page_rect
            crop_filename = f"tab_{tab_counter}_p{page_num}.png"
            crop_path = paper_figures_dir / crop_filename

            try:
                pix = page.get_pixmap(clip=padded_rect, dpi=200)
                pix.save(crop_path)
            except Exception:
                crop_path = Path("")

            body_refs = find_body_references(all_pages_text, element_label)
            nearby_str = " | ".join(body_refs) if body_refs else "Table extracted from document."

            tables.append(
                TableRecord(
                    table_id=f"{paper_id}_p{page_num}_tab{tab_counter}",
                    paper_id=paper_id,
                    paper_title=paper_title,
                    page=page_num,
                    caption=best_caption,
                    markdown_content=md_content,
                    csv_path=str(crop_path) if crop_path else None,
                    nearby_text=nearby_str,
                )
            )

        # Handle any table captions that didn't match a find_tables() table (e.g. borderless tables)
        for c_idx, (b_idx, cap_b, cap_label, cap_txt) in enumerate(page_tab_caps):
            if c_idx in matched_tab_cap_indices:
                continue
            tab_counter += 1
            cap_x0, cap_y0, cap_x1, cap_y1 = cap_b[:4]
            # Region directly above caption
            top_y = max(page_rect.y0 + 35, cap_y0 - 200)
            t_bbox = (max(page_rect.x0 + 30, cap_x0 - 20), top_y, min(page_rect.x1 - 30, cap_x1 + 20), cap_y0 - 4)

            # Extract text in that region
            t_rect = pymupdf.Rect(t_bbox) & page_rect
            raw_text = page.get_text("text", clip=t_rect).strip()
            # Simple markdown table from lines
            lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
            md_lines = ["| Content |", "| --- |"] + [f"| {l} |" for l in lines]
            md_content = "\n".join(md_lines)

            crop_filename = f"tab_{tab_counter}_p{page_num}.png"
            crop_path = paper_figures_dir / crop_filename
            try:
                pix = page.get_pixmap(clip=t_rect, dpi=200)
                pix.save(crop_path)
            except Exception:
                crop_path = Path("")

            body_refs = find_body_references(all_pages_text, cap_label)
            nearby_str = " | ".join(body_refs) if body_refs else "Table extracted from document."

            tables.append(
                TableRecord(
                    table_id=f"{paper_id}_p{page_num}_tab{tab_counter}",
                    paper_id=paper_id,
                    paper_title=paper_title,
                    page=page_num,
                    caption=cap_txt,
                    markdown_content=md_content,
                    csv_path=str(crop_path) if crop_path else None,
                    nearby_text=nearby_str,
                )
            )

        # 2. Detect and Extract Figures
        img_info_list = page.get_image_info(xrefs=True)

        for b_idx, cap_b, element_label, caption_text in page_fig_caps:
            fig_counter += 1
            cap_x0, cap_y0, cap_x1, cap_y1 = cap_b[:4]

            # Context text directly preceding/succeeding caption
            preceding = text_blocks[b_idx - 1][4].strip() if b_idx > 0 else ""
            succeeding = text_blocks[b_idx + 1][4].strip() if b_idx + 1 < len(text_blocks) else ""
            body_refs = find_body_references(all_pages_text, element_label)
            nearby_parts = []
            if body_refs:
                nearby_parts.extend(body_refs)
            if preceding and not CAPTION_REGEX.match(preceding):
                nearby_parts.append(preceding[:200])
            if succeeding and not CAPTION_REGEX.match(succeeding):
                nearby_parts.append(succeeding[:200])
            nearby_str = " ... ".join(nearby_parts) if nearby_parts else caption_text

            # Proximity matching for embedded raster images
            matched_imgs = []
            for img_info in img_info_list:
                ibbox = img_info.get("bbox")
                if not ibbox:
                    continue
                w = ibbox[2] - ibbox[0]
                h = ibbox[3] - ibbox[1]
                if w < 40 or h < 40:
                    continue
                vert_dist_above = cap_y0 - ibbox[3]
                vert_dist_below = ibbox[1] - cap_y1
                horiz_overlap = max(0, min(cap_x1 + 30, ibbox[2]) - max(cap_x0 - 30, ibbox[0]))
                horiz_dist = abs((ibbox[0] + ibbox[2]) / 2.0 - (cap_x0 + cap_x1) / 2.0)

                # Matching candidates within vertical threshold and column alignment
                if (0 <= vert_dist_above < 380 or 0 <= vert_dist_below < 380) and (horiz_overlap > 15 or horiz_dist < 160):
                    dist = vert_dist_above if vert_dist_above >= 0 else vert_dist_below
                    matched_imgs.append((ibbox, dist + horiz_dist * 0.3))

            chosen_bbox = None
            if matched_imgs:
                # Sort by distance to this specific caption
                matched_imgs.sort(key=lambda x: x[1])
                primary_bbox = matched_imgs[0][0]
                union_bbox = list(primary_bbox)
                # Union adjacent sub-panels that belong to the same figure
                for other_bbox, _ in matched_imgs[1:]:
                    if abs(other_bbox[1] - primary_bbox[1]) < 90 or abs(other_bbox[3] - primary_bbox[3]) < 90:
                        union_bbox[0] = min(union_bbox[0], other_bbox[0])
                        union_bbox[1] = min(union_bbox[1], other_bbox[1])
                        union_bbox[2] = max(union_bbox[2], other_bbox[2])
                        union_bbox[3] = max(union_bbox[3], other_bbox[3])
                chosen_bbox = tuple(union_bbox)
            else:
                # Vector diagram crop with column awareness
                is_two_col = page_rect.width > 500 and (cap_x1 - cap_x0 < page_rect.width * 0.58)
                if is_two_col:
                    col_x0 = max(page_rect.x0 + 25, cap_x0 - 20)
                    col_x1 = min(page_rect.x1 - 25, cap_x1 + 20)
                else:
                    col_x0 = page_rect.x0 + 35
                    col_x1 = page_rect.x1 - 35

                if cap_y0 > 160:
                    top_y = max(page_rect.y0 + 35, cap_y0 - 280)
                    chosen_bbox = (col_x0, top_y, col_x1, cap_y0 - 5)
                else:
                    chosen_bbox = (col_x0, cap_y1 + 5, col_x1, min(page_rect.y1 - 35, cap_y1 + 280))

            crop_filename = f"fig_{fig_counter}_p{page_num}.png"
            crop_path = paper_figures_dir / crop_filename

            try:
                rect = pymupdf.Rect(chosen_bbox) & page_rect
                if rect.width > 30 and rect.height > 30:
                    pix = page.get_pixmap(clip=rect, dpi=200)
                    pix.save(crop_path)
                else:
                    crop_path = Path("")
            except Exception:
                crop_path = Path("")

            # Visual type classification heuristic
            vtype = VisualElementType.CHART_OR_GRAPH
            lower_cap = caption_text.lower()
            if any(k in lower_cap for k in ["architecture", "overview", "framework", "pipeline", "workflow"]):
                vtype = VisualElementType.ARCHITECTURE_DIAGRAM
            elif any(k in lower_cap for k in ["curve", "plot", "accuracy", "recall", "score", "f1", "ablation", "heatmap"]):
                vtype = VisualElementType.CHART_OR_GRAPH

            figures.append(
                FigureRecord(
                    figure_id=f"{paper_id}_p{page_num}_fig{fig_counter}",
                    paper_id=paper_id,
                    paper_title=paper_title,
                    page=page_num,
                    caption=caption_text,
                    bbox=BoundingBox(
                        x0=chosen_bbox[0], y0=chosen_bbox[1], x1=chosen_bbox[2], y1=chosen_bbox[3]
                    ) if chosen_bbox else None,
                    image_path=str(crop_path),
                    visual_type=vtype,
                    nearby_text=nearby_str,
                )
            )

    doc.close()
    return figures, tables
