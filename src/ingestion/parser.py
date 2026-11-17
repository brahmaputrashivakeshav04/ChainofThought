"""PDF parser extracting text, structural headers, and page-level layout."""

import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pymupdf

# Common academic section headings
HEADING_REGEX = re.compile(
    r"^(?:(?:[0-9]+\.?)+\s+[A-Z][A-Za-z0-9\s\-]+|"
    r"(?:Abstract|Introduction|Related Work|Background|Methodology|Method|System Architecture|"
    r"Experiments|Experimental Setup|Results|Evaluation|Discussion|Limitations|Conclusion|References))\b",
    re.IGNORECASE,
)


def is_section_heading(text: str, avg_font_size: float = 10.0, span_font_size: float = 10.0) -> bool:
    """Heuristic detector for section headings based on regex and font styling."""
    clean = text.strip()
    if not clean or len(clean) > 90:
        return False
    # If font size is noticeably larger than body text
    if span_font_size > avg_font_size * 1.15:
        return True
    # If matches standard numbered or canonical section names
    if HEADING_REGEX.match(clean):
        return True
    return False


def sort_blocks_reading_order(blocks: List[Tuple], page_width: float) -> List[Tuple]:
    """Sorts text blocks in two-column reading order.

    Blocks completely on left column come first (top to bottom),
    then right column (top to bottom), with full-width header/footer blocks ordered by y0.
    """
    col_split = page_width * 0.52

    def block_key(b: Tuple) -> Tuple[int, float]:
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        is_full_width = (x0 < col_split * 0.8) and (x1 > col_split * 1.1)
        if is_full_width:
            # Full-width spans (e.g. title, abstract spanning both columns)
            return (0, y0)
        elif x1 <= col_split:
            # Left column
            return (1, y0)
        else:
            # Right column
            return (2, y0)

    return sorted(blocks, key=block_key)


def parse_pdf_document(
    file_path: str | Path,
    paper_id: str,
    paper_title: str,
) -> Dict[str, Any]:
    """Parses a PDF into structured pages, extracting text blocks in reading order and section headers.

    Args:
        file_path: Path to the target PDF.
        paper_id: Identifier from catalogue (e.g., 'P01').
        paper_title: Full title of the paper.

    Returns:
        Dictionary containing page-level data, section headings, and warnings.
    """
    path = Path(file_path)
    doc = pymupdf.open(path)
    pages_data = []
    current_section = "Abstract"
    warnings = []

    for idx, page in enumerate(doc):
        page_num = idx + 1
        page_rect = page.rect
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

        if not text_blocks:
            warnings.append(f"Page {page_num} contains no extractable text.")
            pages_data.append({
                "page_number": page_num,
                "raw_text": "",
                "blocks": [],
                "headings": [],
                "word_count": 0,
            })
            continue

        sorted_blocks = sort_blocks_reading_order(text_blocks, page_rect.width)
        page_blocks_data = []
        page_headings = []
        page_text_parts = []

        for b in sorted_blocks:
            x0, y0, x1, y1, text, block_no, _ = b
            clean_text = text.strip()
            first_line = clean_text.splitlines()[0].strip() if clean_text else ""

            if is_section_heading(first_line):
                current_section = first_line
                page_headings.append(current_section)

            page_blocks_data.append({
                "block_no": block_no,
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "text": clean_text,
                "section": current_section,
            })
            page_text_parts.append(clean_text)

        raw_page_text = "\n\n".join(page_text_parts)
        pages_data.append({
            "page_number": page_num,
            "raw_text": raw_page_text,
            "blocks": page_blocks_data,
            "headings": page_headings,
            "word_count": len(raw_page_text.split()),
        })

    doc.close()
    return {
        "paper_id": paper_id,
        "paper_title": paper_title,
        "filename": path.name,
        "total_pages": len(pages_data),
        "pages": pages_data,
        "warnings": warnings,
    }
