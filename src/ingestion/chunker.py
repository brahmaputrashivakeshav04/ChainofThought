"""Document chunker creating searchable units with structural and visual links."""

import re
from typing import List, Dict, Any, Tuple
from ..schemas import TextChunk, ChunkType, FigureRecord, TableRecord


def link_visual_references(
    chunk_text: str,
    figures: List[FigureRecord],
    tables: List[TableRecord],
) -> Tuple[List[str], List[str]]:
    """Detects mentions of figures or tables in text chunk."""
    linked_figs = []
    linked_tabs = []

    for fig in figures:
        # Match e.g. "Figure 1", "Fig. 1", "Fig 1"
        parts = fig.caption.split(":")
        label = parts[0].strip() if parts else ""
        if label:
            # Extract number
            num_match = re.search(r"\d+[a-zA-Z]?", label)
            if num_match:
                num = num_match.group(0)
                pattern = re.compile(rf"\b(?:figure|fig\.?)\s*{re.escape(num)}\b", re.IGNORECASE)
                if pattern.search(chunk_text) and fig.figure_id not in linked_figs:
                    linked_figs.append(fig.figure_id)

    for tab in tables:
        parts = tab.caption.split(":")
        label = parts[0].strip() if parts else ""
        if label:
            num_match = re.search(r"\d+[a-zA-Z]?", label)
            if num_match:
                num = num_match.group(0)
                pattern = re.compile(rf"\b(?:table|tab\.?)\s*{re.escape(num)}\b", re.IGNORECASE)
                if pattern.search(chunk_text) and tab.table_id not in linked_tabs:
                    linked_tabs.append(tab.table_id)

    return linked_figs, linked_tabs


def chunk_document(
    parsed_doc: Dict[str, Any],
    figures: List[FigureRecord],
    tables: List[TableRecord],
    target_chunk_tokens: int = 500,
    overlap_tokens: int = 70,
) -> List[TextChunk]:
    """Splits parsed PDF into merged contextual chunks (400-600 tokens) and adds visual/table chunks.

    Consecutive text blocks within the same paper and section are merged until reaching ~400-600 tokens,
    preserving page provenance (recording page ranges) and relational visual links.
    Figures and tables remain dedicated element chunks.
    """
    chunks: List[TextChunk] = []
    paper_id = parsed_doc["paper_id"]
    paper_title = parsed_doc["paper_title"]
    chunk_counter = 0

    # 1. Collect all text blocks in reading order with their section & page
    all_blocks = []
    for page_data in parsed_doc["pages"]:
        p_num = page_data["page_number"]
        for b in page_data.get("blocks", []):
            txt = b.get("text", "").strip()
            if txt and len(txt.split()) >= 3:
                all_blocks.append({
                    "text": txt,
                    "section": b.get("section", "Main Text"),
                    "page": p_num,
                })

    # 2. Merge consecutive blocks within the same section up to ~400-600 tokens (approx 350-450 words)
    target_words = int(target_chunk_tokens / 1.3)
    overlap_words = int(overlap_tokens / 1.3)

    current_section = None
    accumulated_words: List[str] = []
    accumulated_pages: List[int] = []

    def flush_chunk(words_to_flush: List[str], pages_to_flush: List[int], sec: str):
        nonlocal chunk_counter
        if not words_to_flush:
            return
        chunk_counter += 1
        chunk_str = " ".join(words_to_flush)
        start_page = pages_to_flush[0] if pages_to_flush else 1
        end_page = pages_to_flush[-1] if pages_to_flush else start_page
        chunk_id = f"{paper_id}_p{start_page}_c{chunk_counter:03d}"

        fig_links, tab_links = link_visual_references(chunk_str, figures, tables)
        tok_cnt = int(len(words_to_flush) * 1.3)

        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                paper_title=paper_title,
                page=start_page,
                section=sec,
                text=chunk_str,
                chunk_type="text",
                linked_figures=fig_links,
                linked_tables=tab_links,
                token_count=tok_cnt,
                metadata={"page_range": [start_page, end_page], "end_page": end_page},
            )
        )

    for block in all_blocks:
        b_sec = block["section"]
        b_page = block["page"]
        b_words = block["text"].split()

        # If section changed, flush current accumulator
        if current_section is not None and b_sec != current_section:
            flush_chunk(accumulated_words, accumulated_pages, current_section)
            accumulated_words = []
            accumulated_pages = []

        current_section = b_sec
        accumulated_words.extend(b_words)
        accumulated_pages.append(b_page)

        # While accumulated words exceed target, emit a chunk and keep overlap
        while len(accumulated_words) >= target_words:
            chunk_slice = accumulated_words[:target_words]
            flush_chunk(chunk_slice, accumulated_pages, current_section)
            accumulated_words = accumulated_words[target_words - overlap_words:]
            if accumulated_pages:
                accumulated_pages = [accumulated_pages[-1]]

    # Flush remaining words in the final section
    if accumulated_words and current_section:
        flush_chunk(accumulated_words, accumulated_pages, current_section)

    # 2. Specialized Figure chunks (caption + nearby text + crop reference)
    for fig in figures:
        chunk_counter += 1
        fig_chunk_id = f"{paper_id}_p{fig.page}_fig_{chunk_counter:03d}"
        fig_text = (
            f"Visual Element [{fig.figure_id}]: {fig.caption}\n"
            f"Contextual explanation and surrounding text: {fig.nearby_text}"
        )
        chunks.append(
            TextChunk(
                chunk_id=fig_chunk_id,
                paper_id=paper_id,
                paper_title=paper_title,
                page=fig.page,
                section="Visual Element",
                text=fig_text,
                chunk_type="figure",
                element_number=fig.figure_id,
                caption=fig.caption,
                image_path=fig.image_path,
                bbox=fig.bbox,
                nearby_text=fig.nearby_text,
                linked_figures=[fig.figure_id],
                token_count=len(fig_text.split()),
            )
        )

    # 3. Specialized Table chunks (caption + markdown content + nearby text)
    for tab in tables:
        chunk_counter += 1
        tab_chunk_id = f"{paper_id}_p{tab.page}_tab_{chunk_counter:03d}"
        tab_text = (
            f"Table [{tab.table_id}]: {tab.caption}\n\n"
            f"Table Data:\n{tab.markdown_content}\n\n"
            f"Contextual explanation: {tab.nearby_text}"
        )
        chunks.append(
            TextChunk(
                chunk_id=tab_chunk_id,
                paper_id=paper_id,
                paper_title=paper_title,
                page=tab.page,
                section="Table Element",
                text=tab_text,
                chunk_type="table",
                element_number=tab.table_id,
                caption=tab.caption,
                image_path=tab.csv_path,
                nearby_text=tab.nearby_text,
                linked_tables=[tab.table_id],
                token_count=len(tab_text.split()),
            )
        )

    return chunks
