"""Unit tests for Phase 1 Ingestion, Caption Regex, Metadata Completeness, and Index Roundtrip."""

import tempfile
from pathlib import Path
import numpy as np

from src.ingestion.visual_extractor import CAPTION_REGEX, extract_visual_elements
from src.ingestion.parser import parse_pdf_document
from src.ingestion.chunker import chunk_document
from src.retrieval.vector_store import VectorIndex
from src.retrieval.bm25_index import BM25Index
from src.schemas import TextChunk, BoundingBox


def test_caption_regex():
    """Tests that caption regex detects true figure/table captions and ignores body sentences."""
    # True captions
    valid_captions = [
        "Figure 1: Deep GraphRAG architecture.",
        "Fig. 2. Recall@k curves for BM25.",
        "Fig. 3. Heatmap of retrieval methods.",
        "Table 1: Benchmark results on text-and-table datasets.",
        "Table 4. Comparison of latency across approaches.",
    ]
    for cap in valid_captions:
        match = CAPTION_REGEX.match(cap)
        assert match is not None, f"Failed to match valid caption: {cap}"

    # Body sentences starting with 'Figure' without colon or dot
    body_sentences = [
        "Figure 2 provides a side-by-side comparison across all metrics.",
        "Figure 1 illustrates the overall process of our proposed framework.",
        "Table 3 lists all hyperparameters used in the experiments.",
    ]
    for sent in body_sentences:
        match = CAPTION_REGEX.match(sent)
        assert match is None, f"Incorrectly matched body text as caption: {sent}"


def test_chunk_metadata_completeness():
    """Verifies that every generated chunk has paper_id, page, section, and chunk_type."""
    parsed_stub = {
        "paper_id": "P05",
        "paper_title": "Benchmarking Retrieval",
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {"text": "Introduction to advanced retrieval methods and hybrid search.", "section": "1. Introduction"}
                ],
            }
        ],
    }
    chunks = chunk_document(parsed_stub, figures=[], tables=[])
    assert len(chunks) > 0

    for c in chunks:
        assert c.chunk_id, "Missing chunk_id"
        assert c.paper_id == "P05", "Missing or incorrect paper_id"
        assert c.page == 1, "Missing page number"
        assert c.chunk_type in ["text", "figure", "table"], f"Invalid chunk_type: {c.chunk_type}"
        assert c.section, "Missing section"


def test_figure_caption_linking():
    """Extracts figures and tables from a real sample PDF and validates links."""
    pdf_path = Path("research-data/2604.01733v1.pdf")
    if not pdf_path.exists():
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        figs, tabs = extract_visual_elements(
            pdf_path,
            paper_id="P05",
            paper_title="From BM25 to Corrective RAG",
            output_figures_dir=Path(tmpdir),
        )

        assert len(figs) > 0, "Should extract at least one figure"
        assert len(tabs) > 0, "Should extract at least one table"

        # Check figure properties
        sample_fig = figs[0]
        assert sample_fig.paper_id == "P05"
        assert sample_fig.page >= 1
        assert "fig" in sample_fig.caption.lower() or "figure" in sample_fig.caption.lower()
        assert Path(sample_fig.image_path).exists(), f"Rendered crop does not exist: {sample_fig.image_path}"

        # Check table properties
        sample_tab = tabs[0]
        assert sample_tab.paper_id == "P05"
        assert sample_tab.markdown_content, "Table markdown content should not be empty"


def test_index_save_load_round_trip():
    """Verifies that VectorIndex and BM25Index save and load faithfully."""
    chunks = [
        TextChunk(
            chunk_id="P01_p1_c001",
            paper_id="P01",
            paper_title="Scaling Beyond Context",
            page=1,
            section="Abstract",
            text="Multimodal retrieval augmented generation for documents.",
            chunk_type="text",
        ),
        TextChunk(
            chunk_id="P02_p1_c001",
            paper_id="P02",
            paper_title="Deep GraphRAG",
            page=1,
            section="Introduction",
            text="Hierarchical graph indexing and adaptive integration.",
            chunk_type="text",
        ),
    ]

    embeddings = np.array([
        [0.8, 0.6] + [0.0] * 382,
        [0.0, 1.0] + [0.0] * 382,
    ], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        # 1. Vector Index
        v_idx = VectorIndex(dimension=384)
        v_idx.add_chunks(chunks, embeddings)
        v_idx.save(save_dir)

        loaded_v = VectorIndex(dimension=384)
        loaded_v.load(save_dir)
        assert len(loaded_v.chunks) == 2
        assert loaded_v.vectors is not None
        assert loaded_v.vectors.shape == (2, 384)

        # Query vector search
        q_vec = np.array([0.8, 0.6] + [0.0] * 382, dtype=np.float32)
        res = loaded_v.search(q_vec, top_k=1)
        assert len(res) == 1
        assert res[0].chunk.chunk_id == "P01_p1_c001"

        # 2. BM25 Index
        bm25 = BM25Index()
        bm25.build_index(chunks)
        bm25.save(save_dir)

        loaded_bm25 = BM25Index()
        loaded_bm25.load(save_dir)
        assert len(loaded_bm25.chunks) == 2

        bm25_res = loaded_bm25.search("GraphRAG hierarchical", top_k=1)
        assert len(bm25_res) == 1
        assert bm25_res[0].chunk.chunk_id == "P02_p1_c001"
