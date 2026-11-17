"""Unit tests for Pydantic metadata schemas and citations."""

from src.schemas import TextChunk, ChunkType, Citation, FigureRecord, VisualElementType


def test_citation_formatting():
    c1 = Citation(paper_id="P01", page=3)
    assert c1.format_bracket() == "[P01, p.3]"

    c2 = Citation(paper_id="P02", page=5, element_label="Fig 2")
    assert c2.format_bracket() == "[P02, p.5, Fig 2]"


def test_text_chunk_schema():
    chunk = TextChunk(
        chunk_id="P01_p1_c001",
        paper_id="P01",
        paper_title="Scaling Beyond Context",
        page=1,
        section="Abstract",
        text="A survey on multimodal RAG...",
        chunk_type=ChunkType.ABSTRACT,
        linked_figures=["P01_p2_fig1"],
    )
    assert chunk.paper_id == "P01"
    assert "P01_p2_fig1" in chunk.linked_figures


def test_figure_record_schema():
    fig = FigureRecord(
        figure_id="P02_p3_fig1",
        paper_id="P02",
        paper_title="Deep GraphRAG",
        page=3,
        caption="Figure 1: Hierarchical indexing workflow.",
        image_path="data/figures/P02_p3_fig1.png",
        visual_type=VisualElementType.ARCHITECTURE_DIAGRAM,
        nearby_text="As shown in Figure 1, the pipeline clusters entities...",
    )
    assert fig.visual_type == VisualElementType.ARCHITECTURE_DIAGRAM
