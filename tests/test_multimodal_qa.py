"""Unit tests for Multimodal Visual Pipeline: descriptions, indexing, resolution, and Vision QA."""

import json
import tempfile
from pathlib import Path

from src.schemas import TextChunk, ChunkType, SearchResult
from src.llm.client import MockProvider, BaseLLMClient
from src.llm.multimodal_qa import (
    generate_visual_description,
    build_visual_description_chunk,
    answer_visual_question,
    _parse_json_robustly,
)
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.vector_store import VectorIndex
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embeddings import EmbeddingProvider
from src.llm.qa_engine import answer_question, verify_and_format_citations, format_readable_citation


def make_fig_chunk(chunk_id: str = "P05_p5_fig_023", paper_id: str = "P05", page: int = 5) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        paper_title="Test Benchmark Paper",
        page=page,
        section="Visual Element",
        text=f"Visual Element [{chunk_id}]: Recall@k curves comparison.\nContext: Recall increases with k.",
        chunk_type="figure",
        element_number="Figure 1",
        caption="Fig. 1. Recall@k curves for BM25 and dense models.",
        image_path=str(Path("data/figures/P05/fig_1_p5.png")),
        nearby_text="As shown in Figure 1, the curve shows saturation at k=20.",
    )


def test_description_generation_and_caching():
    """Requirement: valid JSON parsing, malformed JSON handled, unreadable fields preserved, cache hit avoids 2nd call."""
    chunk = make_fig_chunk()

    # 1. Valid JSON from model
    valid_json = json.dumps({
        "figure_type": "bar/line/scatter chart",
        "summary": "Recall comparison across methods",
        "components": ["BM25 Curve", "Dense Curve"],
        "text_labels_read": ["Recall@k", "k", "BM25"],
        "connections": [],
        "data_flow": "",
        "chart": {"title": "Recall@k", "trends": "Upward"},
        "unreadable_or_uncertain": ["Tick mark values below 10 are blurry"],
        "confidence": "high",
    })
    mock_client = MockProvider(canned_response=valid_json)

    desc1 = generate_visual_description(chunk, client=mock_client, force_refresh=True)
    assert desc1["description"]["figure_type"] == "bar/line/scatter chart"
    assert "Tick mark values below 10 are blurry" in desc1["description"]["unreadable_or_uncertain"]
    assert mock_client.call_count == 1

    # 2. Cache hit avoids second API call
    desc2 = generate_visual_description(chunk, client=mock_client, force_refresh=False)
    assert desc2["parent_chunk_id"] == chunk.chunk_id
    assert mock_client.call_count == 1, "Cache hit must avoid repeating LLM call"

    # 3. Malformed JSON handled gracefully
    bad_json_client = MockProvider(canned_response="Here is the chart analysis without valid json formatting.")
    desc_bad = generate_visual_description(chunk, client=bad_json_client, force_refresh=True)
    assert desc_bad["description"]["figure_type"] == "other"
    assert "unreadable_or_uncertain" in desc_bad["description"]


def test_storage_separation_and_provenance():
    """Requirement: AI description kept separate from caption/nearby_text; provenance fields present."""
    chunk = make_fig_chunk()
    desc_record = {
        "source": "ai_generated_gemini_vision",
        "model_name": "gemini-2.5-flash",
        "prompt_version": "v1.0",
        "timestamp": "2026-11-17T12:00:00Z",
        "parent_chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "page": chunk.page,
        "figure_number": chunk.element_number,
        "image_path": chunk.image_path,
        "image_hash": "abc123hash",
        "description": {
            "figure_type": "architecture diagram",
            "summary": "AI description summary",
            "components": ["Encoder", "Decoder"],
            "text_labels_read": ["Embedding Vector"],
            "connections": ["Encoder -> Decoder"],
            "data_flow": "Data flow from query",
            "chart": {},
            "unreadable_or_uncertain": [],
            "confidence": "high",
        },
    }

    desc_chunk = build_visual_description_chunk(chunk, desc_record)

    # Assert provenance fields
    assert desc_chunk.chunk_type == "visual_description"
    assert desc_chunk.parent_chunk_id == chunk.chunk_id
    assert desc_chunk.metadata["source"] == "ai_generated_gemini_vision"
    assert desc_chunk.metadata["image_hash"] == "abc123hash"

    # Assert strict separation: description chunk is NOT merged into chunk.caption or chunk.nearby_text
    assert chunk.caption == "Fig. 1. Recall@k curves for BM25 and dense models."
    assert "AI-GENERATED" not in chunk.caption
    assert "AI-GENERATED" not in chunk.nearby_text
    assert "[AI-GENERATED VISUAL DESCRIPTION" in desc_chunk.text


def test_indexing_and_parent_resolution():
    """Requirement: visual_description chunks appear in index and resolve to parent figure with correct metadata."""
    parent_fig = make_fig_chunk("P05_p5_fig1", "P05", 5)
    desc_record = {
        "source": "ai_generated_gemini_vision",
        "model_name": "gemini-2.5-flash",
        "prompt_version": "v1.0",
        "parent_chunk_id": parent_fig.chunk_id,
        "image_hash": "hash123",
        "description": {
            "figure_type": "bar/line/scatter chart",
            "summary": "Visual description containing UniqueVisualTerm42.",
            "components": ["UniqueComponentAlpha"],
            "text_labels_read": ["UniqueLabelBeta"],
            "connections": [],
            "data_flow": "",
            "chart": {},
            "unreadable_or_uncertain": [],
            "confidence": "high",
        },
    }
    desc_chunk = build_visual_description_chunk(parent_fig, desc_record)

    # Build small in-memory indexes
    vec_index = VectorIndex()
    bm25_index = BM25Index()
    embedder = EmbeddingProvider()

    all_chunks = [parent_fig, desc_chunk]
    import numpy as np
    dummy_embeddings = np.random.randn(len(all_chunks), 384).astype(np.float32)

    vec_index.add_chunks(all_chunks, dummy_embeddings)
    bm25_index.add_chunks(all_chunks)

    retriever = HybridRetriever(
        vector_index=vec_index,
        bm25_index=bm25_index,
        embedding_provider=embedder,
    )

    # Retrieval: query using a term found ONLY in the visual description ("UniqueVisualTerm42")
    results = retriever.retrieve("UniqueVisualTerm42", top_k=5)
    assert len(results) > 0, "Query with description-only term must retrieve candidate"

    # Verify that retrieved chunk was resolved to parent figure chunk!
    top_chunk = results[0].chunk
    assert top_chunk.chunk_id == parent_fig.chunk_id, f"Must resolve to parent figure {parent_fig.chunk_id}, got {top_chunk.chunk_id}"
    assert top_chunk.chunk_type == "figure"
    assert top_chunk.paper_id == "P05"
    assert top_chunk.page == 5
    assert top_chunk.element_number == "Figure 1"
    assert top_chunk.image_path == parent_fig.image_path


def test_vision_qa_arguments_and_uncertainty():
    """Requirement: image path, caption, nearby text and description passed to provider; unreadable produces uncertainty."""
    parent_fig = make_fig_chunk()
    desc_data = {
        "description": {
            "figure_type": "bar/line/scatter chart",
            "summary": "Pre-generated description",
            "components": ["BM25", "Dense"],
        }
    }

    class InspectingMock(BaseLLMClient):
        def __init__(self):
            self.received_prompt = None
            self.received_images = None

        def generate(self, prompt, system=None, images=None) -> str:
            self.received_prompt = prompt
            self.received_images = images
            return (
                "1. [PAPER CAPTION & TEXT STATEMENT]: The caption discusses Recall@k.\n"
                "2. [DIRECT VISUAL OBSERVATION]: Cannot be determined from the image: micro-legend font is unreadable.\n"
                "3. [AI-GENERATED DESCRIPTION DETAILS]: Pre-generated summary notes upward trend."
            )

    inspecting_client = InspectingMock()
    ans, was_cached = answer_visual_question(
        question="What is the micro-legend font value?",
        parent_chunk=parent_fig,
        description_data=desc_data,
        client=inspecting_client,
        force_refresh=True,
    )

    # Assert that image path, caption, nearby text, and description were all passed
    assert inspecting_client.received_images is not None
    assert str(parent_fig.image_path) in str(inspecting_client.received_images[0])
    assert parent_fig.caption in inspecting_client.received_prompt
    assert parent_fig.nearby_text in inspecting_client.received_prompt
    assert "Pre-generated description" in inspecting_client.received_prompt

    # Assert uncertainty statement present
    assert "cannot be determined from the image" in ans.lower()
    assert "[PAPER CAPTION & TEXT STATEMENT]" in ans
    assert "[DIRECT VISUAL OBSERVATION]" in ans


def test_citations_resolve_to_parent_figure_and_drop_fabricated():
    """Requirement: final citation is the parent figure, never a description chunk id; fabricated ids dropped."""
    parent_fig = make_fig_chunk("P05_p5_fig1", "P05", 5)
    valid_map = {"P05_p5_fig1": parent_fig}

    claims_data = [
        {
            "text": "The recall curve peaks at k=20.",
            "citations": ["P05_p5_fig1_visdesc", "fabricated_chunk_999"],
        }
    ]

    verified_claims, formatted_cits, unsupported_cits = verify_and_format_citations(claims_data, valid_map)

    # Assert description chunk id resolved to parent figure chunk id
    assert "P05_p5_fig1" in verified_claims[0].citations
    assert "P05_p5_fig1_visdesc" not in verified_claims[0].citations

    # Assert fabricated chunk id dropped
    assert "fabricated_chunk_999" in unsupported_cits
    assert "fabricated_chunk_999" not in verified_claims[0].citations

    # Assert format is [P05, p.5, Fig. 1]
    assert len(formatted_cits) == 1
    assert "[P05, p.5, Figure 1]" in formatted_cits[0] or "[P05, p.5, Fig. 1]" in formatted_cits[0]
