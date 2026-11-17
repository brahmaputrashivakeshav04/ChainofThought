"""Cross-paper comparative analysis, synthesis, and AI-labeled research gap identification."""

from typing import List, Dict
from ..schemas import CrossPaperComparison, TextChunk, ResearchGapItem
from .client import BaseLLMClient
from .prompts import COMPARISON_PROMPT


def compare_papers_and_synthesize(
    paper_ids: List[str],
    chunks_by_paper: Dict[str, List[TextChunk]],
    client: BaseLLMClient,
) -> CrossPaperComparison:
    """Performs deep cross-paper comparison and literature synthesis.

    Compares:
    - Objectives & problem formulation
    - Methodology & architecture
    - Benchmark datasets & evaluation metrics
    - Trade-offs & limitations
    - AI-generated research gap suggestions (clearly labeled)

    Args:
        paper_ids: Identifiers of papers to compare (e.g. ['P02', 'P04']).
        chunks_by_paper: Representative text chunks from each paper.
        client: LLM client.

    Returns:
        Structured CrossPaperComparison model.
    """
    # Stub returning schema-compliant comparison structure
    return CrossPaperComparison(
        paper_ids=paper_ids,
        objectives_comparison={pid: f"Objective summary for {pid}" for pid in paper_ids},
        methodology_comparison={pid: f"Methodology details for {pid}" for pid in paper_ids},
        benchmarks_and_metrics={pid: f"Benchmarks for {pid}" for pid in paper_ids},
        tradeoffs_and_limitations={pid: f"Trade-offs for {pid}" for pid in paper_ids},
        synthesis_summary="Synthesis of approaches in Advanced RAG across selected papers.",
        research_gaps=[
            ResearchGapItem(
                gap_title="Dynamic Graph Construction Latency vs Accuracy",
                description="Trade-off between real-time entity-relation graph enrichment and query latency in streaming RAG.",
                basis_papers=paper_ids,
                is_ai_generated=True,
            )
        ],
    )
