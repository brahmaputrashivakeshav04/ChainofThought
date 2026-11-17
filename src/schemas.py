"""ResearchIQ metadata schemas and data models.

Defines Pydantic models for text chunks, figures, tables, search results,
citations, comparison outputs, and evaluation metrics.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    TEXT = "text"
    FIGURE = "figure"
    TABLE = "table"
    ABSTRACT = "abstract"
    SECTION_HEADER = "section_header"
    FIGURE_CAPTION = "figure_caption"
    TABLE_CONTENT = "table_content"
    FORMULA = "formula"
    VISUAL_DESCRIPTION = "visual_description"


class VisualElementType(str, Enum):
    ARCHITECTURE_DIAGRAM = "architecture_diagram"
    CHART_OR_GRAPH = "chart_or_graph"
    TABLE = "table"
    ILLUSTRATION = "illustration"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """Bounding box coordinates on a page (PDF coordinates in points)."""
    x0: float
    y0: float
    x1: float
    y1: float


class FigureRecord(BaseModel):
    """Metadata and link for an extracted figure or diagram."""
    figure_id: str = Field(description="Unique identifier, e.g., P01_p3_fig1")
    paper_id: str = Field(description="Catalog ID of the paper, e.g., P01")
    paper_title: str
    page: int
    caption: str
    bbox: Optional[BoundingBox] = None
    image_path: str = Field(description="Local path to extracted PNG/JPEG")
    visual_type: VisualElementType = VisualElementType.UNKNOWN
    nearby_text: str = Field(default="", description="Text directly preceding and following figure on the page")


class TableRecord(BaseModel):
    """Metadata and extracted structured content for a table."""
    table_id: str = Field(description="Unique identifier, e.g., P01_p5_tab1")
    paper_id: str
    paper_title: str
    page: int
    caption: str
    markdown_content: str = Field(description="Markdown-rendered representation of the table")
    csv_path: Optional[str] = None
    nearby_text: str = Field(default="", description="Contextual explanation of table in adjacent paragraphs")


class TextChunk(BaseModel):
    """Searchable representation of document text, figures, and tables with rich provenance."""
    chunk_id: str = Field(description="Unique identifier, e.g., P01_p2_c001")
    paper_id: str
    paper_title: str
    page: int
    section: str = Field(default="Main Text")
    text: str
    chunk_type: str = Field(default="text")
    element_number: Optional[str] = Field(default=None, description="e.g. 'Figure 1', 'Table 2'")
    caption: Optional[str] = Field(default=None, description="Extracted caption if chunk_type is figure/table")
    image_path: Optional[str] = Field(default=None, description="Path to rendered figure/table crop if applicable")
    bbox: Optional[BoundingBox] = Field(default=None, description="Bounding box if applicable")
    nearby_text: Optional[str] = Field(default=None, description="Surrounding contextual text for figures/tables")
    linked_figures: List[str] = Field(default_factory=list, description="IDs of figures mentioned in or near this chunk")
    linked_tables: List[str] = Field(default_factory=list, description="IDs of tables mentioned in or near this chunk")
    parent_chunk_id: Optional[str] = Field(default=None, description="Linked parent figure chunk_id if visual_description")
    figure_number: Optional[str] = Field(default=None, description="e.g. 'Figure 1'")
    visual_description: Optional[Dict[str, Any]] = Field(default=None, description="Structured visual description if available")
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def title(self) -> str:
        """Alias for paper_title."""
        return self.paper_title

    def __getattr__(self, name: str) -> Any:
        # Fallback for fields added across iterations
        if name in {"visual_description", "parent_chunk_id", "figure_number"}:
            return None
        return super().__getattr__(name)


class SearchResult(BaseModel):
    """Retrieved candidate after dense, lexical, or hybrid scoring."""
    chunk: TextChunk
    score: float
    rank: int
    retrieval_method: str = Field(description="'dense', 'bm25', 'hybrid', or 'reranked'")


class Citation(BaseModel):
    """Traceable citation format required by hackathon spec: [paper_id, p.X, Fig/Table Y]."""
    paper_id: str
    page: int
    element_type: Optional[str] = Field(default=None, description="'section', 'fig', 'table', or None")
    element_label: Optional[str] = Field(default=None, description="e.g. 'Fig 2', 'Table 1', 'Section 3.2'")
    quote_snippet: Optional[str] = Field(default=None, description="Short supporting text quote directly from paper")

    def format_bracket(self) -> str:
        """Render standardized citation string."""
        if self.element_label:
            return f"[{self.paper_id}, p.{self.page}, {self.element_label}]"
        return f"[{self.paper_id}, p.{self.page}]"


class Claim(BaseModel):
    """An individual assertion with its supporting citations."""
    text: str
    citations: List[str] = Field(default_factory=list, description="Raw chunk_ids supporting this claim")
    formatted_citations: List[str] = Field(default_factory=list, description="Readable citations e.g. [P05, p.6, Fig 2]")


class AnswerResult(BaseModel):
    """Grounded question answering result with verified provenance and security flags."""
    question: str
    answer: str
    claims: List[Claim] = Field(default_factory=list)
    formatted_citations: List[str] = Field(default_factory=list)
    evidence_sufficient: bool = True
    sources: List[SearchResult] = Field(default_factory=list)
    notes: Optional[str] = None
    has_injection_flag: bool = False
    unsupported_citations_dropped: List[str] = Field(default_factory=list)
    debug_info: Optional[Dict[str, Any]] = None

    @property
    def citations(self) -> List[str]:
        """Convenience alias for formatted_citations."""
        return self.formatted_citations


class QAResponse(BaseModel):
    """Grounded question answering response model."""
    question: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    supporting_chunks: List[TextChunk] = Field(default_factory=list)
    insufficient_evidence: bool = False
    warning: Optional[str] = None


class VisualQAResponse(BaseModel):
    """Visual question answering response model for diagrams and charts."""
    question: str
    figure_id: str
    paper_id: str
    page: int
    analysis: str
    labels_and_trends: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    insufficient_evidence: bool = False


class ResearchGapItem(BaseModel):
    """Identified research gap with strict AI-generated labeling."""
    gap_title: str
    description: str
    basis_papers: List[str] = Field(description="List of paper_ids evidencing this gap")
    is_ai_generated: bool = Field(default=True, description="Strictly True to distinguish from paper findings")


class CrossPaperComparison(BaseModel):
    """Structured comparison across multiple papers."""
    paper_ids: List[str]
    objectives_comparison: Dict[str, str]
    methodology_comparison: Dict[str, str]
    benchmarks_and_metrics: Dict[str, str]
    tradeoffs_and_limitations: Dict[str, str]
    synthesis_summary: str
    research_gaps: List[ResearchGapItem] = Field(default_factory=list)


class EvalMetrics(BaseModel):
    """Information retrieval evaluation metrics."""
    recall_at_k: Dict[int, float]
    mrr: float
    hit_rate_at_k: Dict[int, float]
    faithfulness_score: Optional[float] = None
    faithfulness_feedback: Optional[str] = None
