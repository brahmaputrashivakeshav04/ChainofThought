"""Unit tests for BM25 and Hybrid search RRF logic."""

from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.vector_store import VectorIndex
from src.retrieval.embeddings import EmbeddingProvider
from src.schemas import TextChunk, SearchResult


def test_bm25_search():
    index = BM25Index()
    chunks = [
        TextChunk(chunk_id="c1", paper_id="P01", paper_title="T1", page=1, text="GraphRAG hierarchical clustering and entity relation"),
        TextChunk(chunk_id="c2", paper_id="P02", paper_title="T2", page=2, text="BM25 keyword search benchmarking for tabular data"),
    ]
    index.build_index(chunks)

    results = index.search("hierarchical GraphRAG", top_k=2)
    assert len(results) > 0
    assert results[0].chunk.chunk_id == "c1"


def test_reciprocal_rank_fusion():
    c1 = TextChunk(chunk_id="c1", paper_id="P01", paper_title="T1", page=1, text="A")
    c2 = TextChunk(chunk_id="c2", paper_id="P02", paper_title="T2", page=2, text="B")

    dense_res = [SearchResult(chunk=c1, score=0.9, rank=1, retrieval_method="dense")]
    bm25_res = [SearchResult(chunk=c2, score=5.0, rank=1, retrieval_method="bm25"), SearchResult(chunk=c1, score=4.0, rank=2, retrieval_method="bm25")]

    retriever = HybridRetriever(
        vector_index=VectorIndex(),
        bm25_index=BM25Index(),
        embedding_provider=EmbeddingProvider(),
    )
    fused = retriever.reciprocal_rank_fusion(dense_res, bm25_res, k=60)

    # c1 is rank 1 in dense and rank 2 in bm25, should have highest score
    assert len(fused) == 2
    assert fused[0].chunk.chunk_id == "c1"
