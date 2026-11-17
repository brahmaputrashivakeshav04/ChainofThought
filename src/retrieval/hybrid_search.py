"""Hybrid retriever combining dense semantic search and BM25 keyword matching with cross-encoder reranking."""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from ..schemas import SearchResult, TextChunk
from ..config import TOP_K_RETRIEVAL, TOP_K_RERANK, HYBRID_ALPHA, INDEX_DIR
from .vector_store import VectorIndex
from .bm25_index import BM25Index
from .embeddings import EmbeddingProvider

logger = logging.getLogger("researchiq.retrieval")


class HybridRetriever:
    """Orchestrates hybrid dense + sparse retrieval, paper filtering, and cross-encoder reranking."""

    RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        vector_index: VectorIndex,
        bm25_index: BM25Index,
        embedding_provider: EmbeddingProvider,
        alpha: float = HYBRID_ALPHA,
    ):
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.embedding_provider = embedding_provider
        self.alpha = alpha
        self._reranker = None
        self._reranker_tokenizer = None
        self._reranker_failed = False

    def _lazy_load_reranker(self):
        """Attempts to load local cross-encoder reranker; falls back cleanly if unavailable."""
        if self._reranker is None and not self._reranker_failed:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self._reranker_tokenizer = AutoTokenizer.from_pretrained(self.RERANKER_ID, local_files_only=True)
                self._reranker = AutoModelForSequenceClassification.from_pretrained(self.RERANKER_ID, local_files_only=True)
                self._reranker.eval()
            except Exception as e:
                self._reranker_failed = True
                logger.warning(f"Cross-encoder reranker '{self.RERANKER_ID}' unavailable locally; falling back to RRF ranking.")

    def reciprocal_rank_fusion(
        self,
        dense_results: List[SearchResult],
        bm25_results: List[SearchResult],
        k: int = 60,
    ) -> List[SearchResult]:
        """Merges two ranked lists using Reciprocal Rank Fusion (RRF)."""
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, TextChunk] = {}

        for res in dense_results:
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (k + res.rank))

        for res in bm25_results:
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (k + res.rank))

        sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            SearchResult(
                chunk=chunk_map[cid],
                score=score,
                rank=i + 1,
                retrieval_method="rrf_hybrid",
            )
            for i, (cid, score) in enumerate(sorted_items)
        ]

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = TOP_K_RERANK,
    ) -> List[SearchResult]:
        """Reranks candidates using cross-encoder or falls back to RRF ordering."""
        if not candidates:
            return []

        self._lazy_load_reranker()

        if self._reranker is not None and self._reranker_tokenizer is not None:
            import torch
            pairs = [[query, c.chunk.text[:800]] for c in candidates]
            with torch.no_grad():
                inputs = self._reranker_tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
                scores = self._reranker(**inputs).logits.squeeze(-1).cpu().numpy()

            scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
            return [
                SearchResult(
                    chunk=cand.chunk,
                    score=float(score),
                    rank=idx + 1,
                    retrieval_method="cross_encoder",
                )
                for idx, (cand, score) in enumerate(scored)
            ]

        # Graceful fallback: return top candidates from RRF
        return candidates[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        top_k_rerank: int = TOP_K_RERANK,
        paper_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """End-to-end retrieval with paper_id filter and RRF + reranking."""
        query_vec = self.embedding_provider.embed_query(query)

        # Pull broader candidate pool (up to 40) before filtering
        dense_results = self.vector_index.search(query_vec, top_k=top_k * 2)
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)

        # Filter by paper_ids if requested
        if paper_ids:
            allowed_pids = set(paper_ids)
            dense_results = [r for r in dense_results if r.chunk.paper_id in allowed_pids]
            bm25_results = [r for r in bm25_results if r.chunk.paper_id in allowed_pids]

        fused = self.reciprocal_rank_fusion(dense_results, bm25_results)[:top_k]

        # Resolution of visual_description chunks to parent figure and deduplication
        resolved_fused: List[SearchResult] = []
        seen_chunk_ids = set()

        # Build chunk lookup for parent resolution
        chunk_map = {c.chunk_id: c for c in self.vector_index.chunks}

        for item in fused:
            c = item.chunk
            if c.chunk_type == "visual_description" and c.parent_chunk_id:
                parent = chunk_map.get(c.parent_chunk_id)
                if parent:
                    # Inherit visual description if not already set
                    if getattr(parent, "visual_description", None) is None:
                        try:
                            parent.visual_description = getattr(c, "visual_description", None)
                        except Exception:
                            pass
                    if parent.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(parent.chunk_id)
                        resolved_fused.append(
                            SearchResult(
                                chunk=parent,
                                score=item.score,
                                rank=len(resolved_fused) + 1,
                                retrieval_method="hybrid_resolved_visual",
                            )
                        )
                    continue

            if c.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(c.chunk_id)
                resolved_fused.append(item)

        # Figure/Table Number Awareness: e.g., "Figure 1", "Fig. 2", "Table 1", "Table I"
        num_match = re.search(r"\b(figure|fig\.?|table)\s*([0-9ivx]+[a-z]?)\b", query, re.IGNORECASE)
        if num_match:
            elem_kind = "figure" if "fig" in num_match.group(1).lower() else "table"
            target_num = num_match.group(2).lower()

            exact_matches = []
            other_candidates = []
            for item in resolved_fused:
                chunk_label = (item.chunk.element_number or "").lower()
                chunk_caption = (item.chunk.caption or "").lower()
                is_match = (
                    item.chunk.chunk_type == elem_kind
                    and (f"{elem_kind} {target_num}" in chunk_label
                         or f"fig {target_num}" in chunk_label
                         or f"tab {target_num}" in chunk_label
                         or f"{elem_kind} {target_num}" in chunk_caption
                         or f"fig. {target_num}" in chunk_caption
                         or f"table {target_num}" in chunk_caption)
                )
                if is_match:
                    exact_matches.append(item)
                else:
                    other_candidates.append(item)

            if exact_matches:
                # If paper_ids filter is specified, place the exact matching element first
                if paper_ids:
                    resolved_fused = exact_matches + other_candidates
                else:
                    # Boost matching element numbers
                    for m in exact_matches:
                        m.score += 0.05
                    resolved_fused = sorted(exact_matches + other_candidates, key=lambda x: x.score, reverse=True)

        final_ranked = self.rerank(query, resolved_fused, top_k=top_k_rerank)
        return final_ranked

    @classmethod
    def from_persisted(cls, index_dir: Path = INDEX_DIR) -> "HybridRetriever":
        """Loads vector and BM25 indices from disk and initializes HybridRetriever."""
        vec_index = VectorIndex()
        vec_index.load(index_dir)
        bm25_idx = BM25Index()
        bm25_idx.load(index_dir)
        provider = EmbeddingProvider()
        return cls(vector_index=vec_index, bm25_index=bm25_idx, embedding_provider=provider)
