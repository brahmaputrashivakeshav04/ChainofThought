"""Vector store index wrapping FAISS or vectorized numpy matrix operations."""

import pickle
from pathlib import Path
from typing import List, Optional
import numpy as np

from ..schemas import TextChunk, SearchResult
from ..config import INDEX_DIR


class VectorIndex:
    """Manages dense vector storage and nearest neighbor retrieval.

    Uses normalized inner product (cosine similarity).
    Persists vectors to vector_index.npz and chunk metadata to chunks.pkl.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.chunks: List[TextChunk] = []
        self.chunk_ids: set[str] = set()
        self.vectors: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray):
        """Indexes text chunks and their dense embeddings incrementally.

        Args:
            chunks: List of TextChunk objects.
            embeddings: Numpy array of shape (len(chunks), dimension).
        """
        if not chunks or len(embeddings) == 0:
            return

        new_chunks = []
        new_indices = []

        for idx, chunk in enumerate(chunks):
            if chunk.chunk_id not in self.chunk_ids:
                new_chunks.append(chunk)
                new_indices.append(idx)
                self.chunk_ids.add(chunk.chunk_id)

        if not new_chunks:
            return

        filtered_embeddings = embeddings[new_indices]
        self.chunks.extend(new_chunks)

        if self.vectors is None:
            self.vectors = filtered_embeddings
        else:
            self.vectors = np.vstack([self.vectors, filtered_embeddings])

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """Performs cosine nearest-neighbor search.

        Args:
            query_vector: 1D dense vector.
            top_k: Number of nearest matches to return.

        Returns:
            Ranked list of SearchResult objects.
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []

        # Ensure query is unit normalized
        q_norm = np.linalg.norm(query_vector)
        if q_norm > 0:
            q = query_vector / q_norm
        else:
            q = query_vector

        # Dot product with normalized document vectors yields cosine similarity
        scores = np.dot(self.vectors, q)
        top_k_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_k_indices):
            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    rank=rank + 1,
                    retrieval_method="dense",
                )
            )
        return results

    def save(self, index_dir: Path = INDEX_DIR):
        """Serializes vector store and chunk metadata to disk."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        # Save numpy vectors
        if self.vectors is not None:
            np.savez_compressed(index_dir / "vector_index.npz", vectors=self.vectors)

        # Save chunk objects
        with open(index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, index_dir: Path = INDEX_DIR):
        """Loads vector store and chunk metadata from disk."""
        index_dir = Path(index_dir)
        vec_path = index_dir / "vector_index.npz"
        chunks_path = index_dir / "chunks.pkl"

        if vec_path.exists() and chunks_path.exists():
            npz = np.load(vec_path)
            self.vectors = npz["vectors"]
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            self.chunk_ids = {c.chunk_id for c in self.chunks}
            if self.vectors is not None and len(self.vectors) > 0:
                self.dimension = self.vectors.shape[1]
