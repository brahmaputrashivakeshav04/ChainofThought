"""BM25 keyword index for exact token and vocabulary matching with persistence."""

import math
import pickle
import re
from pathlib import Path
from typing import List, Optional

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    # Pure-Python BM25 fallback when rank_bm25 is not installed
    class BM25Okapi:  # type: ignore
        def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
            self.corpus_size = len(corpus)
            self.avgdl = sum(map(len, corpus)) / self.corpus_size if self.corpus_size > 0 else 0
            self.corpus = corpus
            self.k1 = k1
            self.b = b
            self.doc_freqs = []
            self.idf = {}
            self.doc_len = [len(doc) for doc in corpus]
            nd = {}
            for document in corpus:
                frequencies = {}
                for word in document:
                    frequencies[word] = frequencies.get(word, 0) + 1
                self.doc_freqs.append(frequencies)
                for word in frequencies:
                    nd[word] = nd.get(word, 0) + 1
            for word, freq in nd.items():
                self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

        def get_scores(self, query: List[str]) -> List[float]:
            scores = [0.0] * self.corpus_size
            for q in query:
                q_idf = self.idf.get(q, 0.0)
                for i, doc in enumerate(self.doc_freqs):
                    freq = doc.get(q, 0)
                    denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                    scores[i] += (q_idf * freq * (self.k1 + 1)) / (denom or 1)
            return scores

from ..schemas import TextChunk, SearchResult
from ..config import INDEX_DIR


class BM25Index:
    """Lexical inverted index with incremental build and persistence."""

    def __init__(self):
        self.chunks: List[TextChunk] = []
        self.chunk_ids: set[str] = set()
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple lowercase alphanumeric tokenizer."""
        return re.findall(r"\b\w+\b", text.lower())

    def build_index(self, chunks: List[TextChunk]):
        """Builds fresh BM25 index over provided text chunks."""
        self.chunks = chunks
        self.chunk_ids = {c.chunk_id for c in chunks}
        self.tokenized_corpus = [self.tokenize(c.text) for c in chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def add_chunks(self, chunks: List[TextChunk]):
        """Incrementally adds new chunks and updates index."""
        new_chunks = [c for c in chunks if c.chunk_id not in self.chunk_ids]
        if not new_chunks:
            return
        combined = self.chunks + new_chunks
        self.build_index(combined)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Searches BM25 index for query keywords."""
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0.0:
                results.append(
                    SearchResult(
                        chunk=self.chunks[idx],
                        score=float(scores[idx]),
                        rank=rank + 1,
                        retrieval_method="bm25",
                    )
                )
        return results

    def save(self, index_dir: Path = INDEX_DIR):
        """Serializes BM25 index to disk."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "tokenized_corpus": self.tokenized_corpus}, f)

    def load(self, index_dir: Path = INDEX_DIR):
        """Deserializes BM25 index from disk."""
        index_dir = Path(index_dir)
        pkl_path = index_dir / "bm25_index.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self.chunks = data.get("chunks", [])
            self.chunk_ids = {c.chunk_id for c in self.chunks}
            self.tokenized_corpus = data.get("tokenized_corpus", [])
            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)
