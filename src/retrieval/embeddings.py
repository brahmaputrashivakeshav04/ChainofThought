"""Pluggable embedding generator using local all-MiniLM-L6-v2 model."""

from typing import List, Optional
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class EmbeddingProvider:
    """Generates 384-dimensional dense semantic embeddings using local MiniLM model."""

    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = MODEL_ID):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.dimension = 384

    def _lazy_load(self):
        """Loads tokenizer and weights from local Hugging Face cache."""
        if self.model is None and TORCH_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self.model = AutoModel.from_pretrained(self.model_name, local_files_only=True)
                self.model.eval()
            except Exception:
                # If local_files_only with model ID fails, attempt fallback
                self.model = None

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embeds a list of texts into L2-normalized 384-dimensional dense vectors.

        Args:
            texts: List of strings.
            batch_size: Batch size for tokenization and model inference.

        Returns:
            2D numpy array of shape (len(texts), 384).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._lazy_load()

        if self.model is not None and self.tokenizer is not None:
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = [t[:1500] for t in texts[i : i + batch_size]]  # guard max chars
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    out = self.model(**encoded)
                    pooled = self._mean_pooling(out, encoded["attention_mask"])
                    # L2 normalize
                    norm = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    all_embeddings.append(norm.cpu().numpy())
            return np.vstack(all_embeddings).astype(np.float32)

        # Fallback: deterministic pseudo-embedding if model unavailable
        return self._fallback_embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string."""
        vecs = self.embed_texts([query])
        return vecs[0]

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """Lightweight deterministic feature vectors if neural weights cannot be loaded."""
        mat = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split()[:100]:
                idx = hash(word) % self.dimension
                mat[i, idx] += 1.0
            norm = np.linalg.norm(mat[i])
            if norm > 0:
                mat[i] /= norm
        return mat


# Global default provider instance
_default_provider: Optional[EmbeddingProvider] = None


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Convenience helper to embed text chunks."""
    global _default_provider
    if _default_provider is None:
        _default_provider = EmbeddingProvider()
    return _default_provider.embed_texts(texts)
