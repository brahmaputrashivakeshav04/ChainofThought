"""Evaluation module for retrieval performance and generation faithfulness."""

from .metrics import compute_recall_at_k, compute_mrr, compute_hit_rate
from .llm_judge import evaluate_faithfulness

__all__ = [
    "compute_recall_at_k",
    "compute_mrr",
    "compute_hit_rate",
    "evaluate_faithfulness",
]
