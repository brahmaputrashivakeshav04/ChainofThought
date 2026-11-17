"""Information retrieval evaluation metrics: Recall@K, MRR, and HitRate."""

from typing import List, Dict, Set


def compute_recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Computes Recall@K: fraction of relevant chunks retrieved in top K.

    Recall@K = |Retrieved_top_K ∩ Relevant| / |Relevant|
    """
    if not ground_truth_ids:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    hits = len(top_k_set.intersection(ground_truth_ids))
    return hits / len(ground_truth_ids)


def compute_mrr(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """Computes Mean Reciprocal Rank (MRR): 1 / rank of first relevant result."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def compute_hit_rate(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Computes HitRate@K: 1.0 if at least one relevant document is in top K, else 0.0."""
    top_k_set = set(retrieved_ids[:k])
    return 1.0 if len(top_k_set.intersection(ground_truth_ids)) > 0 else 0.0


def evaluate_retrieval_batch(
    eval_dataset: List[Dict],
    retriever_fn,
    k_values: List[int] = [3, 5, 10],
) -> Dict[str, float]:
    """Runs batch evaluation over an annotated benchmark dataset.

    Args:
        eval_dataset: List of dicts with 'query' and 'relevant_chunk_ids'.
        retriever_fn: Function taking query string and returning list of retrieved chunk_ids.
        k_values: List of K thresholds.

    Returns:
        Aggregated metric scores.
    """
    if not eval_dataset:
        return {}

    metrics = {f"Recall@{k}": 0.0 for k in k_values}
    metrics["MRR"] = 0.0

    for item in eval_dataset:
        query = item["query"]
        relevant_ids = set(item["relevant_chunk_ids"])
        retrieved_ids = retriever_fn(query)

        for k in k_values:
            metrics[f"Recall@{k}"] += compute_recall_at_k(retrieved_ids, relevant_ids, k)
        metrics["MRR"] += compute_mrr(retrieved_ids, relevant_ids)

    n = len(eval_dataset)
    for key in metrics:
        metrics[key] /= n

    return metrics
