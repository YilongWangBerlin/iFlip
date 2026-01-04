"""
Convenience wrapper that bundles all evaluation metrics.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

from .metrics import compute_flip_rate, compute_perplexity, compute_semantic_similarity


def evaluate_counterfactuals(
    original_texts: List[str], cf_texts: List[str]
) -> Tuple[Dict[str, float], Dict[str, List[int]]]:

    flip_rate, orig_preds, cf_preds = compute_flip_rate(original_texts, cf_texts)
    ppl = compute_perplexity(cf_texts)
    sim_mean = compute_semantic_similarity(original_texts, cf_texts)

    metrics = {
        "flip_rate": flip_rate,
        "avg_perplexity": ppl,
        "semantic_similarity": sim_mean, 
    }
    raw_preds = {"original": orig_preds, "counterfactual": cf_preds}
    return metrics, raw_preds