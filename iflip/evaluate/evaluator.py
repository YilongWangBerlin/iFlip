"""
Convenience wrapper that bundles all evaluation metrics.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

from .metrics import compute_flip_rate, compute_edit_distance, compute_perplexity


def evaluate_counterfactuals(
    original_texts: List[str], cf_texts: List[str]
) -> Tuple[Dict[str, float], Dict[str, List[int]]]:
    """
    Compute flip‑rate, average edit distance and average perplexity.

    Returns a tuple *(metric_dict, raw_predictions)* where raw_predictions
    contains the classifier outputs for further analysis.
    """
    flip_rate, orig_preds, cf_preds = compute_flip_rate(original_texts, cf_texts)
    edit_distance = compute_edit_distance(original_texts, cf_texts)
    ppl = compute_perplexity(cf_texts)

    metrics = {
        "flip_rate": flip_rate,
        "avg_edit_distance": edit_distance,
        "avg_perplexity": ppl,
    }
    raw_preds = {"original": orig_preds, "counterfactual": cf_preds}
    return metrics, raw_preds