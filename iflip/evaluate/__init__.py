"""
Evaluation utilities for counterfactual quality.
"""
from .metrics import (
    compute_flip_rate,
    compute_edit_distance,
    compute_perplexity,
    predict_with_sliding_window,
)


from .evaluator import evaluate_counterfactuals

__all__ = [
    "compute_flip_rate",
    "compute_edit_distance",
    "compute_perplexity",
    "predict_with_sliding_window",
    "evaluate_counterfactuals",
]