from __future__ import annotations

from typing import List, Tuple,Dict

import numpy as np
from transformers import pipeline, AutoTokenizer
from difflib import SequenceMatcher
import evaluate

from collections import defaultdict
import Levenshtein
from transformers import GPT2TokenizerFast



from ..config import config


def _top_label(pred_output) -> str:
    """
    Get the label with the highest score from HF pipeline output.
    Works with both return_all_scores=True and False.
    """
    first = pred_output[0]
    if isinstance(first, dict):
        return first["label"]
    best = max(first, key=lambda d: d["score"])
    return best["label"]

# ----------------------------------------------------------------


def _normalise_sentiment_label(label: str) -> int:
    """
    Map model‑specific sentiment labels onto {0 = negative, 1 = positive}.
    """
    label = label.lower()
    if label in {"pos", "positive", "label_1"}:
        return 1
    if label in {"neg", "negative", "label_0"}:
        return 0
    raise ValueError(f"Unknown sentiment label: {label}")


def _normalise_label(label: str) -> int:

    label = label.lower()

    mapping = {
        "positive": 1, "pos": 1, "label_1": 1,
        "negative": 0, "neg": 0, "label_0": 0,

        "world": 0, "label_0": 0,
        "sports": 1, "label_1": 1,
        "business": 2, "label_2": 2,
        "sci/tech": 3, "science": 3, "technology": 3, "label_3": 3,

        "contradiction": 0,
        "neutral": 1,
        "entailment": 2,
    }

    if label in mapping:
        return mapping[label]
    else:
        raise ValueError(f"Unknown label: {label}")


def _majority_vote(votes: List[str]) -> str:

    return max(set(votes), key=votes.count)


def predict_with_confidence(
    text: str,
    clf,                         # HF pipeline already constructed with
    tokenizer,                   # return_all_scores=True
    window_size: int = config.token_window_size,
    stride: int = config.token_stride,
) -> Tuple[str, float]:

    # ---- helper: one-shot prediction with scores ----
    def _predict_once(txt: str) -> Tuple[str, float]:

        outputs = clf(
            txt,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )  
        scores = outputs[0]         
        best   = max(scores, key=lambda d: d["score"])
        return best["label"].lower(), best["score"]


    # ---- short text: no sliding needed ----
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    if len(token_ids) <= window_size:
        return _predict_once(text)

    # ---- sliding windows ----
    vote_buckets: dict[str, List[float]] = defaultdict(list)  # label -> list of scores
    start = 0
    while start < len(token_ids):
        end = min(start + window_size, len(token_ids))
        window_ids = token_ids[start:end]
        window_text = tokenizer.decode(window_ids, skip_special_tokens=True)

        lbl, sc = _predict_once(window_text)
        vote_buckets[lbl].append(sc)

        if end == len(token_ids):
            break
        start += stride

    # majority vote
    majority_label = max(vote_buckets, key=lambda k: len(vote_buckets[k]))

    # average confidence among the windows that cast the majority vote
    confidence = float(np.mean(vote_buckets[majority_label]))

    return majority_label, confidence


def _safe_classify(text: str, clf, tok):
    # Remove special tokens, check length, then truncate if needed
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > tok.model_max_length - 2:
        ids = ids[: tok.model_max_length - 2]
        text = tok.decode(ids, skip_special_tokens=True)

    # No truncation/padding flags here – we control the length ourselves
    return clf(text)          # returns list[list[dict[label, score]]]


def predict_with_sliding_window(
    texts: List[str],
    clf,
    tokenizer,
    window_size: int = config.token_window_size,
    stride     : int = config.token_stride,
) -> List[str]:

    preds = []
    for txt in texts:
        ids = tokenizer.encode(txt, add_special_tokens=True)

        # ── Short sample: classify once ───────────────────────────
        if len(ids) <= window_size:
            preds.append(_top_label(_safe_classify(txt, clf, tokenizer)))
            continue

        # ── Long sample: sliding windows ─────────────────────────
        votes, start = [], 0
        while start < len(ids):
            end = min(start + window_size, len(ids))
            window_txt = tokenizer.decode(ids[start:end], skip_special_tokens=True)
            votes.append(_top_label(_safe_classify(window_txt, clf, tokenizer)))

            if end == len(ids):        # last segment processed
                break
            start += stride

        preds.append(_majority_vote(votes))
    return preds






def compute_flip_rate(
    original_texts: List[str],
    cf_texts      : List[str],
    *,
    classifier_model: str | None = None,
) -> Tuple[float, List[int], List[int]]:
    """Percentage of examples whose predicted label flips."""
    clf_name  = classifier_model or config.classifier_model
    clf       = pipeline("text-classification", model=clf_name, device=0)
    tokenizer = AutoTokenizer.from_pretrained(clf_name, use_fast=False)  # ← slow tokenizer

    orig_preds = predict_with_sliding_window(original_texts, clf, tokenizer)
    cf_preds   = predict_with_sliding_window(cf_texts,      clf, tokenizer)

    original_int = list(map(_normalise_label, orig_preds))
    cf_int = list(map(_normalise_label, cf_preds))

    flips = np.sum(np.array(original_int) != np.array(cf_int))
    rate = flips / len(original_texts)
    return float(rate), original_int, cf_int
    


def compute_edit_distance(original_texts: List[str], cf_texts: List[str]) -> float:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def token_level_levenshtein(ids1: List[int], ids2: List[int]) -> int:
        dp = [[0 for _ in range(len(ids2) + 1)] for _ in range(len(ids1) + 1)]

        for i in range(len(ids1) + 1):
            dp[i][0] = i
        for j in range(len(ids2) + 1):
            dp[0][j] = j

        for i in range(1, len(ids1) + 1):
            for j in range(1, len(ids2) + 1):
                if ids1[i - 1] == ids2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Deletion
                        dp[i][j - 1],      # Insertion
                        dp[i - 1][j - 1],  # Substitution
                    )

        return dp[-1][-1]

    distances = []
    for o, c in zip(original_texts, cf_texts):
        if not o.strip():
            continue
        ids_o = tokenizer.encode(o, add_special_tokens=False)
        ids_c = tokenizer.encode(c, add_special_tokens=False)
        d = token_level_levenshtein(ids_o, ids_c)
        norm_d = d / len(ids_o) if len(ids_o) > 0 else 0
        distances.append(norm_d)

    return sum(distances) / len(distances) if distances else 0.0



def compute_perplexity(
    texts: List[str],
    *,
    lm_model: str = "gpt2",
) -> float:
    tokenizer = AutoTokenizer.from_pretrained(lm_model)
    max_len = tokenizer.model_max_length
    perplexity = evaluate.load("perplexity")
    truncated = [t[:max_len] for t in texts]
    result = perplexity.compute(predictions=truncated, model_id=lm_model)
    return float(np.mean(result["perplexities"]))


def predict_scores_with_sliding_window(
    texts: List[str],
    clf,
    tokenizer,
    window_size: int = config.token_window_size,
    stride: int = config.token_stride,
) -> List[List[Dict[str, float]]]:

    all_outputs = []

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=True)

        # Short enough → direct prediction
        if len(token_ids) <= window_size:
            output = clf(
                text,
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_all_scores=True,
            )[0]
            all_outputs.append(output)
            continue

        vote_buckets = defaultdict(list)  # label → scores

        start = 0
        while start < len(token_ids):
            end = min(start + window_size, len(token_ids))
            window_ids = token_ids[start:end]
            window_text = tokenizer.decode(window_ids, skip_special_tokens=True)

            try:
                '''
                scores = clf(
                    window_text,
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_all_scores=True,
                )[0]
                '''
                
                scores = _safe_classify(window_text, clf, tokenizer)[0]
            except Exception:
                start += stride
                continue

            for item in scores:
                vote_buckets[item["label"]].append(item["score"])

            if end == len(token_ids):
                break
            start += stride

        averaged_scores = [
            {"label": label, "score": float(np.mean(score_list))}
            for label, score_list in vote_buckets.items()
        ]
        all_outputs.append(averaged_scores)

    return all_outputs
