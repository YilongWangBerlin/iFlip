from typing import List, Tuple
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer
from lime.lime_text import LimeTextExplainer
from .config import config
import re



DEVICE = 0 if torch.cuda.is_available() else -1
MODEL_NAME = config.classifier_model()

MAX_TOKEN_LENGTH = 512
WINDOW_SIZE = 512
STRIDE = 256


LABEL_MAPS = {
    "imdb": ["negative", "positive"],
    "snli": ["entailment", "neutral", "contradiction"],
    "agnews": ["world", "sports", "business", "sci/tech"]
}


clf_pipe = pipeline(
    "text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=DEVICE,
    return_all_scores=True,
    truncation=True
)

tokenizer: AutoTokenizer = clf_pipe.tokenizer
task_name = config.task_name
if task_name not in LABEL_MAPS:
    raise ValueError(f"Unsupported dataset: {task_name}")

class_names = LABEL_MAPS[task_name]
explainer = LimeTextExplainer(class_names=class_names)


def merge_wordpieces(tokens: List[str], scores: List[float]) -> List[Tuple[str, float]]:
    words, relevances = [], []
    current_word, current_score, count = "", 0.0, 0
    for tok, score in zip(tokens, scores):
        if tok.startswith("##"):
            current_word += tok[2:]
        else:
            if current_word:
                words.append(current_word)
                relevances.append(current_score / max(1, count))
            current_word, current_score, count = tok, 0.0, 0
        current_score += score
        count += 1
    if current_word:
        words.append(current_word)
        relevances.append(current_score / max(1, count))
    return list(zip(words, relevances))





def _normalize_label(name: str) -> str:

    return name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")

def _lime_predict(texts: List[str]):

    outputs = []

    norm_class_names = [_normalize_label(n) for n in class_names]

    model_cfg = getattr(getattr(clf_pipe, "model", None), "config", None)
    id2label = getattr(model_cfg, "id2label", None)

    def _id2name(idx):
        if id2label is None:
            return None
        return id2label.get(idx) or id2label.get(str(idx))

    for text in texts:
        scores = clf_pipe(text)[0]  # [{'label': str, 'score': float}, ...]

        norm_label_score = {_normalize_label(d["label"]): d["score"] for d in scores}


        looks_like_indexed = all(k.startswith("label") for k in norm_label_score.keys())


        if looks_like_indexed and id2label:
            norm_label_score = {}
            for d in scores:
                raw = d["label"]
                m = re.search(r'label[_-]?(\d+)', raw, flags=re.IGNORECASE)
                if not m:
                    continue
                idx = int(m.group(1))
                mapped = _id2name(idx)
                if mapped is None:
                    continue
                norm_label = _normalize_label(mapped)
                norm_label_score[norm_label] = d["score"]


        row = [norm_label_score.get(n, 0.0) for n in norm_class_names]


        if not any(row):
            sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)

            for i in range(len(row)):
                row[i] = sorted_scores[i % len(sorted_scores)]["score"]
            s = sum(row)
            if s > 0:
                row = [x / s for x in row]
        else:

            s = sum(row)
            if s > 0:
                row = [x / s for x in row]

        outputs.append(row)

    return np.array(outputs, dtype=float)

def _get_lime_explanation(text: str):
    if len(text.strip().split()) <= 1:
        return None
        
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=_lime_predict,
        num_features=20,        
        num_samples=500         
    )
    return exp


def get_top_relevant_words_lime(text: str, k: int = 5) -> List[str]:
    input_ids = tokenizer.encode(text, truncation=False, add_special_tokens=True)
    if len(input_ids) <= MAX_TOKEN_LENGTH:
        exp = _get_lime_explanation(text)
        if exp is None:
            return []
        exp_list = exp.as_list()
        exp_list.sort(key=lambda x: abs(x[1]), reverse=True)
        return [word for word, _ in exp_list[:k]]

    # Sliding window
    words = text.split()
    chunks = []
    for i in range(0, len(words), STRIDE):
        chunk = " ".join(words[i:i + WINDOW_SIZE])
        if chunk.strip():
            chunks.append(chunk)

    word_scores = {}
    for chunk in chunks:
        exp = _get_lime_explanation(chunk)
        
        if exp is None:
            continue
        
        for word, score in exp.as_list():
            word_scores.setdefault(word, []).append(abs(score))

    avg_scores = [(w, sum(s) / len(s)) for w, s in word_scores.items()]
    avg_scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in avg_scores[:k]]

def get_top_unrelevant_words_lime(text: str, k: int = 5) -> List[str]:
    input_ids = tokenizer.encode(text, truncation=False, add_special_tokens=True)
    if len(input_ids) <= MAX_TOKEN_LENGTH:
        exp = _get_lime_explanation(text)
        exp_list = exp.as_list()
        exp_list.sort(key=lambda x: abs(x[1]))
        return [word for word, _ in exp_list[:k]]

    # Sliding window
    words = text.split()
    chunks = []
    for i in range(0, len(words), STRIDE):
        chunk = " ".join(words[i:i + WINDOW_SIZE])
        if chunk.strip():
            chunks.append(chunk)

    word_scores = {}
    for chunk in chunks:
        exp = _get_lime_explanation(chunk)
        for word, score in exp.as_list():
            word_scores.setdefault(word, []).append(abs(score))

    avg_scores = [(w, sum(s) / len(s)) for w, s in word_scores.items()]
    avg_scores.sort(key=lambda x: x[1])
    return [word for word, _ in avg_scores[:k]]

