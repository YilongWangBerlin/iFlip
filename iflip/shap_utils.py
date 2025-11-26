from typing import List, Tuple
import torch
import shap
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from .config import config 


DEVICE = 0 if torch.cuda.is_available() else -1
#MODEL_NAME = "textattack/bert-base-uncased-imdb"
MODEL_NAME = config.classifier_model()

MAX_TOKEN_LENGTH = 512
WINDOW_SIZE = 512
STRIDE = 256

clf_pipe = pipeline("text-classification",
                    model=MODEL_NAME,
                    tokenizer=MODEL_NAME,
                    device=DEVICE,
                    return_all_scores=True,
                    truncation=True)

tokenizer: AutoTokenizer = clf_pipe.tokenizer
explainer = shap.Explainer(clf_pipe, masker=shap.maskers.Text(tokenizer))


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


def get_top_relevant_words_shap(text: str, k: int = 5) -> List[str]:
    
    input_ids = tokenizer.encode(text, truncation=False, add_special_tokens=True)
    if len(input_ids) <= MAX_TOKEN_LENGTH:
        shap_values = explainer([text], fixed_context=0)
        tokens = shap_values.data[0]
        values = shap_values.values[0]
        pred_label = max(clf_pipe(text)[0], key=lambda d: d["score"])["label"].lower()
        class_idx = 1 if pred_label.startswith("pos") else 0
        contrib = [abs(v[class_idx]) for v in values]
        merged = merge_wordpieces(tokens, contrib)
        merged.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in merged[:k]]


    words = text.split()
    chunks = []
    for i in range(0, len(words), STRIDE):
        chunk = " ".join(words[i:i + WINDOW_SIZE])
        if chunk.strip():
            chunks.append(chunk)

    word_scores = {}
    for chunk in chunks:
        shap_values = explainer([chunk], fixed_context=0)
        tokens = shap_values.data[0]
        values = shap_values.values[0]
        pred_label = max(clf_pipe(chunk)[0], key=lambda d: d["score"])["label"].lower()
        class_idx = 1 if pred_label.startswith("pos") else 0
        contrib = [abs(v[class_idx]) for v in values]
        merged = merge_wordpieces(tokens, contrib)
        for word, score in merged:
            word_scores.setdefault(word, []).append(score)

    avg_scores = [(w, sum(s) / len(s)) for w, s in word_scores.items()]
    avg_scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in avg_scores[:k]]



def get_top_unrelevant_words_shap(text: str, k: int = 5) -> List[str]:

    input_ids = tokenizer.encode(text, truncation=False, add_special_tokens=True)
    if len(input_ids) <= MAX_TOKEN_LENGTH:

        shap_values = explainer([text], fixed_context=0)
        tokens = shap_values.data[0]
        values = shap_values.values[0]
        pred_label = max(clf_pipe(text)[0], key=lambda d: d["score"])["label"].lower()
        class_idx = 1 if pred_label.startswith("pos") else 0
        contrib = [abs(v[class_idx]) for v in values]
        merged = merge_wordpieces(tokens, contrib)
        merged.sort(key=lambda x: x[1])
        return [word for word, _ in merged[:k]]


    words = text.split()
    chunks = []
    for i in range(0, len(words), STRIDE):
        chunk = " ".join(words[i:i + WINDOW_SIZE])
        if chunk.strip():
            chunks.append(chunk)

    word_scores = {}
    for chunk in chunks:
        shap_values = explainer([chunk], fixed_context=0)
        tokens = shap_values.data[0]
        values = shap_values.values[0]
        pred_label = max(clf_pipe(chunk)[0], key=lambda d: d["score"])["label"].lower()
        class_idx = 1 if pred_label.startswith("pos") else 0
        contrib = [abs(v[class_idx]) for v in values]
        merged = merge_wordpieces(tokens, contrib)
        for word, score in merged:
            word_scores.setdefault(word, []).append(score)

    avg_scores = [(w, sum(s) / len(s)) for w, s in word_scores.items()]
    avg_scores.sort(key=lambda x: x[1])
    return [word for word, _ in avg_scores[:k]]