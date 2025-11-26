from __future__ import annotations
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers.models.bert.modeling_bert as modeling_bert

from lxt.efficient import monkey_patch
from lxt.utils import clean_tokens
from .config import config 

# ---------- set-up ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#MODEL_NAME = "textattack/bert-base-uncased-imdb"
MODEL_NAME = config.classifier_model()


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
monkey_patch(modeling_bert, verbose=False)

model = modeling_bert.BertForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()         



from collections import defaultdict

def get_top_relevant_words_lxt(text: str, k: int = 5) -> List[str]:

    max_len = 512
    stride = 256
    tokens = tokenizer.tokenize(text)

    if len(tokens) <= max_len:

        return _extract_top_words_lxt(text, k)
    else:
        word_scores = defaultdict(int)
        for i in range(0, len(tokens), stride):
            window_tokens = tokens[i:i + max_len]
            window_text = tokenizer.convert_tokens_to_string(window_tokens)
            top_words = _extract_top_words_lxt(window_text, k)
            for word in top_words:
                word_scores[word] += 1

        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:k]]

def _extract_top_words_lxt(text: str, k: int = 5) -> List[str]:
    with torch.no_grad():
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=True,
                              truncation=True, max_length=512).input_ids.to(DEVICE)

    embeds = model.bert.get_input_embeddings()(input_ids).requires_grad_(True)
    logits = model(inputs_embeds=embeds).logits
    pred_idx = logits.argmax(dim=-1)

    grad = torch.autograd.grad(
        outputs=logits[0, pred_idx],
        inputs=embeds,
        retain_graph=False,
        create_graph=False,
    )[0]

    relevance = (embeds * grad).sum(dim=-1).squeeze(0)
    relevance = relevance / relevance.abs().max()

    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    words, relevances = [], []
    current_word, current_score, current_len = "", 0.0, 0
    for tok, rel in zip(raw_tokens, relevance.tolist()):
        if tok in ("[CLS]", "[SEP]"):
            continue
        if tok.startswith("##"):
            current_word += tok[2:]
        else:
            if current_word:
                words.append(current_word)
                relevances.append(current_score / max(current_len, 1))
            current_word = tok
            current_score = 0.0
            current_len = 0
        current_score += abs(rel)
        current_len += 1
    if current_word:
        words.append(current_word)
        relevances.append(current_score / max(current_len, 1))


    pairs = sorted(zip(words, relevances), key=lambda x: x[1], reverse=True)
    return [word for word, _ in pairs[:k]]



def get_top_unrelevant_words_lxt(text: str, k: int = 5) -> List[str]:

    max_len = 512
    stride = 256
    tokens = tokenizer.tokenize(text)

    if len(tokens) <= max_len:
        return _extract_least_words_lxt(text, k)
    else:
        word_scores = defaultdict(int)
        for i in range(0, len(tokens), stride):
            window_tokens = tokens[i:i + max_len]
            window_text = tokenizer.convert_tokens_to_string(window_tokens)
            least_words = _extract_least_words_lxt(window_text, k)
            for word in least_words:
                word_scores[word] += 1

        sorted_words = sorted(word_scores.items(), key=lambda x: x[1])
        return [word for word, _ in sorted_words[:k]]



def _extract_least_words_lxt(text: str, k: int = 5) -> List[str]:
    with torch.no_grad():
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=True,
                              truncation=True, max_length=512).input_ids.to(DEVICE)

    embeds = model.bert.get_input_embeddings()(input_ids).requires_grad_(True)
    logits = model(inputs_embeds=embeds).logits
    pred_idx = logits.argmax(dim=-1)

    grad = torch.autograd.grad(
        outputs=logits[0, pred_idx],
        inputs=embeds,
        retain_graph=False,
        create_graph=False,
    )[0]

    relevance = (embeds * grad).sum(dim=-1).squeeze(0)
    relevance = relevance / relevance.abs().max()

    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    words, relevances = [], []
    current_word, current_score, current_len = "", 0.0, 0
    for tok, rel in zip(raw_tokens, relevance.tolist()):
        if tok in ("[CLS]", "[SEP]"):
            continue
        if tok.startswith("##"):
            current_word += tok[2:]
        else:
            if current_word:
                words.append(current_word)
                relevances.append(current_score / max(current_len, 1))
            current_word = tok
            current_score = 0.0
            current_len = 0
        current_score += abs(rel)
        current_len += 1
    if current_word:
        words.append(current_word)
        relevances.append(current_score / max(current_len, 1))

    pairs = sorted(zip(words, relevances), key=lambda x: x[1])
    return [word for word, _ in pairs[:k]]
