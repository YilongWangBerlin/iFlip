from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from .config import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = config.classifier_model()

MAX_TOKEN_LENGTH = 512
WINDOW_SIZE = 512
STRIDE = 256



tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

clf_pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    return_all_scores=True,
    truncation=True
)



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



def _compute_gradxinput(text: str) -> Tuple[List[str], List[float]]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKEN_LENGTH)
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    input_ids = encoded["input_ids"]

    # Get embeddings as leaf + requires_grad
    embeds = model.get_input_embeddings()(input_ids).detach()
    embeds.requires_grad_(True)

    outputs = model(inputs_embeds=embeds, attention_mask=encoded["attention_mask"])
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_class = probs.argmax(dim=-1).item()

    # Compute gradient
    grad = torch.autograd.grad(
        outputs=probs[0, pred_class],
        inputs=embeds,
        retain_graph=False
    )[0]  # shape: (1, L, H)

    # grad * input, sum over hidden dim
    gradxinput = (grad * embeds).sum(dim=-1).squeeze(0)  # shape: (L,)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    scores = gradxinput.detach().cpu().tolist()
    return tokens, scores


def get_top_relevant_words_gradxinput(text: str, k: int = 5) -> List[str]:
    input_ids = tokenizer.encode(text, truncation=False, add_special_tokens=True)
    if len(input_ids) <= MAX_TOKEN_LENGTH:
        tokens, scores = _compute_gradxinput(text)
        merged = merge_wordpieces(tokens, [abs(s) for s in scores])
        merged.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in merged[:k]]

    # Sliding window
    words = text.split()
    chunks = []
    for i in range(0, len(words), STRIDE):
        chunk = " ".join(words[i:i + WINDOW_SIZE])
        if chunk.strip():
            chunks.append(chunk)

    word_scores = {}
    for chunk in chunks:
        tokens, scores = _compute_gradxinput(chunk)
        merged = merge_wordpieces(tokens, [abs(s) for s in scores])
        for word, score in merged:
            word_scores.setdefault(word, []).append(score)

    avg_scores = [(w, sum(s) / len(s)) for w, s in word_scores.items()]
    avg_scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in avg_scores[:k]]

def get_top_unrelevant_words_gradxinput(text: str, k: int = 5) -> List[str]:
    input_ids = tokenizer.encode(text, truncation=False, add_special_tokens=True)
    if len(input_ids) <= MAX_TOKEN_LENGTH:
        tokens, scores = _compute_gradxinput(text)
        merged = merge_wordpieces(tokens, [abs(s) for s in scores])
        merged.sort(key=lambda x: x[1])
        return [word for word, _ in merged[:k]]

    # Sliding window
    words = text.split()
    chunks = []
    for i in range(0, len(words), STRIDE):
        chunk = " ".join(words[i:i + WINDOW_SIZE])
        if chunk.strip():
            chunks.append(chunk)

    word_scores = {}
    for chunk in chunks:
        tokens, scores = _compute_gradxinput(chunk)
        merged = merge_wordpieces(tokens, [abs(s) for s in scores])
        for word, score in merged:
            word_scores.setdefault(word, []).append(score)

    avg_scores = [(w, sum(s) / len(s)) for w, s in word_scores.items()]
    avg_scores.sort(key=lambda x: x[1])
    return [word for word, _ in avg_scores[:k]]

