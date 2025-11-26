import os
import json
import argparse
import pandas as pd
import torch
import gc
import string
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ferret import (
    Benchmark,
    GradientExplainer,
    IntegratedGradientExplainer,
    LIMEExplainer,
    SHAPExplainer,
    LXTExplainer,
)

import logging
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Config
# -------------------------------
LABEL_MAP = {
    "imdb": {
        "pos": 1,
        "positive": 1,
        "neg": 0,
        "negative": 0,
    },
    "snli": {
        "contradiction": 0,
        "entailment": 1,
        "neutral": 2,
    },
    "agnews": {
        "world": 0,
        "sports": 1,
        "business": 2,
        "sci/tech": 3,
    },
}

MODEL_MAP = {
    "imdb": "textattack/bert-base-uncased-imdb",
    "snli": "textattack/bert-base-uncased-snli",
    "agnews": "textattack/bert-base-uncased-ag-news",
}


# -------------------------------
# Detect dataset from filename
# -------------------------------
def detect_dataset_from_filename(filename):
    fname = filename.lower()
    if "imdb" in fname:
        return "imdb"
    elif "snli" in fname:
        return "snli"
    elif "agnews" in fname:
        return "agnews"
    else:
        return None


# -------------------------------
# Top-k word selection
# -------------------------------
def select_topk_words(topk, explanation):
    rankings = list(np.argsort(explanation.scores))
    topk_words = []

    for j in rankings[-topk:]:
        topk_word = explanation.tokens[j]


        if topk_word.startswith("##"):
            topk_word = explanation.tokens[j - 1] + topk_word.replace("##", "")


        if topk_word not in string.punctuation and topk_word not in ["[SEP]", "[CLS]"]:
            topk_words.append(topk_word)


    if len(topk_words) < topk:
        for j in rankings[:-topk][::-1]:
            if len(topk_words) < topk:
                topk_word = explanation.tokens[j]
                if topk_word.startswith("##"):
                    topk_word = explanation.tokens[j - 1] + topk_word.replace("##", "")
                if topk_word not in string.punctuation and topk_word not in ["[SEP]", "[CLS]"]:
                    topk_words.append(topk_word)
            else:
                break
    return topk_words


# -------------------------------
# Choose explainers based on filename
# -------------------------------
def get_explainers_from_filename(model, tokenizer, filename):
    fname = filename.lower()

    if "shap" in fname:
        return [SHAPExplainer(model, tokenizer)]
    elif "lime" in fname:
        return [LIMEExplainer(model, tokenizer)]
    elif "gradxinput" in fname:
        return [GradientExplainer(model, tokenizer, multiply_by_inputs=True)]
    elif "lxt" in fname:
        return [LXTExplainer(model, tokenizer)]
    elif "ig" in fname or "intgrad" in fname:
        return [IntegratedGradientExplainer(model, tokenizer, multiply_by_inputs=True)]
    elif "conf" in fname:
        print(f"Skipping file {filename} (unsupported explainer: conf)")
        return []
    else:
        print(f"Warning: no explainer found for {filename}, skipping")
        return []


# -------------------------------
# Core
# -------------------------------
def explanation_generation_and_evaluation_by_ferret(results_dir, output_prefix, limit=None, topk=5, file_name=None):
    if file_name:
        files = [file_name]
    else:
        files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]


    dataset_results = {"imdb": [], "snli": [], "agnews": []}

    for file in files:
        dataset_name = detect_dataset_from_filename(file)
        if dataset_name is None:
            print(f"⚠️ Skipping {file}: dataset could not be detected")
            continue

        # load model
        model_name = MODEL_MAP[dataset_name]
        tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, max_length=512)
        if tokenizer.sep_token is None:
            tokenizer.sep_token = "[SEP]"

        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

        filepath = os.path.join(results_dir, file)
        df = pd.read_csv(filepath)

        if "counterfactual" not in df.columns:
            print(f"File {file} has no 'counterfactual' column, skipping")
            continue

        explainers = get_explainers_from_filename(model, tokenizer, file)
        if not explainers:
            continue

        bench = Benchmark(model, tokenizer, explainers=explainers)
        print(f"Evaluating {file} with detected task={dataset_name} and explainers={[e.__class__.__name__ for e in explainers]} ...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{file}", unit="sample"):
            if limit and idx >= limit:
                break

            text = row["counterfactual"]

            # label mapping
            raw_label = None
            if "pred_labels_history" in df.columns:
                history = str(row["pred_labels_history"]).split("|")
                raw_label = history[-1].strip()

            mapped_label = None
            if raw_label and raw_label != "":
                clean_label = raw_label.strip().lower()
                mapped_label = LABEL_MAP.get(dataset_name, {}).get(clean_label)
                if mapped_label is None:
                    print(f"Warning: unmapped label '{raw_label}' (file={file}, idx={idx}), skipping")
                    continue

            try:
                explanations = bench.explain(text, target=mapped_label) if mapped_label is not None else bench.explain(text)
                evaluations = bench.evaluate_explanations(explanations, target=mapped_label) if mapped_label is not None else bench.evaluate_explanations(explanations)

                explanation_details = {}
                for ev, ex in zip(evaluations, explanations):
                    topk_words = select_topk_words(topk, ex)
                    scores = ev.to_dict() if hasattr(ev, "to_dict") else str(ev)

                    explanation_details[ex.explainer] = {
                        "topk_words": topk_words,
                        "scores": scores
                    }

                result = {
                    "file": file,
                    "idx": idx,
                    "text": text,
                    "target": raw_label,
                    "details": explanation_details,
                }
                dataset_results[dataset_name].append(result)

            except Exception as e:
                print(f"Error (file={file}, idx={idx}): {e}")
                continue

            torch.cuda.empty_cache()
            gc.collect()

    # save json per dataset
    for dataset_name, results in dataset_results.items():
        if results:
            out_file = f"{output_prefix}_{dataset_name}.json"
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved {len(results)} results for {dataset_name} in {out_file}")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        default="./results_llama",
        help="Directory with counterfactual CSV files",
    )
    parser.add_argument(
        "--output_prefix",
        default="ferret_llama_summary",
        help="Prefix for output file path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Limit number of instances for debugging",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top important words to extract",
    )

    args, _ = parser.parse_known_args()

    explanation_generation_and_evaluation_by_ferret(
        results_dir=args.results_dir,
        output_prefix=args.output_prefix,
        limit=args.limit,
        topk=args.topk,
    )
