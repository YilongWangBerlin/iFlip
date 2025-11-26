"""
Optimized counterfactual generation script with configurable methods and datasets.
"""
from __future__ import annotations

import argparse
import re
import csv
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer as ClfTokenizer
from pathlib import Path

from .config import config
from .data import load_imdb, load_snli,load_agnews
from .model import load_generator
from .evaluate.metrics import predict_with_sliding_window, predict_scores_with_sliding_window, predict_with_confidence
from .lxt_utils import get_top_relevant_words_lxt,get_top_unrelevant_words_lxt
from .shap_utils import get_top_relevant_words_shap,get_top_unrelevant_words_shap
from .lime_utils import get_top_relevant_words_lime, get_top_unrelevant_words_lime
from .gradient_x_input_utils import get_top_relevant_words_gradxinput, get_top_unrelevant_words_gradxinput


START_TAG = "<cf>"
END_TAG = "</cf>"
SKIP_VALUES = {"your final edited review here", "...", "edited review", "", "\n","premise   : <your modified premise>","hypothesis: <your modified hypothesis>","premise   : <your revised premise>","hypothesis: <your revised hypothesis>"}

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"



TOPIC_CONTENT = {
    "WORLD": (
        "Include key terms such as diplomacy, conflict, international relations, and treaties. "
    
        "Mention entities like the United Nations, NATO, or national governments (e.g., China, the U.S., the EU). "
        "Use settings involving geopolitical events, summits, global crises, or foreign affairs."
    ),
    "SPORTS": (
        "Include key terms such as matches, championships, scores, and performance. "
        "Mention entities like professional teams (e.g., Manchester United, Lakers), athletes, leagues (e.g., NBA, FIFA), or tournaments (e.g., the Olympics). "
        "Use settings such as stadiums, post-game interviews, training camps, or major sporting events."
    ),
    "BUSINESS": (
        "Include key terms such as stock prices, quarterly earnings, market trends, and economic forecasts. "
        "Mention companies like Goldman Sachs, or startups, as well as financial institutions and CEOs. "
        "Use settings involving financial reports, board meetings, mergers & acquisitions, investment news, or market reactions."
    ),
    "SCI/TECH": (
        "Include key terms such as artificial intelligence, new technologies, scientific research, or innovation. "
        "Mention entities like NASA, Google, MIT, or specific products (e.g., iPhone, ChatGPT). "
        "Use settings such as tech conferences, research labs, product launches, or scientific studies."
    )
                    }

RELATION_GUIDANCE = {
    "contradiction": (
        "Introduce information that clearly conflicts with the other sentence – use negation, mutually exclusive facts, or temporal/quantitative clashes."
    ),
    "entailment": (
        "Add or reinforce details that must be true if the other sentence is true – specify clarifying attributes, causality, or synonymous phrasing."
    ),
    "neutral": (
        "Insert plausible but unrelated details that neither contradict nor logically follow from the other sentence – change topics, add new actors, or shift context."
    ),
            }



def _get_target_label(text: str, dataset: str) -> Tuple[str, str]:
    all_scores = predict_scores_with_sliding_window([text], clf, clf_tokenizer)[0]
    scores_sorted = sorted(all_scores, key=lambda x: x['score'], reverse=True)

    label_map = get_label_map(dataset)
    original_label_id = scores_sorted[0]['label']
    original_label = label_map.get(original_label_id, original_label_id)

    if dataset == "imdb":
        target_label = "neg" if original_label == "pos" else "pos"
    elif dataset in {"snli", "agnews"}:
        if len(scores_sorted) >= 2:
            target_label_id = scores_sorted[1]['label']
            target_label = label_map.get(target_label_id, target_label_id)
        else:
            raise ValueError(f"{dataset.upper()} classifier returned fewer than 2 labels.")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return original_label, target_label



def get_label_map(dataset: str) -> Dict[str, str]:
    if dataset == "imdb":
        return {"LABEL_0": "neg", "LABEL_1": "pos"}
    elif dataset == "snli":
        return {
            "LABEL_0": "entailment",
            "LABEL_1": "neutral",
            "LABEL_2": "contradiction"
        }
    elif dataset == "agnews":
        return {
            "LABEL_0": "world",
            "LABEL_1": "sports",
            "LABEL_2": "business",
            "LABEL_3": "sci/tech"
        }
    else:
        return {}



def leave_one_out_word_importance(text: str, clf_pipe, k: int = 5) -> List[Tuple[str, float]]:
    original_pred = clf_pipe(text)[0]
    original_label = original_pred['label']
    original_score = original_pred['score']
    
    words = re.findall(r"\b\w+\b", text)
    if len(words) < 2:
        return [(words[0], 0.0)] if words else []

    scores = []
    
    for i in range(len(words)):
        reduced_words = words[:i] + words[i+1:]
        reduced_text = " ".join(reduced_words)
        
        try:
            new_pred = clf_pipe(reduced_text)[0]
        except Exception:
            scores.append((words[i], 0.0))
            continue
        
        if new_pred['label'] == original_label:
            delta = original_score - new_pred['score']
        else:
            delta = original_score  

        scores.append((words[i], delta))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]



def _extract_between(text: str, start_tag: str, end_tag: str) -> str:
    pattern = re.compile(re.escape(start_tag) + r"(.*?)" + re.escape(end_tag), re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else text.strip()




def _predict_sentiment(text: str) -> str:
    raw_label = predict_with_sliding_window([text], clf, clf_tokenizer)[0]
    label_map = get_label_map(config.task_name)
    return label_map.get(raw_label, raw_label).lower()



def generate_counterfactuals(
    method: str,
    dataset_name: str,
    max_refinement_rounds: int,
    max_base_attempts: int,
    output_file: str,
    early_stop: bool = False,
    edit_target: str = "hypothesis",
) -> None:
    tokenizer, model = load_generator()
    
    PROMPT_LOG_DIR = Path("../scripts")
    PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    prompt_log_path = PROMPT_LOG_DIR / f"{Path(output_file).stem}.all_prompts.txt"

    
    
    if dataset_name == "imdb":
        dataset = load_imdb()
    elif dataset_name == "snli":
        dataset = load_snli()
    elif dataset_name == "agnews":
        dataset = load_agnews()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

        
    
    results: List[Dict[str, str]] = []

    for idx, example in enumerate(tqdm(dataset, desc="Generating", dynamic_ncols=True), start=1):
        if dataset_name == "snli":
            premise = example["premise"]
            hypothesis = example["hypothesis"]
            review_text = f"Premise   : {premise}\nHypothesis: {hypothesis}"
            ground_truth_label = example["label"]
        elif dataset_name == "imdb":
            review_text = example["text"]
            ground_truth_label = "positive" if example["label"] == 1 else "negative"
        elif dataset_name == "agnews":
            review_text = example["text"]
            ground_truth_label = example["label"]

                
        original_label, target_label = _get_target_label(review_text, dataset_name)
        orig_token_count = len(tokenizer.encode(review_text, add_special_tokens=False))
        stalled_rounds: List[int] = []  
        pred_history: List[str] = []

        print(f"\n=== [{idx}/{config.dataset_size}] {original_label} ➜ {target_label} ===", flush=True)
        print("Original (first 200 chars): "
              + review_text.replace("\n", " "), flush=True)


        #highlight = ", ".join(top_words)
        base_prompt = (
            f"A classifier has determined that the label of the following text is {original_label}.\n"
            f"Please flip it to {target_label} with minimal changes.\n"
            f"Wrap your answer in {START_TAG} ... {END_TAG}.\n\n"
            f"Original review:\n{review_text}"
        )
        
        if dataset_name == "agnews":


            base_prompt += (
                "\n\n"
                f"Hint: To reflect the topic of {target_label.upper()}, "
                f"{TOPIC_CONTENT.get(target_label.upper(), '')} "
                f"Also remove concepts more typical of {original_label.upper()} news."
            )

        
        elif dataset_name == "imdb":
            base_prompt += (
                "\n\n"
                f"Hint: Adjust the overall tone, wording, and emotional content to reflect a clearly {target_label.upper()} movie review "
                f"(e.g., change expressions of approval/disapproval, descriptions of acting, story, and enjoyment)."
            )


        elif dataset_name == "snli":

        
            if edit_target == "premise":
                base_prompt += (
                    "\n\n"
                    f"Hint: Modify ONLY the premise so that its relationship to the hypothesis clearly becomes {target_label.upper()}.\n"
                    f"- {RELATION_GUIDANCE [target_label]}\n"
                    "- You may negate, add, replace, or remove facts to achieve the desired relation.\n"
                    "- Return output in exactly this format (no extra text):\n"
                    f"{START_TAG}Premise   : <your revised premise>{END_TAG}\n"
                    f"Hypothesis: {hypothesis}\n"
                )
            else:  # edit_target == "hypothesis"
                base_prompt += (
                    "\n\n"
                    f"Hint: Modify ONLY the hypothesis so that its relationship to the premise clearly becomes {target_label.upper()}.\n"
                    f"- {RELATION_GUIDANCE [target_label]}\n"
                    "- You may change wording, add contradictions/supporting details, or insert unrelated information as appropriate.\n"
                    "- Return output in exactly this format (no extra text):\n"
                    f"{START_TAG}Hypothesis: <your revised hypothesis>{END_TAG}\n"
                )
        

        cf_text = None
        for attempt in range(1, max_base_attempts + 1):
            chat = [{"role": "user", "content": base_prompt}]
            model_input = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer([model_input], return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    do_sample=config.do_sample,
                    eos_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            candidate_cf = _extract_between(decoded, START_TAG, END_TAG)

            if candidate_cf.lower() in SKIP_VALUES:
                print(f"[Attempt {attempt}] placeholder output — retrying base prompt…", flush=True)
                if attempt == max_base_attempts:
                    print("Reached max attempts. Sample skipped.\n", flush=True)
                continue
            else:
                cf_text = candidate_cf.replace("\n", " ")
                final_cf_text = cf_text
                
                with prompt_log_path.open("a", encoding="utf-8") as pf:
                    pf.write(f"\n[Sample {idx} | Attempt {attempt} | method={method}]\n")
                    pf.write("=== BASE PROMPT ===\n")
                    pf.write(base_prompt + "\n\n")
                    pf.write("=== Extracted candidate_cf ===\n")
                    pf.write(candidate_cf + "\n")
                    pf.write("=" * 100 + "\n")
                    print("=== BASE CF ===\n")
                    print(candidate_cf + "\n")
                break
            

        if cf_text is None:
            continue
        
        last_text = cf_text
        
        if dataset_name == "snli":
            if edit_target == "premise":
                test_text = f"{cf_text}\nHypothesis: {hypothesis}"
            elif edit_target == "hypothesis":
                test_text = f"Premise: {premise}\n{cf_text}"
            pred_label = _predict_sentiment(test_text)
        else:
            pred_label = _predict_sentiment(cf_text)
            

        #pred_label = _predict_sentiment(cf_text)
        pred_history.append(pred_label)
        print(f"[Iteration 0] classifier prediction: {pred_label.upper()} | original: {original_label.upper()}", flush=True)
        
        if pred_label.upper() != original_label.upper():
            print(f"[Iteration 0] prediction: {pred_label.upper()}, original: {original_label.upper()}", flush=True)
            final_cf_text = cf_text
            if dataset_name == "snli":
                if edit_target == "premise":
                    final_cf_text = f"{cf_text}\nHypothesis: {hypothesis}"
                elif edit_target == "hypothesis":
                    final_cf_text = f"Premise: {premise}\n{cf_text}"
                    
            print("\nFinal CF:", final_cf_text, flush=True)
            print()
            if early_stop:
                print(f"[Iteration 0]Early stopping as the label has flipped; skip refinement for this sample.")
                results.append(
                    {
                        "original_review": review_text,
                        "counterfactual": final_cf_text,
                        "thinking_content": "",
                        "original_label": original_label,
                        "ground_truth_label": ground_truth_label,
                        "refinement_rounds": 0,
                        "pred_labels_history": "|".join(pred_history),
                        "orig_token_count": orig_token_count,
                        "stalled_rounds": [],
                    }
                )
                continue  

        # -------------------- Refinement loop --------------------
        for r in range(1, max_refinement_rounds + 1):
            if method == "conf":
                if dataset_name == "snli":
                    if edit_target == "premise":
                        test_text = f"{last_text}\nHypothesis: {hypothesis}"
                    elif edit_target == "hypothesis":
                        test_text = f"Premise: {premise}\n{last_text}"
                    raw_pred, conf = predict_with_confidence(test_text, clf, clf_tokenizer)
                else:
                    raw_pred, conf = predict_with_confidence(last_text, clf, clf_tokenizer)
                

                label_map  = get_label_map(dataset_name)
                pred_label = label_map.get(raw_pred.upper(), raw_pred).lower()
                print(f"Classifier: {pred_label.upper()} ({conf*100:.1f}% confident)\n")
                

                if pred_label != original_label: 
                    refinement_prompt = (
                        f"The classifier now correctly predicts your text as {pred_label.upper()} ({conf*100:.1f}% confidence), "
                        f"which is the desired label.\n"
                    
                        f"Your task is to minimize the edits compared to the original review, "
                        f"while still preserving the current label: {pred_label.upper()}.\n"
                        f"Try to make the revised text as close as possible to the original, "
                        f"but do not revert to the original label: {original_label.upper()}.\n\n"
                    
                        f"Wrap ONLY the final text in {START_TAG} ... {END_TAG}.\n\n"
                        f"Original review (classifier label {original_label}):\n{review_text}\n\n"
                        f"Current counterfactual:\n{last_text}"
                    )
    
    
                        
                        
                    if dataset_name == "snli":
                        if edit_target == "premise":
                            refinement_prompt += (
                                f"Hypothesis: {hypothesis}"
                                "\n\n"
                                f"Hint: Modify ONLY the premise so that its relationship to the hypothesis clearly becomes {target_label.upper()}.\n"
                                "- Return output in exactly this format (no extra text):\n"
                                f"{START_TAG}Premise   : <your revised premise>{END_TAG}\n"
                                f"Hypothesis: {hypothesis}\n"
                            )
                        else:  # edit_target == "hypothesis"
                            refinement_prompt += (
                                f"Premise: {premise}"
                                "\n\n"
                                f"Hint: Modify ONLY the hypothesis so that its relationship to the premise clearly becomes {target_label.upper()}.\n"
                                "- Return output in exactly this format (no extra text):\n"
                                f"{START_TAG}Hypothesis: <your revised hypothesis>{END_TAG}\n"
                            )

                else:
                    refinement_prompt = (
                        f"The classifier still predicts your text as {pred_label.upper()} ({conf*100:.1f}% confidence), "
                        f"but the goal is to make it {target_label.upper()}.\n"
                        f"Please revise the counterfactual to flip the label while keeping the text fluent and coherent.\n\n"
                        
                        f"Wrap ONLY the final text in {START_TAG} ... {END_TAG}.\n\n"
                        f"Original review (classifier label {original_label}):\n{review_text}\n\n"
                        f"Current counterfactual:\n{last_text}"
                    )
    
                    if dataset_name == "imdb":
                        refinement_prompt += (
                            "\n\n"
                            "Hint: Please revise the counterfactual by rewording or replacing emotionally charged phrases to better match the target sentiment.\n"
                            "Focus on adjusting subjective language, tone, and expressions of approval/disapproval to align with the new sentiment.\n\n"
                        )
    
                    elif dataset_name == "agnews":
            
                        refinement_prompt += (
                            "\n\n"
                            f"Hint: To reflect the topic of {target_label.upper()}, "
                            f"{TOPIC_CONTENT.get(target_label.upper(), '')} "
                            f"Also remove concepts more typical of {original_label.upper()} news."
                        )

                    
                    elif dataset_name == "snli":
                        
                    
                        if edit_target == "premise":
                            refinement_prompt += (
                                f"Hypothesis: {hypothesis}"
                                "\n\n"
                                f"Hint: Modify ONLY the premise so that its relationship to the hypothesis clearly becomes {target_label.upper()}.\n"
                                f"- {RELATION_GUIDANCE [target_label]}\n"
                                "- You may negate, add, replace, or remove facts to achieve the desired relation.\n"
                                "- Return output in exactly this format (no extra text):\n"
                                f"{START_TAG}Premise   : <your revised premise>{END_TAG}\n"
                                f"Hypothesis: {hypothesis}\n"
                            )
                        else:  # edit_target == "hypothesis"
                            refinement_prompt += (
                                f"Premise: {premise}"
                                "\n\n"
                                f"Hint: Modify ONLY the hypothesis so that its relationship to the premise clearly becomes {target_label.upper()}.\n"
                                f"- {RELATION_GUIDANCE [target_label]}\n"
                                "- You may change wording, add contradictions/supporting details, or insert unrelated information as appropriate.\n"
                                "- Return output in exactly this format (no extra text):\n"
                                f"{START_TAG}Hypothesis: <your revised hypothesis>{END_TAG}\n"
                                #f"Premise: {premise}\n"
                            )
    
    
                
               
                
                    
            elif pred_label != original_label and method in {"lxt","shap","lime","gradxinput"}:  
                # without early stopping for attribution feedback

                orig_words = re.findall(r"\b\w+\b", review_text)
                k = max(10, int(len(orig_words) * 0.10))
                #top_words = get_top_relevant_words_lxt(last_text, k=k)

                # Select relevance method
                if method == "lxt":
                    top_words = get_top_relevant_words_lxt(last_text, k)
                elif method == "shap":
                    top_words = get_top_relevant_words_shap(last_text, k)
                elif method == "lime":
                    top_words = get_top_relevant_words_lime(last_text, k)
                elif method == "gradxinput":
                    top_words = get_top_relevant_words_gradxinput(last_text, k)
    
                else:
                    top_words = []
                
                
                excluded_words = {"premise", "hypothesis",",",":",".","-","a","the"}
                top_words = [w for w in top_words if w.lower() not in excluded_words]
                print(f"[Refinement Round {r}] Top {k} relevant words: {', '.join(top_words)}", flush=True)
                #top_words = get_top_relevant_words(cf_text, k=5)
                highlight = ", ".join(top_words)
                refinement_prompt = (
                    f"The classifier now correctly predicts your text as {pred_label.upper()}, "
                    f"which is the desired label.\n"
                    f"Key words influencing this prediction: {highlight}.\n\n"
                
                    f"Your task is to minimize the edits compared to the original review, "
                    f"while still preserving the current label: {pred_label.upper()}.\n"
                    f"Try to make the revised text as close as possible to the original, "
                    f"but do not revert to the original label: {original_label.upper()}.\n\n"
                
                    f"Wrap ONLY the final text in {START_TAG} ... {END_TAG}.\n\n"
                    f"Original review (classifier label {original_label}):\n{review_text}\n\n"
                    f"Current counterfactual:\n{last_text}"
                )


                    
                    
                if dataset_name == "snli":
                
                    if edit_target == "premise":
                        refinement_prompt += (
                            f"Hypothesis: {hypothesis}"
                            "\n\n"
                            f"Hint: Modify ONLY the premise so that its relationship to the hypothesis clearly becomes {target_label.upper()}.\n"
                            "- Return output in exactly this format (no extra text):\n"
                            f"{START_TAG}Premise   : <your revised premise>{END_TAG}\n"
                            f"Hypothesis: {hypothesis}\n"
                        )
                    else:  # edit_target == "hypothesis"
                        refinement_prompt += (
                            f"Premise: {premise}"
                            "\n\n"
                            f"Hint: Modify ONLY the hypothesis so that its relationship to the premise clearly becomes {target_label.upper()}.\n"
                            "- Return output in exactly this format (no extra text):\n"
                            f"{START_TAG}Hypothesis: <your revised hypothesis>{END_TAG}\n"
                        )


                
            
            if method == "nl":
                # Step 1: ask for natural language feedback

                feedback_prompt = (
                    f"Analyze the current counterfactual and suggest improvements "
                    f"to achieve the target label {target_label.upper()}.\n"
                    f"You should make the smallest possible edits to the text while still achieving the target.\n"
                    f"Wrap your reasoning inside {THINK_OPEN}...{THINK_CLOSE}.\n\n"
                    f"Original (classifier label {original_label}):\n{review_text}\n\n"
                    f"Current counterfactual:\n{last_text}"
                )


                
                if dataset_name == "snli":
                    if edit_target == "premise":
                        feedback_prompt += (
                            f"Hint: Modify ONLY the premise so that its relationship to the hypothesis clearly becomes {target_label.upper()}.\n"
                        )
                    else:  # edit_target == "hypothesis"
                        feedback_prompt += (
                            f"Hint: Modify ONLY the hypothesis so that its relationship to the premise clearly becomes {target_label.upper()}.\n"
                        )
            
                chat_fb = [{"role": "user", "content": feedback_prompt}]
                fb_input = tokenizer.apply_chat_template(chat_fb, tokenize=False, add_generation_prompt=True)
                fb_inputs = tokenizer([fb_input], return_tensors="pt").to(model.device)
            
                with torch.no_grad():
                    fb_outputs = model.generate(
                        **fb_inputs,
                        max_new_tokens=config.max_new_tokens,
                        temperature=config.temperature,
                        top_k=config.top_k,
                        top_p=config.top_p,
                        do_sample=config.do_sample,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                fb_decoded = tokenizer.decode(fb_outputs[0], skip_special_tokens=True)
                feedback_text = _extract_between(fb_decoded, THINK_OPEN, THINK_CLOSE)
            
                print(f"[Refinement Round {r}] FEEDBACK:\n{feedback_text}\n", flush=True)
            
                # Step 2: generate new CF
                def make_refinement_prompt(feedback_text, last_text):
                    refinement_prompt = (
                        f"Based on the feedback below, revise the text so that it flips to {target_label.upper()}.\n\n"
                        f"Feedback:\n{feedback_text}\n\n"
                        f"Wrap ONLY the final text in {START_TAG}...{END_TAG}.\n\n"
                        f"Original (classifier label {original_label}):\n{review_text}\n\n"
                        f"Current counterfactual:\n{last_text}"
                    )
            
                    if dataset_name == "snli":
                        if edit_target == "premise":
                            refinement_prompt += (
                                f"Hypothesis: {hypothesis}"
                                "\n\n"
                                f"Hint: Modify ONLY the premise so that its relationship to the hypothesis clearly becomes {target_label.upper()}.\n"
                                "Return output in exactly this format (no extra text):\n"
                                f"{START_TAG}Premise   : <your revised premise>{END_TAG}\n"
                                f"Hypothesis: {hypothesis}\n"
                            )
                        else:  # edit_target == "hypothesis"
                            refinement_prompt += (
                                f"Premise: {premise}"
                                "\n\n"
                                f"Hint: Modify ONLY the hypothesis so that its relationship to the premise clearly becomes {target_label.upper()}.\n"
                                "Return output in exactly this format (no extra text):\n"
                                f"{START_TAG}Hypothesis: <your revised hypothesis>{END_TAG}\n"
                            )
                    return refinement_prompt
            
                refinement_prompt = make_refinement_prompt(feedback_text, last_text)
            
                def generate_cf(refinement_prompt):
                    chat_cf = [{"role": "user", "content": refinement_prompt}]
                    cf_input = tokenizer.apply_chat_template(chat_cf, tokenize=False, add_generation_prompt=True)
                    cf_inputs = tokenizer([cf_input], return_tensors="pt").to(model.device)
            
                    with torch.no_grad():
                        cf_outputs = model.generate(
                            **cf_inputs,
                            max_new_tokens=config.max_new_tokens,
                            temperature=config.temperature,
                            top_k=config.top_k,
                            top_p=config.top_p,
                            do_sample=config.do_sample,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    refined_decoded = tokenizer.decode(cf_outputs[0], skip_special_tokens=True)
                    new_cf_text = _extract_between(refined_decoded, START_TAG, END_TAG)
                    return new_cf_text, refined_decoded
            

                new_cf_text, refined_decoded = generate_cf(refinement_prompt)

                if not new_cf_text or new_cf_text.lower() in SKIP_VALUES:
                    print(f"[Refinement Round {r}] invalid output, retrying...\n", flush=True)
                    new_cf_text, refined_decoded = generate_cf(refinement_prompt)
            
                with prompt_log_path.open("a", encoding="utf-8") as pf:
                    pf.write(f"\n[Sample {idx} | Round {r} | method={method}]\n")
                    pf.write("=== REFINEMENT PROMPT ===\n")
                    pf.write(refinement_prompt + "\n\n")
                    pf.write("=== Extracted new_cf_text ===\n")
                    pf.write(new_cf_text + "\n")
                    pf.write("=" * 100 + "\n")



            

            else:  
                
                orig_words = re.findall(r"\b\w+\b", review_text)
                k = max(10, int(len(orig_words) * 0.10))
                #top_words = get_top_relevant_words_lxt(last_text, k=k)
                
                
                # Select relevance method
                if method == "lxt":
                    top_words = get_top_relevant_words_lxt(last_text, k)
                elif method == "shap":
                    top_words = get_top_relevant_words_shap(last_text, k)
                elif method == "lime":
                    top_words = get_top_relevant_words_lime(last_text, k)
                elif method == "gradxinput":
                    top_words = get_top_relevant_words_gradxinput(last_text, k)
    
                else:
                    top_words = []
                
                
                excluded_words = {"premise", "hypothesis",",",":",".","-","a","the"}
                top_words = [w for w in top_words if w.lower() not in excluded_words]
                print(f"[Refinement Round {r}] Top {k} relevant words: {', '.join(top_words)}", flush=True)
                #top_words = get_top_relevant_words(cf_text, k=5)
                highlight = ", ".join(top_words)
                refinement_prompt = (
                    f"The classifier still predicts your text as {pred_label.upper()}, "
                    f"but the goal is to make it {target_label.upper()}.\n"
                    f"Key words influencing this prediction: {highlight}.\n\n"
                    
                    
                    f"Wrap ONLY the final text in {START_TAG} ... {END_TAG}.\n\n"
                    f"Original review (classifier label {original_label}):\n{review_text}\n\n"
                    f"Current counterfactual:\n{last_text}"
                )
                if dataset_name == "imdb":
                    refinement_prompt += (
                        "\n\n"
                        "Hint: Please revise the counterfactual by rewording or replacing emotionally charged phrases to better match the target sentiment.\n"
                        "Focus on adjusting subjective language, tone, and expressions of approval/disapproval to align with the new sentiment.\n\n"
                    )

                elif dataset_name == "agnews":
                    
        
                    refinement_prompt += (
                        "\n\n"
                        f"Hint: To reflect the topic of {target_label.upper()}, "
                        f"{TOPIC_CONTENT.get(target_label.upper(), '')} "
                        f"Also remove concepts more typical of {original_label.upper()} news."
                    )

        
                elif dataset_name == "snli":
                
                    if edit_target == "premise":
                        refinement_prompt += (
                            f"Hypothesis: {hypothesis}"
                            "\n\n"
                            f"Hint: Modify ONLY the premise so that its relationship to the hypothesis clearly becomes {target_label.upper()}.\n"
                            f"- {RELATION_GUIDANCE[target_label]}\n"
                            "- You may negate, add, replace, or remove facts to achieve the desired relation.\n"
                            "- Return output in exactly this format (no extra text):\n"
                            f"{START_TAG}Premise   : <your revised premise>{END_TAG}\n"
                            f"Hypothesis: {hypothesis}\n"
                        )
                    else:  # edit_target == "hypothesis"
                        refinement_prompt += (
                            f"Premise: {premise}"
                            "\n\n"
                            f"Hint: Modify ONLY the hypothesis so that its relationship to the premise clearly becomes {target_label.upper()}.\n"
                            f"- {RELATION_GUIDANCE[target_label]}\n"
                            "- You may change wording, add contradictions/supporting details, or insert unrelated information as appropriate.\n"
                            "- Return output in exactly this format (no extra text):\n"
                            f"{START_TAG}Hypothesis: <your revised hypothesis>{END_TAG}\n"
                            #f"Premise: {premise}\n"
                        )


            chat2 = [{"role": "user", "content": refinement_prompt}]
            refined_input = tokenizer.apply_chat_template(chat2, tokenize=False,
                                                          add_generation_prompt=True)
            refined_inputs = tokenizer([refined_input], return_tensors="pt").to(model.device)

            with torch.no_grad():
                refined_outputs = model.generate(
                    **refined_inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    do_sample=config.do_sample,
                    eos_token_id=tokenizer.eos_token_id,
                )

            refined_decoded = tokenizer.decode(refined_outputs[0], skip_special_tokens=True)
            new_cf_text = _extract_between(refined_decoded, START_TAG, END_TAG)


            PROMPT_LOG_DIR = Path("../scripts")
            PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
            
            prompt_log_path = PROMPT_LOG_DIR / f"{Path(output_file).stem}._prompts.txt"
            with prompt_log_path.open("a", encoding="utf-8") as pf:
                pf.write(f"[Sample {idx} | Round {r} | method={method}]\n")
                
                pf.write("=== MODEL OUTPUT ===\n")
                pf.write(refined_decoded.rstrip() + "\n")
                pf.write("=" * 80 + "\n")


            if new_cf_text.strip() == cf_text.strip():
                stalled_rounds.append(r)  
                print(f"[Refinement Round {r}] identical cf.", flush=True)


            if new_cf_text.lower() in SKIP_VALUES:
                print(f"[Refinement Round {r}] invalid output.\n", flush=True)
                
                #cf_text = 'invalid'
                stalled_rounds.append(r)
                continue
            else:
                cf_text = new_cf_text
                last_text = cf_text
                final_cf_text = cf_text
                
            if dataset_name == "snli":
                if edit_target == "premise":
                    test_text = f"{cf_text}\nHypothesis: {hypothesis}"
                elif edit_target == "hypothesis":
                    test_text = f"Premise: {premise}\n{cf_text}"
                pred_label = _predict_sentiment(test_text)
            else:
                pred_label = _predict_sentiment(cf_text)

            #pred_label = _predict_sentiment(cf_text)
            pred_history.append(pred_label)
            
            #if idx <= 9:
                #print(f"[Refinement Round {r}] CF:", cf_text, flush=True)
                #print(f"[Refinement Round {r}]:", refined_decoded, flush=True)
            print(f"[Refinement Round {r}] CF:", cf_text, flush=True)
            print(f"[Refinement Round {r}] classifier prediction: {pred_label.upper()} "
                  f"| original: {original_label.upper()}", flush=True)
            
            if early_stop and pred_label.upper() != original_label.upper():
                print("Early stopping – label successfully flipped.")
                break
                        
        if dataset_name == "snli":
            if edit_target == "premise":
                #final_cf_text = f"{cf_text}\nHypothesis: {hypothesis}"
                if "Hypothesis" not in cf_text:
                    final_cf_text = f"{cf_text}\nHypothesis: {hypothesis}"
                else:
                    final_cf_text = cf_text
            elif edit_target == "hypothesis":
                #final_cf_text = f"Premise: {premise}\n{cf_text}"
                if "Premise" not in cf_text:
                    final_cf_text = f"Premise: {premise}\n{cf_text}"
                else:
                    final_cf_text = cf_text
                
        print("\nFinal CF:", final_cf_text, flush=True)
        print("\n")   

        thinking_segment = (
            _extract_between(refined_decoded if 'refined_decoded' in locals() else decoded,
                             THINK_OPEN, THINK_CLOSE)
            if THINK_OPEN in (refined_decoded if 'refined_decoded' in locals() else decoded)
            else ""
        )

        results.append(
            {
                "original_review": review_text,
                "counterfactual": final_cf_text,
                "thinking_content": thinking_segment,
                "original_label": original_label,
                "ground_truth_label": ground_truth_label,
                "refinement_rounds": len(pred_history) - 1, 
                "pred_labels_history": "|".join(pred_history),
                "orig_token_count": orig_token_count,  
                "stalled_rounds": stalled_rounds,
            }
        )

    # Save results to CSV
    with Path(output_file).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate counterfactuals.")
    parser.add_argument("--method", choices=["lxt", "shap", "conf","lime", "gradxinput", "nl"], required=True, help="Relevance method")
    parser.add_argument("--dataset", choices=["imdb", "snli","agnews"], required=True, help="Dataset name")
    parser.add_argument("--max_refinement_rounds", type=int, default=5, help="Max refinement rounds")
    parser.add_argument("--max_base_attempts", type=int, default=3, help="Max base attempts")
    parser.add_argument("--output_file", type=str, default="counterfactuals.csv", help="Output CSV file")
    parser.add_argument("--early_stop", action="store_true", help="Enable early stopping if sentiment flips")
    parser.add_argument(
        "--edit_target", 
        choices=["premise", "hypothesis"], 
        default="hypothesis", 
        help="Which part to edit in SNLI: premise or hypothesis"
    )


    args = parser.parse_args()
    config.task_name = args.dataset 
    
    clf = pipeline("text-classification", model=config.classifier_model(), device=0, return_all_scores=True)
    clf_tokenizer = ClfTokenizer.from_pretrained(config.classifier_model())
    clf_tokenizer.model_max_length = 512


    generate_counterfactuals(
        method=args.method,
        dataset_name=args.dataset,
        max_refinement_rounds=args.max_refinement_rounds,
        max_base_attempts=args.max_base_attempts,
        output_file=args.output_file,
        early_stop=args.early_stop,
        edit_target=args.edit_target,
    )