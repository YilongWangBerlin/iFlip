
import sys
import os
import gc
import re
import csv
import argparse
import traceback
import pandas as pd
import tiktoken

import torch
from typing import List, Dict, Any, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

hf_home: str = "your local path to cache huggingface models"
model_name: str = "allenai/OLMo-2-1124-7B-Instruct"

os.environ.setdefault("HF_HOME", hf_home)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))

def count_tokens(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def truncate_text(text, encoding_name='o200k_base', max_tokens=256):
    encoding = tiktoken.get_encoding(encoding_name)
    tokenized_text = encoding.encode(text)
    num_tokens = len(tokenized_text)

    if num_tokens > max_tokens:
        num_tokens_to_keep = max_tokens - 1
        truncated_tokenized_text = {
            'tokens': tokenized_text[:num_tokens_to_keep],
            'is_truncated': True
        }
        truncated_text = encoding.decode(truncated_tokenized_text['tokens'])
        return truncated_text, True
    else:
        return text, False

def generate_set_from_csv(input_string):
    elements = input_string.split(',')
    result_set = set(element.strip() for element in elements)
    return result_set

def parse_text_within_tags(input_text):
    pattern = r'<[nN]ew>(.*?)<\/[nN]ew>'
    matches = re.findall(pattern, input_text, re.DOTALL)

    if not matches:
        pattern = r'<[nN]ew>(.*)'
        matches = re.findall(pattern, input_text, re.DOTALL)

    return [m.strip() for m in matches]

def generate_comma_separated_string(input_set):
    return ', '.join(str(element) for element in input_set)

def get_opposite_label(pred_label, task):
    if task=='snli':
        opp_map = {'contradiction':'entailment', 'entailment':'contradiction', 'neutral':'contradiction'}
        return opp_map[pred_label]
    elif task=='imdb':
        opp_map = {'positive': 'negative', 'negative':'positive', 0.0:1.0, 1.0:0.0}
        return opp_map[pred_label]
    elif task=='ag_news':
        opp_map = {'the world': 'business', 'business':'sports', 'sports':'the world', 'science/tech': 'sports'}
        return opp_map[pred_label]


class LocalChatModel:
    def __init__(self,
                 model_id: str,
                 load_4bit: bool = False,   # 不用 4bit
                 dtype: Optional[torch.dtype] = None):

        self.model_id = model_id
        quantization_config = None
        if dtype is None:
            if torch.cuda.is_available():
                dtype = torch.bfloat16
            else:
                dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype,
            quantization_config=None,
            trust_remote_code=True
        )
        
        #self.model.config.max_position_embeddings = 4096
        if hasattr(self.model.config, "max_position_embeddings"):
            self.model.config.max_position_embeddings = 4096

        self.tokenizer.model_max_length = 4096
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )


        self.has_chat_template = hasattr(self.tokenizer, "apply_chat_template") and callable(
            getattr(self.tokenizer, "apply_chat_template")
        )

    def format_messages(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if self.has_chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            except Exception:
                pass

        sys_part = f"[SYSTEM]\n{system_prompt}\n\n" if system_prompt else ""
        usr_part = f"[USER]\n{user_prompt}\n\n[ASSISTANT]\n"
        return sys_part + usr_part

    @torch.inference_mode()
    def chat(self,
             system_prompt: str,
             user_prompt: str,
             temperature: float = 0.7,
             top_p: float = 0.8,
             max_new_tokens: int = 512) -> str:
        prompt_text = self.format_messages(system_prompt, user_prompt)
        outputs = self.generator(
            prompt_text,
            max_new_tokens=max_new_tokens,
            max_length=4096,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            return_full_text=False
        )
        return outputs[0]["generated_text"]


    def format_messages(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if self.has_chat_template:
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                return text
            except Exception:
                pass

        sys_part = f"[SYSTEM]\n{system_prompt}\n\n" if system_prompt else ""
        usr_part = f"[USER]\n{user_prompt}\n\n[ASSISTANT]\n"
        return sys_part + usr_part

    @torch.inference_mode()
    def chat(self,
             system_prompt: str,
             user_prompt: str,
             temperature: float = 0.7,
             top_p: float = 0.8,
             max_new_tokens: int = 512) -> str:
        prompt_text = self.format_messages(system_prompt, user_prompt)

        outputs = self.generator(
            prompt_text,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )

        text = outputs[0]["generated_text"]
        return text

def main(args):
    file_path_map = {
        'distilbert-snli': "./data-files/distilbert-snli-triples.csv",
        'distilbert-imdb': "./data-files/distilbert-imdb-triples.csv",
        'distilbert-ag_news': "./data-files/distilbert-ag_news-triples.csv"
    }

    test_type = args.test_type
    task = args.task

    all_data = []

    with open(file_path_map[test_type], 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # skip header

        for row in csv_reader:
            if task == "snli":
                premise, hypothesis, gt, pred = row
                all_data.append(row)
            else:
                text, gt, pred = row
                all_data.append(row)

    print(f"Loading local model: {model_name}")
    local_llm = LocalChatModel(model_id=model_name, load_4bit=False, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)


    cf_explanations = []
    noisy_explanations = []
    parsing_fail = 0

    if task == "snli":
        column_names = ["premise", "hypothesis", "ground_truth", "y_pred_original"]
    else:
        column_names = ["original_text", "ground_truth", "y_pred_original"]

    out_df = pd.DataFrame(all_data[:args.num_samples], columns=column_names)

    task_desc = {
        'snli': "natural language inference (premise-hypothesis pair classification) on the SNLI dataset",
        'imdb': "sentiment classification on the IMDB dataset",
        'ag_news': "news topic classification on the AG News dataset"
    }

    for i, instance in enumerate(all_data[:args.num_samples]):
        sys.stdout.write("data instance: "+str(i)+"\n")
        sys.stdout.flush()

        if task == "snli":
            premise, hypothesis, gt, pred = instance
            truncated_premise, _ = truncate_text(premise)
            truncated_hypothesis, _ = truncate_text(hypothesis)
            text_for_prompt = f"Premise: {truncated_premise}\nHypothesis: {truncated_hypothesis}"
        else:
            text, gt, pred = instance
            truncated_text_val, _ = truncate_text(text)
            text_for_prompt = truncated_text_val

        opposite_label = get_opposite_label(gt, task=task)

        if task == "snli":
            premise, hypothesis, gt, pred = instance
            truncated_premise, _ = truncate_text(premise)
            truncated_hypothesis, _ = truncate_text(hypothesis)

            if args.modify == "premise":
                initial_prompt = (
                    f"In the task of {task_desc[task]}, "
                    f"a trained black-box classifier correctly predicted the label '{pred}' "
                    f"for the following pair.\n"
                    f"Premise: {premise}\nHypothesis: {hypothesis}\n\n"
                    f"Generate a counterfactual explanation by only modifying the premise, so that the label changes from '"+pred+"' to '"+opposite_label+"'. Use the following definition of 'counterfactual explanation': \"A counterfactual explanation reveals what should have been different in an instance to observe a diverse outcome.\". Enclose the generated text within \"<new>\" tags.\nThe generated text must explicitly include only a 'Premise:' field and must not contain any 'Hypothesis:' field."
                    f"Your response must strictly follow this format:\n"
                    f"<new>\nPremise: <your modified premise here>\n</new>\n\n"
                )
            elif args.modify == "hypothesis":
                initial_prompt = (
                    f"In the task of {task_desc[task]}, "
                    f"a trained black-box classifier correctly predicted the label '{pred}' "
                    f"for the following pair.\n"
                    f"Premise: {premise}\nHypothesis: {hypothesis}\n\n"
                    f"Generate a counterfactual explanation by only modifying the hypothesis, so that the label changes from '"+pred+"' to '"+opposite_label+"'. Use the following definition of 'counterfactual explanation': \"A counterfactual explanation reveals what should have been different in an instance to observe a diverse outcome.\". Enclose the generated text within \"<new>\" tags.\nThe generated text must explicitly include only a 'Hypothesis:' field and must not contain any 'Premise:' field."
                )
        else:
                
                initial_prompt = "In the task of "+task_desc[task]+", a trained black-box classifier correctly predicted the label '"+pred+" for the following text. Generate a counterfactual explanation from the input text, so that the label changes from '"+pred+"' to '"+opposite_label+"'. Use the following definition of 'counterfactual explanation': \"A counterfactual explanation reveals what should have been different in an instance to observe a diverse outcome.\". Enclose the generated text within \"<new>\" tags.\n---\nInput Text: "+text+"\n---\n"



        system_prompt = (
            "Follow the instructions as closely as possible. Output exactly in the format that is specified by the user."
        )

        try:
            cf_response = local_llm.chat(
                system_prompt=system_prompt,
                user_prompt=initial_prompt,
                temperature=0.7,
                top_p=0.8,
            )
        except Exception as e:
            traceback.print_exc()
            cf_response = ""

        gc.collect()

        parsed = parse_text_within_tags(cf_response)
        if parsed == []:
            cf_explanations.append('null')
            noisy_explanations.append(cf_response if cf_response else 'null')
            parsing_fail += 1
        else:
            cf_explanations.append(parsed[0].strip())
            noisy_explanations.append('null')

    print('Done!')
    print('Parsing Fail Count: ', parsing_fail)

    try:
        if task == "snli":
            if args.modify == "premise":
                out_df['counterfactual_premise'] = cf_explanations
            elif args.modify == "hypothesis":
                out_df['counterfactual_hypothesis'] = cf_explanations
            else:
                out_df['counterfactual_text'] = cf_explanations  # for 'either'
            out_df['noisy_cf'] = noisy_explanations

            safe_model_name = model_name.replace("/", "_")
            out_df.to_csv(f"{test_type}-{args.modify}-{safe_model_name}-cf-explanations.csv", index=False)
        else:
            out_df['counterfactual_text'] = cf_explanations
            out_df['noisy_cf'] = noisy_explanations

            safe_model_name = model_name.replace("/", "_")
            out_df.to_csv(f"{test_type}-{safe_model_name}-cf-explanations.csv", index=False)
    except Exception as e:
        print(cf_explanations)
        print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate counterfactual explanations using a local HF model (OLMo-2-1124-7B-Instruct)"
    )
    parser.add_argument("--model", type=str, default="local-olmo2", help="(ignored) kept for CLI compatibility")
    parser.add_argument("--task", type=str, default="snli", choices=['snli', 'ag_news', 'imdb'], help="Target task")
    parser.add_argument("--test_type", type=str, default="distilbert-snli", help="Test CSV type (should match task)")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to process")
    parser.add_argument("--modify", type=str, default="either", choices=["premise", "hypothesis", "either"],
                        help="For SNLI: choose which part to modify")

    args = parser.parse_args()
    main(args)

