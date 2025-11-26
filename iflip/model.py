"""
Model & tokenizer loading helpers.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .config import config

def load_generator():
    print('#####loading model#####')

    model_id = config.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_id ,
                                              trust_remote_code=True,
                                              local_files_only=True, )
    quant_kwargs = {}
    if config.load_in_8bit:
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id ,
        torch_dtype="auto",
        device_map="auto",
        **quant_kwargs
    )
    return tokenizer, model