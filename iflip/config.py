"""
Centralised configuration for the project.

All hyper-parameters and file paths live here so that you only need to change
them in one place.
"""
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class GenerationConfig:
    # General
    seed: int = 3407
    dataset_size: int = 10
    task_name: str = "imdb"
    
    # Paths
    hf_home: str = "your local path to huggingface"
    model_name: str = "allenai/OLMo-2-1124-7B-Instruct"
    #model_name: str = "meta-llama/Llama-3.3-70B-Instruct"


    # Quantisation
    load_in_8bit: bool = False   # set to False for full-precision

    # Text generation
    max_new_tokens: int = 4096
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True

    # Classifier (for evaluation)
    def classifier_model(self) -> str:
        if self.task_name == "imdb":
            return "textattack/bert-base-uncased-imdb"
            #return "textattack/roberta-base-imdb"
        elif self.task_name == "snli":
            return "textattack/bert-base-uncased-snli"
            #return "utahnlp/snli_roberta-base_seed-2"
        elif self.task_name == "agnews":
            return "textattack/bert-base-uncased-ag-news"
            #return "textattack/roberta-base-ag-news"
    
    token_window_size: int = 512   # window length for sliding-window voting
    token_stride: int = 256        # stride between windows

config = GenerationConfig()