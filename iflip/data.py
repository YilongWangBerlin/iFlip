"""
Dataset helpers.
"""
from datasets import load_dataset
from .config import config

def load_imdb(split: str = "test"):
    """
    Load and shuffle IMDB split; subsample to *config.dataset_size*.
    """
    print('#####loading data#####')
    ds = load_dataset("imdb", split=split).shuffle(seed=config.seed)
    
    return ds.select(range(config.dataset_size))


def load_snli(split: str = "test"):
    """
    Load and shuffle SNLI split; subsample to *config.dataset_size*.
    """
    print('#####loading SNLI data#####')
    ds = load_dataset("snli", split=split).shuffle(seed=config.seed)
    return ds.select(range(config.dataset_size))

def load_agnews(split: str = "test"):
    """
    Load and shuffle AG News split; subsample to *config.dataset_size*.
    """
    print('#####loading AG News data#####')
    ds = load_dataset("ag_news", split=split).shuffle(seed=config.seed)
    return ds.select(range(config.dataset_size))
