from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

from transformer.model.transformer import build_transformer


# Use word level tokenization
def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item["translation"][lang]


def build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        # Train tokenizer with special tokens:
        # unknown, padding, start of sentence, end of sentence
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )

        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_dataset(config):
    dataset_raw = load_dataset(
        "opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train"
    )

    # Build tokenizers
    tokenizer_src = build_tokenizer(config, dataset_raw, config["lang_src"])
    tokenizer_tgt = build_tokenizer(config, dataset_raw, config["lang_tgt"])

    # Keep 90% for training and 10% for validation
    train_size = int(0.9 * len(dataset_raw))
    val_size = len(dataset_raw) - train_size
    train_dataset, val_dataset = random_split(dataset_raw, [train_size, val_size])

    ...
