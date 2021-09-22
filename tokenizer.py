
import torch
from transformers import BartTokenizer


def get_tokenizer():    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True)
    return tokenizer


def input_tokenize(tokenizer, batch):
    batch_token = tokenizer(batch, padding='max_length', return_tensors="pt", truncation=True, max_length=150)
    return batch_token

def target_tokenize(tokenizer, batch):
    batch_token = tokenizer(batch, padding='max_length', return_tensors="pt", truncation=True, max_length=100)
    return batch_token