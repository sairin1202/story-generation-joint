import torch
from transformers import BartForConditionalGeneration




def get_model():
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    return model

