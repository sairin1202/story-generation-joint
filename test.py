import torch
import torch.nn as nn
from dataset import get_test_loader
from model import get_model
from optimizer import get_optimizer
from tokenizer import get_tokenizer, input_tokenize, target_tokenize
import pickle
from tqdm import tqdm
from generate import generate
import os
import json
from tqdm import tqdm
device = torch.device('cuda') 
from nlgeval import NLGEval
nlgeval = NLGEval()  # loads the models


def test(story_model, storyline_model, dataset, tokenizer):
    with torch.no_grad():
        story_model.eval()
        storyline_model.eval()
        references = []
        hypothesis = []
        with open('data/storyline_dev.pkl', 'rb') as f:
            data = pickle.load(f)
        with open('data/story_dev.pkl', 'rb') as f:
            story_data = pickle.load(f)
        for index in range(len(data['input'])):
            inputs = data['input'][index]
            targets = story_data['target'][index]
            storyline = generate_storyline_one(storyline_model, [inputs], tokenizer)
            res = generate_story_one(story_model, [storyline], tokenizer)
            hypothesis.append(res.lower())
            references.append(targets.lower())
            print('-'*50)
            print('output', res)
            print('target', targets)
            if index == 100:
                break
            # print(references[0])
        metrics_dict = nlgeval.compute_metrics([references], hypothesis)
        print(metrics_dict)

def generate_storyline_one(model, input_sentence, tokenizer):
    inputs = input_tokenize(tokenizer, input_sentence)
    if isinstance(model, nn.DataParallel):
        model = model.module  

    generated_ids = model.generate(inputs["input_ids"].cuda(), attention_mask=inputs["attention_mask"].cuda(), max_length=100, do_sample=False, num_beams=1, early_stopping=True)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def generate_story_one(model, input_sentence, tokenizer):
    inputs = input_tokenize(tokenizer, input_sentence)
    if isinstance(model, nn.DataParallel):
        model = model.module  
    generated_ids = model.generate(
                    inputs["input_ids"].cuda(), 
                    attention_mask=inputs["attention_mask"].cuda(),
                    decoder_start_token_id = tokenizer.pad_token_id,
                    max_length=200,
                    ) 
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def main():
    batch_size = 1
    dataset = get_test_loader(batch_size)
    # get model
    story_model = get_model()
    story_model = story_model.to(device)
    story_model.load_state_dict(torch.load('../story/model/3.pt'))
    # story_model.load_state_dict(torch.load('model/story9.pt'))
    story_model.eval()

    storyline_model = get_model()
    storyline_model = storyline_model.to(device)
    storyline_model.load_state_dict(torch.load('../storyline/model/3.pt'))
    # storyline_model.load_state_dict(torch.load('model/storyline9.pt'))
    storyline_model.eval()

    # get tokenizer
    tokenizer = get_tokenizer()
    test(story_model, storyline_model, dataset, tokenizer)


if __name__ == "__main__":
    main()
