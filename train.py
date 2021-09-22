
import torch
import torch.nn as nn
from dataset import get_loader, get_test_loader
from model import get_model
from optimizer import get_optimizer
from tokenizer import get_tokenizer, input_tokenize, target_tokenize
import pickle
from tqdm import tqdm
from generate import generate
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device('cuda') 
from nlgeval import NLGEval
nlgeval = NLGEval()  # loads the models

def cnt(loss):
    cnter = torch.Tensor([torch.count_nonzero(l) for l in loss])
    cnter.require_grad = False
    return cnter.to(device)


def shift_tokens_right(input_ids, pad_token_id):
  """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
  """
  prev_output_tokens = input_ids.clone()
  index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
  prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
  prev_output_tokens[:, 1:] = input_ids[:, :-1]
  return prev_output_tokens

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])


def test(model, tokenizer):
    with torch.no_grad():
        model.eval()
        references = []
        hypothesis = []
        dataset = get_test_loader(32)
        for i, (inputs, targets) in enumerate(tqdm(dataset)):
            res = generate_one(model, inputs, tokenizer)
            hypothesis.extend(res)
            references.extend(targets)
            if i == 0:
                for _ in range(10):
                    print('-'*50)
                    print('output', res[_])
                    print('target', targets[_])
                break
            # print(references[0])
        metrics_dict = nlgeval.compute_metrics([references], hypothesis)
        print(metrics_dict)


def generate_one(model, input_sentence, tokenizer):
    inputs = input_tokenize(tokenizer, input_sentence)
    if isinstance(model, nn.DataParallel):
        model = model.module  
    generated_ids = model.generate(
                    inputs["input_ids"].cuda(), 
                    attention_mask=inputs["attention_mask"].cuda(),
                    do_sample=False, 
                    decoder_start_token_id = tokenizer.pad_token_id,
                    max_length=200,
                    ) 
    return [tokenizer.decode(g_ids, skip_special_tokens=True) for g_ids in generated_ids]

def train_model(story_model, storyline_model, dataset, storyline_optimizer, storyline_lr_scheduler, story_optimizer, story_lr_scheduler, tokenizer, epochs):
    # test(model, tokenizer)
    for epoch in range(epochs): 
        story_model.train()
        storyline_model.train()
        total_loss_list = []       
        for i, (storyline_inputs, storyline_targets, story_inputs, story_targets) in enumerate(tqdm(dataset)):
            story_optimizer.zero_grad()  
            storyline_optimizer.zero_grad()   
            input_batch = input_tokenize(tokenizer, storyline_inputs)
            input_ids = input_batch['input_ids'].to(device)
            attention_mask = input_batch['attention_mask'].to(device)
            output_batch = target_tokenize(tokenizer, storyline_targets)
            output_ids = output_batch['input_ids'].to(device)
            # print(input_ids.size(), attention_mask.size())
            decoder_input_ids = shift_tokens_right(output_ids, tokenizer.pad_token_id)
            storyline_output = storyline_model(input_ids, attention_mask=attention_mask,
                                decoder_input_ids=decoder_input_ids)
            lm_logits = storyline_output[0]
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
            storyline_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), output_ids.view(-1))
            storyline_loss = storyline_loss.view(output_ids.size(0), -1)

            # generated storyline as input --> story loss
            generated_storyline_ids = storyline_model.module.generate(input_ids, attention_mask=attention_mask, max_length=100, do_sample=False, num_beams=1, early_stopping=True)
            generated_storyline = tokenizer.batch_decode(generated_storyline_ids, skip_special_tokens=True)
            generated_storyline_ids = tokenizer(generated_storyline, padding=True, truncation=True, return_tensors="pt")
            output_story_ids = target_tokenize(tokenizer, story_targets)
            output_story_ids = output_story_ids['input_ids'].to(device)
            decoder_input_ids = shift_tokens_right(output_story_ids, tokenizer.pad_token_id)
            story_joint_output = story_model(generated_storyline_ids['input_ids'].to(device), attention_mask=generated_storyline_ids['attention_mask'].to(device),
                                decoder_input_ids=decoder_input_ids)
            
            lm_logits = story_joint_output[0]
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
            story_joint_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), output_story_ids.view(-1))


            reward = story_joint_loss.clone()
            story_joint_loss = story_joint_loss.mean()
            # reinforcement learning: storyline-->story loss by scaling
            reward = reward.view(output_story_ids.size(0), -1)
            cnter = cnt(reward)
            reward = reward.sum(dim=-1)
            reward = reward/cnter
            reward = reward/reward.sum()
            reward.require_grad = False
            storyline_loss = storyline_loss*reward.unsqueeze(-1)
            storyline_loss = storyline_loss.mean(-1).sum()
            

            loss = storyline_loss + story_joint_loss
        
            loss.backward()
            story_optimizer.step()
            story_lr_scheduler.step()
            storyline_optimizer.step()
            storyline_lr_scheduler.step()

            total_loss_list.append(loss.item())

        print(f"Epoch {epoch} ce loss : ", sum(total_loss_list)/len(total_loss_list))
        if epoch % 1 == 0:
            torch.save(storyline_model.module.state_dict(), f'model/storyline{epoch}.pt')
            torch.save(story_model.module.state_dict(), f'model/story{epoch}.pt')
            # test(storyline_model, story_model, story, tokenizer)



def main():
    # get data loader
    batch_size = 32
    src_dir = 'data/'
    dataset = get_loader(batch_size)

    # get model
    story_model = get_model()
    story_model = story_model.to(device) 
    # story_model.load_state_dict(torch.load('story_model/18.pt'))
    story_model = nn.DataParallel(story_model)


    storyline_model = get_model()
    storyline_model = storyline_model.to(device)
    # storyline_model.load_state_dict(torch.load('storyline_model/18.pt'))
    storyline_model = nn.DataParallel(storyline_model)
    # get optimizer
    LR = 4e-5
    ADAM_EPSILON = 1e-8
    WEIGHT_DECAY = 0.
    WARMUP_PROPORTION =0
    # WARMUP_PROPORTION = 0.1
    EPOCH = 10
    TRAIN_STEP = EPOCH * (len(dataset) + 1)
    WARMUP_STEP = TRAIN_STEP*WARMUP_PROPORTION
    # WARMUP_STEP = TRAIN_STEP*0.1
    story_optimizer, story_lr_scheduler = get_optimizer(model=storyline_model, lr=LR, train_steps=TRAIN_STEP, warmup_steps=WARMUP_STEP, weight_decay=WEIGHT_DECAY, adam_epsilon=ADAM_EPSILON)
    storyline_optimizer, storyline_lr_scheduler = get_optimizer(model=story_model, lr=LR, train_steps=TRAIN_STEP, warmup_steps=WARMUP_STEP, weight_decay=WEIGHT_DECAY, adam_epsilon=ADAM_EPSILON)


    # get tokenizer
    tokenizer = get_tokenizer()
    train_model(story_model, storyline_model, dataset, storyline_optimizer, storyline_lr_scheduler, story_optimizer, story_lr_scheduler, tokenizer, EPOCH)


if __name__ == "__main__":
    main()
