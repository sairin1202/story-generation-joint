import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import json
import random
from copy import deepcopy
from tqdm import tqdm
from glob import glob
import pickle
import pandas as pd


class Dataset():
    def __init__(self):
        with open('data/storyline_train.pkl', 'rb') as f:
            self.data = pickle.load(f)
        with open('data/story_train.pkl', 'rb') as f:
            self.data2 = pickle.load(f)

    def __getitem__(self, index):
        story_inputs = self.data2['input'][index]
        story_targets = self.data2['target'][index]
        storyline_inputs = self.data['input'][index]
        storyline_targets = self.data['target'][index]
        return storyline_inputs, storyline_targets, story_inputs, story_targets

    def __len__(self):
        return len(self.data['input'])




def get_loader(batch_size, shuffle=True):
    dataset = Dataset()
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=10,
                                              drop_last=True)
    return data_loader



class TestDataset():
    def __init__(self):
        with open('data/storyline_dev.pkl', 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        inputs = self.data['input'][index]
        targets = self.data['target'][index]
        return inputs, targets

    def __len__(self):
        return len(self.data['input'])




def get_test_loader(batch_size, shuffle=True):
    dataset = TestDataset()
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=10,
                                              drop_last=True)
    return data_loader