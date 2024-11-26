# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:58:29 2024

@author: e2204699
"""

import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime


if pt.cuda.is_available():
    device='cuda'
else:
    device='cpu'


#### Network ####
class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=896, out_features=40),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=40, out_features=32),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=32, out_features=8),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(in_features=8, out_features=4)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        logits = self.fc4(x)
        output = self.output(logits)
        return output
    
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

#### Train ####
def train_single_epoch(model,dataloader,loss_fn,optimizer,device):
    for waveform,label in tqdm.tqdm(dataloader):
        waveform=waveform.to(device)
        # label=pt.from_numpy(numpy.array(label))
        label=label.to(device)
        # calculate loss and preds
        logits=model(waveform)
        loss=loss_fn(logits.float(),label.float().view(-1,1))
        # backpropogate the loss and update the gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss:{loss.item()}")
    
def train(model,dataloader,loss_fn,optimizer,device,epochs):
    for i in tqdm.tqdm(range(epochs)):
        print(f"epoch:{i+1}")
        train_single_epoch(model,dataloader,loss_fn,optimizer,device)
        print('-------------------------------------------')
    print('Finished Training')


if __name__ == "__main__":
    BATCH_SIZE=128
    EPOCHS=1
    loss_fn=pt.nn.MSELoss()
    optimizer=pt.optim.SGD(model.parameters(),lr=0.1,momentum=0.9)

    train(model,train_dataloader,loss_fn,optimizer,device,EPOCHS)