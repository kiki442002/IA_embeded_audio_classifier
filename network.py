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
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=2),
            nn.ReLU(), # Activation function
            nn.MaxPool2d(kernel_size=2) # Keep max value in kernel
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Flatten() # Converts from an n-dimension matrices into a vector
        self.linear1=nn.Linear(in_features=32,out_features=32)
        self.linear2=nn.Linear(in_features=32,out_features=2)
        self.output=nn.Softmax(dim=1) # Converts logits into probabilities
    
    def forward(self,input_data):
        x=self.conv1(input_data)
        x=self.conv2(x)
        x=self.flatten(x)
        x=self.linear1(x)
        logits=self.linear2(x)
        output=self.output(logits)
        
        return output


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