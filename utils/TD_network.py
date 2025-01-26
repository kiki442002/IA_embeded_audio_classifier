# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:58:29 2024

@author: e2204699
"""

import torch as pt
import torch.nn as nn

#### Network ####
class CNNNetwork(nn.Module):
    def __init__(self, outsize=True, outlier=False):
        super().__init__()
        self.outlier = outlier
        self.outsize = outsize
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
        self.fc4A = nn.Linear(in_features=8, out_features=2)
        self.fc4B = nn.Linear(in_features=8, out_features=4)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        #Outlier sampling methode
        if self.outlier:
            return x
        
        #Normal inference
        if self.outsize:
            logits = self.fc4A(x)
        else:
            logits = self.fc4B(x)
        output = self.output(logits)
        return output
    
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

