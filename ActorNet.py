# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:44:22 2022

@author: hibad
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(10,100)
        torch.nn.init.xavier_uniform_(self.fc1 .weight)

        self.fc2 = nn.Linear(100,100)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(100,15)
        torch.nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self, x):
        x =F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x= self.fc3(x)
        distribution = Categorical(F.softmax(x, dim=-1))
        return distribution
    
net=ActorNet().to(device)
pickle.dump( net, open( "Actor_net.p", "wb" ) )
