# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:17:17 2022

@author: hibad
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(10,100)
        torch.nn.init.xavier_uniform_(self.fc1 .weight)

        self.fc2 = nn.Linear(100,100)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(100,1)
        torch.nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self, x):
        x =F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x= self.fc3(x)
        x=torch.clip(x, min=-1000, max=10000)
        return x
    
net=CriticNet().to(device)
pickle.dump( net, open( "Q_net.p", "wb" ) )
