#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 19:20:52 2021

@author: yousuf_nzm
"""

import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np



class PacmanNetwork(nn.Module):
    
    def __init__(self, lr, input_dims, output_dim):
        super().__init__()
        self.lr = lr;
        
        c, h, w = input_dims
        self.network = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            );
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr);
        self.loss = nn.
    

