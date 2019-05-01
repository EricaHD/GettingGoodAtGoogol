import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class BasicDQN(nn.Module):

    def __init__(self, inp_size, hid_size, out_size):
        super(BasicDQN, self).__init__()
        
        self.hid = nn.Linear(inp_size, hid_size)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(hid_size, out_size)
        
    def forward(self, x):
        hid = F.relu(self.hid(x))
        dropped = self.drop(hid)
        out = self.out(dropped)
        return F.softmax(out, dim=-1)
