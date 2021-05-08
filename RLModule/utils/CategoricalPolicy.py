
import sys
sys.path.append("..")
import os
from os.path import expanduser, join

import numpy as np  
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as tdist
from torch.autograd import Variable

from scipy.stats import truncnorm

class PolicyNetwork(nn.Module):
    def __init__(self, learning_rate=1e-1, weight_decay=0.0, num_actions = 2):
        super(PolicyNetwork, self).__init__()
        self.num_actions = num_actions
        self.logits = nn.Parameter(torch.randn((num_actions)))
      #   self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def forward(self, state):
        return torch.softmax(self.logits,dim=0)
    
    def get_action(self, state):
        probs = self.forward(state)
        b = tdist.Categorical(probs=probs)
        action = b.sample()
        log_prob = b.log_prob(action)
        return action, log_prob, probs

    def reset(self):
        self.logits.data.fill_(1/self.num_actions) 





