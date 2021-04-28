'''
    Policy models

    Some of the code is inspired by https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
'''

import sys
# sys.path.append("~/Work/PyMFEM/RLModule/examples")
# sys.path.append("~/Work/PyMFEM/")
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
    def __init__(self, num_inputs, learning_rate=1e-3, weight_decay=0.0):
        super(PolicyNetwork, self).__init__()

        # self.linear = nn.Linear(num_inputs, 2)
        self.mu = nn.Parameter(torch.randn(()))
        self.log_sigma = nn.Parameter(torch.randn(()))

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)


    def forward(self, state):
        # x = torch.tensor([self.mu,self.sigma])
        # x = self.linear(state)
        # x[1] = torch.exp(x[1])
        # x[1] = 0.01
        # return x
        # return torch.sigmoid(self.mu), torch.exp(self.log_sigma)
        return self.mu, torch.exp(self.log_sigma)
        # return self.mu, torch.tensor(0.01)
    
    def get_action(self, state):
        dist_params = self.forward(state)
        b = tdist.Normal(dist_params[0], dist_params[1])
        # b = TruncatedNormal(dist_params[0], dist_params[1])
        action = b.sample()
        log_prob = b.log_prob(action)
        # log_prob = b.log_prob(action) - torch.log(torch.sigmoid(action)*(1.0 - torch.sigmoid(action)))
        # regularization = 1e0*torch.sigmoid(dist_params[0])**2
        # regularization = 1e-4* ( dist_params[0]**2 + dist_params[1]**2 )
        regularization = torch.tensor(0.0)
        return torch.sigmoid(action), log_prob, \
               dist_params[0], dist_params[1], regularization

        # return action, log_prob, dist_params[0], dist_params[1], regularization

    def reset(self):
        # self.linear.weight.data.fill_(0.01)
        # self.linear.bias.data.fill_(0.5)
        self.mu.data = torch.tensor(0.5)
        self.log_sigma.data = torch.log(torch.tensor(0.1))






########### UNUSED ###########

class TruncatedNormal():

    def __init__(self, mean, sd, lower_bound=0.0, upper_bound=1.0):
        self.mu = mean.detach().numpy()
        self.sigma = sd.detach().numpy()
        self.rv = truncnorm( (lower_bound - self.mu) / self.sigma, (upper_bound - self.mu) / self.sigma, loc=self.mu, scale=self.sigma )
        # 
        self.phi = lambda x: 1/np.sqrt(2*np.pi)*torch.exp(-0.5*x**2)
        self.Phi = lambda x: 0.5*(1+torch.erf(x/np.sqrt(2)))
        self.pdf = lambda theta: 1/sd * self.phi((theta - mean)/sd) /   \
        (self.Phi((upper_bound - mean)/sd) - self.Phi((lower_bound - mean)/sd))

    def sample(self):
        x = self.rv.rvs()
        return torch.from_numpy(x)
        # return torch.from_numpy(np.array([x]))

    def log_prob(self, x):
        return torch.log(self.pdf(x))
