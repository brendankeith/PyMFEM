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
    def __init__(self, num_inputs, learning_rate=1e-1, weight_decay=0.0, momentum=0.0):
        super(PolicyNetwork, self).__init__()

        # self.linear = nn.Linear(num_inputs, 2)
        self.mu = nn.Parameter(torch.randn(()))
        self.log_sigma = nn.Parameter(torch.randn(()))
        self.softplus = nn.Softplus(beta=4.0)

        self.optimizer = PSGD(self.parameters(), lr=learning_rate)
        # self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        # self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)


    def forward(self, state):
        # x = torch.tensor([self.mu,self.sigma])
        # x = self.linear(state)
        # x[1] = torch.exp(x[1])
        # x[1] = 0.01
        # return x
        # mu = torch.relu(1.0 - torch.relu(1.0-self.mu))
        # mu = torch.relu(0.99 - torch.relu(0.99-self.mu))
        # return mu, 0.1*torch.sigmoid(self.log_sigma)
        # return mu, torch.exp(self.log_sigma)
        # return torch.relu(self.mu), torch.exp(self.log_sigma)
        print('mu = ', self.mu.data)
        print('sigma = ', self.softplus(self.log_sigma.data))
        return self.mu, self.softplus(self.log_sigma)
        # return self.mu, torch.exp(self.log_sigma)
        # return self.mu, torch.tensor(0.01)
    
    def get_action(self, state):
        dist_params = self.forward(state)
        # ref_dist = tdist.Normal(dist_params[0], dist_params[1])
        ref_dist = TruncatedNormal(dist_params[0], dist_params[1])
        # Set de-refinement distribution here:
        deref_dist = TruncatedNormal(dist_params[0], dist_params[1])
        ref_axn = ref_dist.sample()
        deref_axn = deref_dist.sample()
        action = [ref_axn, deref_axn]
        log_prob = ref_dist.log_prob(ref_axn)  
        # log_prob = ref_dist.log_prob(action) - torch.log(torch.sigmoid(action)*(1.0 - torch.sigmoid(action)))
        # regularization = 1e0*torch.sigmoid(dist_params[0])**2
        # regularization = 1e-2*dist_params[1]**2
        # regularization = 1e3 * dist_params[0]**2 + 1e-1 * dist_params[1]**2
        # regularization = 1e-2 * ( dist_params[0]**2 + dist_params[1]**2 )
        regularization = torch.tensor(0.0)
        # return torch.sigmoid(action), log_prob, \
            #    dist_params[0], dist_params[1], regularization

        return action, log_prob, dist_params[0], dist_params[1], regularization

    def reset(self):
        # self.linear.weight.data.fill_(0.01)
        # self.linear.bias.data.fill_(0.5)
        self.mu.data = torch.tensor(0.5)
        self.log_sigma.data = torch.tensor(-1.)
        # self.log_sigma.data = torch.log(torch.tensor(0.01))
        # self.mu.data = torch.tensor(-1.4675)
        # self.log_sigma.data = torch.tensor(-1.4409)


class PSGD:
    r"""Implements projected stochastic gradient descent
    """

    def __init__(self, params, lr=1e-3):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.params = list(params)
        self.lr = lr

    # @torch.no_grad()
    def step(self):
        """Performs a single optimization step.
        """
        cnt = 0
        with torch.no_grad():
            for p in self.params:
                p.data -= self.lr * p.grad
                if cnt == 0:
                    # p = torch.max(torch.tensor(0.001, requires_grad=True),p)
                    # p = torch.min(torch.tensor(0.999, requires_grad=True),p)
                    p.data = torch.relu(1.0 - torch.relu(1.0-p.data))
                    cnt+=1
        
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()



########### UNUSED ###########

class TruncatedNormal():

    def __init__(self, mean, sd, lower_bound=0.0, upper_bound=1.0):
        
        mu = mean.item()
        sigma = sd.item()
        if mu < 0.0 or mu > 1.0:
            raise ValueError("Invalid mean: {}".format(mu))

        self.mu = mu
        self.sigma = sigma
        self.rv = truncnorm( (lower_bound - self.mu) / self.sigma, (upper_bound - self.mu) / self.sigma, loc=self.mu, scale=self.sigma )
        # 
        self.phi = lambda x: 1/np.sqrt(2*np.pi)*torch.exp(-0.5*x**2)
        self.Phi = lambda x: 0.5*(1+torch.erf(x/np.sqrt(2)))
        self.pdf = lambda theta: 1/sd * self.phi((theta - mean)/sd) /   \
        (self.Phi((upper_bound - mean)/sd) - self.Phi((lower_bound - mean)/sd))

    def sample(self):
        x = self.rv.rvs()
        return torch.tensor(x)
        # return torch.from_numpy(np.array([x]))

    def log_prob(self, x):
        return torch.log(self.pdf(x))
