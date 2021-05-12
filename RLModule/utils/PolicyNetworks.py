'''
    Policy Networks
'''

import numpy as np  
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as tdist
from scipy.stats import truncnorm


class TwoParamNormal(nn.Module):
    def __init__(self, **kwargs):
        super(TwoParamNormal, self).__init__()
        self.mu = nn.Parameter(torch.randn(()))
        self.log_sigma = nn.Parameter(torch.randn(()))
        self.softplus = nn.Softplus(beta=4.0)
        
        learning_rate = kwargs.get('learning_rate',1e-2)
        weight_decay = kwargs.get('weight_decay',0.)
        momentum = kwargs.get('momentum',0.)
        update_rule = kwargs.get('update_rule','SGD')
        # in python 3.10 we will be able to use a "switch" (https://stackoverflow.com/questions/60208/replacements-for-switch-statement-in-python)
        if update_rule == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif update_rule == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        self.baseline = 0.0

    def forward(self, state):
        return self.mu, self.softplus(self.log_sigma)
    
    def get_action(self, state):
        dist_params = self.forward(state)
        dist = tdist.Normal(dist_params[0], dist_params[1])
        action = dist.sample()
        # log_prob = dist.log_prob(action)
        log_prob = dist.log_prob(action) - torch.log(torch.sigmoid(action)*(1.0 - torch.sigmoid(action)))
        return torch.sigmoid(action), log_prob, (torch.sigmoid(dist_params[0]), dist_params[1])

    def reset(self):
        self.mu.data = torch.tensor(0.5)
        self.log_sigma.data = torch.tensor(-1.)
    
    def update_baseline(self, costs):
        self.baseline = np.minimum(self.baseline,costs[0].item()) + 0.01
        return self.baseline



class TwoParamTruncatedNormal(nn.Module):
    def __init__(self, **kwargs):
        super(TwoParamTruncatedNormal, self).__init__()
        self.mu = nn.Parameter(torch.randn(()))
        self.log_sigma = nn.Parameter(torch.randn(()))
        self.softplus = nn.Softplus(beta=4.0)
        learning_rate = kwargs.get('learning_rate',1e-2)
        self.optimizer = PSGD(self.parameters(), lr=learning_rate)
        self.baseline = 0.0

    def forward(self, state):
        return self.mu, self.softplus(self.log_sigma)
    
    def get_action(self, state):
        dist_params = self.forward(state)
        dist = TruncatedNormal(dist_params[0], dist_params[1])
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, [dist_params[0], dist_params[1]]

    def reset(self):
        self.mu.data = torch.tensor(0.5)
        self.log_sigma.data = torch.tensor(-1.)
    
    def update_baseline(self, costs):
        self.baseline = np.minimum(self.baseline,costs[0].item()) + 0.01
        return self.baseline






###########################
# CUSTOM UPDATE RULES
###########################

class PSGD:
    r"""Implements projected stochastic gradient descent
    """

    def __init__(self, params, lr=1e-3):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.params = list(params)
        self.lr = lr

    def step(self):
        """Performs a single optimization step.
        """
        cnt = 0
        with torch.no_grad():
            for p in self.params:
                p.data -= self.lr * p.grad
                if cnt == 0:
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

###########################
# CUSTOM DISTIBUTIONS
###########################

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

    def log_prob(self, x):
        return torch.log(self.pdf(x))
