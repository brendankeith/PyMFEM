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
        return action, log_prob, dist_params

    def reset(self):
        self.mu.data = torch.tensor(0.5)
        self.log_sigma.data = torch.tensor(-1.)
    
    def update_baseline(self, costs):
        self.baseline = np.minimum(self.baseline,costs[0].item()) + 0.01
        return self.baseline

class LinearNormal(nn.Module):
    def __init__(self, **kwargs):
        super(LinearNormal, self).__init__()

        self.linear = nn.Linear(5, 2) # state parameters // for now only 4
        
        self.learning_rate = kwargs.get('learning_rate',1e-2)
        learning_rate = kwargs.get('learning_rate',1e-2)
        weight_decay = kwargs.get('weight_decay',0.)
        momentum = kwargs.get('momentum',0.)
        update_rule = kwargs.get('update_rule','SGD')
        # in python 3.10 we will be able to use a "switch" (https://stackoverflow.com/questions/60208/replacements-for-switch-statement-in-python)
        if update_rule == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), \
                 lr=learning_rate, weight_decay=weight_decay)
        elif update_rule == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), \
                 lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        else:
            self.optimizer = optim.SGD(self.parameters(), \
                 lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        self.baseline = 0.0

    def forward(self, state, sd_tol=1e-2):
        x = self.linear(state)
        # with torch.no_grad():
        #     ddx_sigmoid = torch.sigmoid(x[0])*(1.0 - torch.sigmoid(x[0]))
        # return x[0], torch.exp(x[1]) + sd_tol/(torch.abs(ddx_sigmoid) + 1e-1)
        return x[0], torch.exp(x[1]) + sd_tol
    
    def get_action(self, state):
        params = self.forward(state)
        b = tdist.Normal(params[0],params[1])
        action = b.sample()
        log_prob = b.log_prob(action)
        return torch.sigmoid(action), log_prob, params

    def reset(self):
        return None
    
    def update_baseline(self, costs):
        self.baseline = np.minimum(self.baseline,costs[0].item()) + 0.01
        return self.baseline

class LinearTruncatedNormal(nn.Module):
    def __init__(self, **kwargs):
        super(LinearTruncatedNormal, self).__init__()
        self.linear = nn.Linear(5, 2) # state parameters // for now only 4
        self.learning_rate = kwargs.get('learning_rate',1e-2)
        learning_rate = kwargs.get('learning_rate',1e-2)
        weight_decay = kwargs.get('weight_decay',0.)
        momentum = kwargs.get('momentum',0.)
        update_rule = kwargs.get('update_rule','SGD')
        if update_rule == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), \
                 lr=learning_rate, weight_decay=weight_decay)
        elif update_rule == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), \
                 lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        else:
            self.optimizer = optim.SGD(self.parameters(), \
                 lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        self.baseline = 0.0

    def forward(self, state, sd_tol=1e-3):
        x = self.linear(state)
        return torch.sigmoid(x[0]), torch.exp(x[1]) + sd_tol
    
    def get_action(self, state):
        dist_params = self.forward(state)
        dist = TruncatedNormal(dist_params[0], dist_params[1])
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist_params

    def reset(self):
        # self.linear.weight.data.fill_(0.0)
        # self.linear.bias.data.fill_(0.0)
        return None
    
    def update_baseline(self, costs):
        self.baseline = np.minimum(self.baseline,costs[0].item()) + 0.01
        return self.baseline


class Categorical(nn.Module):
    def __init__(self, **kwargs):
        super(Categorical, self).__init__()

        self.linear = nn.Linear(4, 2) # state parameters // for now only 4
        
        self.num_actions = kwargs.get('num_actions',5)
        learning_rate = kwargs.get('learning_rate',1e-2)
        weight_decay = kwargs.get('weight_decay',0.)
        momentum = kwargs.get('momentum',0.)
        update_rule = kwargs.get('update_rule','SGD')
        self.logits = nn.Parameter(torch.randn((self.num_actions)))

        if update_rule == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), \
                 lr=learning_rate, weight_decay=weight_decay)
        elif update_rule == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), \
                 lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        else:
            self.optimizer = optim.SGD(self.parameters(), \
                 lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        self.baseline = 0.0

    def forward(self, state):
        return torch.softmax(self.logits,dim=0)

    
    def get_action(self, state):
        probs = self.forward(state)
        b = tdist.Categorical(probs=probs)
        action = b.sample()
        log_prob = b.log_prob(action)
        return action/(self.num_actions-1), log_prob, probs.detach().numpy()

    def reset(self):
        self.logits.data.fill_(1/self.num_actions) 
    
    def update_baseline(self, costs):
        return self.baseline



###########################
##### CRITICS
###########################

class LinearCritic(nn.Module):
    def __init__(self, **kwargs):
        super(LinearCritic, self).__init__()
        self.learning_rate = kwargs.get('learning_rate_critic',1e-2)
        weight_decay = kwargs.get('weight_decay',0.)
        momentum = kwargs.get('momentum',0.)
        self.linear = nn.Linear(5, 1) # state parameters // for now only 4
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=weight_decay, momentum=momentum)

    def forward(self, state):
        x = self.linear(state)
        return x

    def reset(self):
        self.linear.weight.data.fill_(0.0)
        # self.linear.bias.data.fill_(0.0)
        self.linear.bias.data.fill_(-0.0)




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
