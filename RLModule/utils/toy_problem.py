'''
   Class for defining an fem problem, solve and return element error estimators.
'''
import numpy as np
import torch
import torch.distributions as tdist

class toy_problem:

    # constructor
    def __init__(self, minimum=1.0, initial_state=10.0, sd=0.1):
        self.minimum = minimum
        # self.minimum = torch.tensor(minimum)
        self.intitial_state = initial_state
        self.sd = sd

    def reset(self):
        initial_state = self.intitial_state
        return  initial_state

    def step(self, action):
        dist = tdist.Normal(action,self.sd)
        state = dist.sample()
        cost = torch.floor((state - self.minimum)**2)
        done = True
        info = None
        return state, cost, done, info



