'''
   Class for defining an fem problem, solve and return element error estimators.
'''
import numpy as np
import torch
import torch.distributions as tdist
from FEM_env import FEM_env

class toy_problem(FEM_env):

    # constructor
    def __init__(self, mesh, order):
        super().__init__(mesh, order)
        self.minimum = 4.0
        self.intitial_state = 10.0
        self.sd = 0.00001

    def reset(self):
        initial_state = self.intitial_state
        return  initial_state

    def step(self, action):
        dist = tdist.Normal(action,self.sd)
        state = dist.sample()
        # cost = (state - self.minimum)**2
        # cost = torch.floor((state - self.minimum)**2)
        cost = torch.floor(torch.max(torch.tensor([0.]),state - self.minimum)**2)
        done = True
        info = None
        return state, cost, done, info



