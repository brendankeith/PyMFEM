'''
   Class for defining an fem problem, solve and return element error estimators.
'''
import numpy as np
import torch
import torch.distributions as tdist
from prob_envs.FEM_env import FEM_env

class toy_problem(FEM_env):

    # constructor
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.minimum = 4.0
        self.initial_state = [10.0]*5 
        self.sd = 0.00001

    def reset(self):
        initial_state = torch.tensor(self.initial_state).float()
        return  initial_state

    def step(self, action):
        # dist = tdist.Normal(action,self.sd)
        # state = dist.sample()
        # cost = (state - self.minimum)**2
        # cost = torch.floor((state - self.minimum)**2)
        # cost = torch.floor(torch.max(torch.tensor([0.]),state - self.minimum)**2)
        
        theta = action.item()
        if theta < 0.0 or theta > 1.0:
            raise ValueError("Invalid action: {}".format(theta))

        cost_min = 0.25
        steps = 100000.
        dist = tdist.Normal(theta,0.01)
        state = dist.sample()
        cts_cost = (state - cost_min)**2
        cost = torch.floor(steps * cts_cost)/steps# + 0.01

        # if theta < 0.3:
        #     cost = torch.tensor(-7.255)
        # elif theta < 0.426:
        #     cost = torch.tensor(-7.236)
        # elif theta < 0.666:
        #     cost = torch.tensor(-7.01)
        # elif theta < 0.8:
        #     cost = torch.tensor(-6.89)
        # elif theta < 0.95:
        #     cost = torch.tensor(-6.817)
        # elif theta < 0.995:
        #     cost = torch.tensor(-6.442)
        # else:
        #     cost = torch.tensor(-6.044)

        # cost += 7.255

        state = self.initial_state
        done = True
        info = None
        return state, cost, done, info



