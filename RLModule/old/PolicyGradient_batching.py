'''
    Policy gradient test problem

    Some of the code is inspired by https://github.com/Finspire13/pytorch-policy-gradient-example/blob/master/pg.py
'''

import sys
# sys.path.append("~/Work/PyMFEM/RLModule/examples")
# sys.path.append("~/Work/PyMFEM/")
sys.path.append("..")
import os
from os.path import expanduser, join

import torch
import mfem.ser as mfem

import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as tdist
from torch.autograd import Variable
import matplotlib.pyplot as plt
from itertools import count

from scipy.stats import truncnorm

from StatisticsAndCost import StatisticsAndCost
from problem_fem import fem_problem
from PolicyModels import PolicyNetwork

def main():

    # Parameters
    num_episode = 5000
    max_steps = 1
    batch_size = 1
    learning_rate = 1e-2
    gamma = 0.99

    meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', 'l-shape.mesh'))
    # meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', 'star.mesh'))
    mesh = mfem.Mesh(meshfile, 1,1)
    mesh.UniformRefinement()

    penalty = 0.0
    poisson = fem_problem(mesh,ORDER,penalty)
    policy_net = PolicyNetwork(4)
    optimizer = policy_net.optimizer

    # Batch History
    steps = 0
    state_pool = []
    action_pool = []
    cost_pool = []

    for e in range(1,num_episode):

        state = poisson.reset()
        state = torch.from_numpy(state).float()
        state = Variable(state)

        for t in count():

            # probs = policy_net(state)
            # m = Bernoulli(probs)
            # action = m.sample()
            action, _, _, _ = policy_net.get_action(state)

            action = action.data.numpy().astype(float)[0]
            next_state, cost, done, _ = poisson.step(action)

            # To mark boundarys between episodes
            # if done:
            #     cost = 0 # ?

            state_pool.append(state)
            action_pool.append(float(action))
            cost_pool.append(cost)

            state = next_state
            state = torch.from_numpy(state).float()
            state = Variable(state)

            steps += 1

            if done or steps % max_steps == 0:
                break

        # Update policy
        if e % batch_size == 0:

            # Discount cost
            running_add = 0
            for i in reversed(range(steps)):
                if cost_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * gamma + cost_pool[i]
                    cost_pool[i] = running_add

            # Normalize cost
            # cost_mean = np.mean(cost_pool)
            # cost_std = np.std(cost_pool)
            # for i in range(steps):
            #     cost_pool[i] = (cost_pool[i] - cost_mean) / cost_std

            # Gradient Desent
            optimizer.zero_grad()

            for i in range(steps):
                state = state_pool[i]
                action = Variable(torch.FloatTensor([action_pool[i]]))
                cost = cost_pool[i]

                probs = policy_net(state)
                m = Bernoulli(probs)
                loss = m.log_prob(action) * cost
                loss.backward()

            optimizer.step()

            state_pool = []
            action_pool = []
            cost_pool = []
            steps = 0


if __name__ == '__main__':
    main()
