'''
    Policy gradient test problem

    Some of the code is inspired by https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
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

from scipy.stats import truncnorm

from StatisticsAndCost import StatisticsAndCost
from problem_fem import fem_problem
from toy_problem import toy_problem
from PolicyModels import PolicyNetwork

# Constants
GAMMA = 0.9
ORDER = 1

def update_policy(policy_network, costs, log_probs, update=True):
    discounted_costs = []

    for t in range(len(costs)):
        Gt = 0 
        pw = 0
        for c in costs[t:]:
            Gt = Gt + GAMMA**pw * c
            pw = pw + 1
        discounted_costs.append(Gt)
        
    # discounted_costs = torch.tensor(discounted_costs)
    # discounted_costs = (discounted_costs - discounted_costs.mean()) / (discounted_costs.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_costs):
        policy_gradient.append(log_prob * Gt)
    
    # policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    if update:
        policy_network.optimizer.step()


if __name__ == "__main__":

    meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', 'l-shape.mesh'))
    # meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', 'star.mesh'))
    mesh = mfem.Mesh(meshfile, 1,1)
    mesh.UniformRefinement()

    # penalty = 1.0e1
    penalty = 0.0
    # env = fem_problem(mesh,ORDER,penalty)
    env = toy_problem()
    # env = gym.make('CartPole-v0')
    policy_net = PolicyNetwork(4)
    # policy_net.reset()
    
    max_episode_num = 1000
    batch_size = 10
    max_steps = 1
    numsteps = []
    all_costs = []
    actions = []
    means = []
    sds = []

    for episode in range(1,max_episode_num):
        # state = env.reset()
        update=False
        state = env.reset()
        log_probs = []
        costs = []

        if episode % batch_size == 0:
            update=True

        if batch_size == 1 or episode % batch_size == 1:
            policy_net.optimizer.zero_grad()

        for steps in range(1,max_steps+1):
            # env.render()
            action, log_prob, mean, sd = policy_net.get_action(state)
            actions.append(action)
            means.append(mean)
            sds.append(sd)
            # new_state, cost, done, _ = env.step(action)
            new_state, cost, done, _ = env.step(action)
            # if episode % 500 == 0 and episode < 5001:
            #     env.PlotSolution()

            log_probs.append(log_prob)
            costs.append(torch.tensor([cost]))

            if done or steps == max_steps:
                update_policy(policy_net, costs, log_probs, update=update)
                numsteps.append(steps)
                all_costs.append(np.sum(costs[0].detach().numpy()))
                if episode % 1 == 0:
                    # sys.stdout.write("episode: {}, episode/total cost: {}, average_cost: {}, length: {}\n".format(episode, np.round(np.sum(costs.detach().numpy()), decimals = 10),  np.round(np.mean(all_costs[-10:]), decimals = 10), steps))
                    sys.stdout.write("episode: {}, average_cost: {}, length: {}\n".format(episode, np.round(np.mean(all_costs[-10:]), decimals = 10), steps))
                    # sys.stdout.write("action: {}, mean of policy: {}, SD of policy: {}\n\n".format(action, mean, sd))
                break
            
            state = new_state
        
    fig, ax = plt.subplots(4, sharex=True)
    ax[0].plot(actions)
    ax[0].set_ylabel('Theta')
    ax[1].plot(means)
    ax[1].set_ylabel('sigmoid(mu)')
    ax[2].semilogy(sds)
    ax[2].set_ylabel('sigma')
    ax[3].plot(all_costs)
    ax[3].set_ylabel('Cost')
    ax[3].set_xlabel('Episode')
    plt.show()

