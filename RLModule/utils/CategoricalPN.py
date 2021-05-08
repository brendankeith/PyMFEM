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
from CategoricalPolicy import PolicyNetwork

# Constants
GAMMA = 0.9
ORDER = 1

def update_policy(policy_network, costs, log_probs, update=True, batch_size = 1):
    discounted_costs = []

    for t in range(len(costs)):
        Gt = 0 
        pw = 0
        for c in costs[t:]:
            Gt = Gt + GAMMA**pw * c
            pw = pw + 1
        discounted_costs.append(Gt)
        
    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_costs):
        policy_gradient.append(log_prob * Gt)
    
    policy_gradient = torch.stack(policy_gradient).sum()/batch_size 
    policy_gradient.backward()
    if update:
        policy_network.optimizer.step()


if __name__ == "__main__":

    meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', 'l-shape.mesh'))
    mesh = mfem.Mesh(meshfile, 1,1)
    mesh.UniformRefinement()
    mesh.UniformRefinement()
    # mesh.UniformRefinement()

    penalty = 0.0
    env = fem_problem(mesh,ORDER,penalty)
    max_episode_num = 5000
    batch_size = 10
    max_steps = 1
    numsteps = []
    all_costs = []
    actions = []
    all_probs = []
    num_actions = 5

    policy_net = PolicyNetwork(learning_rate=5e-1,weight_decay=0.0,num_actions=num_actions)
    policy_net.reset()

    for episode in range(1,max_episode_num):

        state = env.reset()
        log_probs = []
        costs = []
        update=False

        if episode % batch_size == 0:
            update=True

        if batch_size == 1 or episode % batch_size == 1:
            policy_net.optimizer.zero_grad()

        for steps in range(1,max_steps+1):
            action, log_prob, probs = policy_net.get_action(state)
            actions.append(action/(num_actions-1))

            prob = probs.detach().numpy()

            if episode == 1:
                all_probs = prob
            else:
                all_probs = np.vstack([all_probs, prob])

            new_state, cost, done, _ = env.step(action/(num_actions-1))
            log_probs.append(log_prob)
            costs.append(torch.tensor([cost]))

            if done or steps == max_steps:
                update_policy(policy_net, costs, log_probs, update=update, batch_size=batch_size)
                numsteps.append(steps)
                all_costs.append(np.sum(costs[0].detach().numpy()))
                if episode % 1 == 0:
                    sys.stdout.write("episode: {}, theta: {:.3f} cost: {:.3f}, length: {}\n".format(episode, np.mean(actions[-1:]), np.mean(all_costs[-1:]), steps))
                break
            
            state = new_state
        
# ------------------PLOTS----------------------------------------------
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].set_ylim(0, 1)
    # ax[i].set_ylabel("p({:.1f})".format(i/(num_actions-1)))
    ax[0].set_ylabel("Probabilities")
    for i in range(0,num_actions):
        # ax[i].set_xlabel('Episode')
        ax[0].plot(all_probs[:,i],label="Action: {:.2f}".format(i/(num_actions-1)))

    ax[0].legend()

    ax[1].plot(actions,'-o', linewidth=0.2, markersize=0.2)
    ax[1].set_ylabel('Theta')
    ax[2].plot(all_costs,'-o',linewidth=0.2, markersize=0.2)
    ax[2].set_ylabel('Cost')

    plt.show()

