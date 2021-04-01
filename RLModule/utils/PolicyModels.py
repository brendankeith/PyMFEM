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

from StatisticsAndCost import StatisticsAndCost
# from problem_fem import fem_problem
from problem_fem import fem_problem

# Constants
GAMMA = 0.9

# class PolicyNetwork(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
#         super(PolicyNetwork, self).__init__()

#         self.num_actions = num_actions
#         self.linear1 = nn.Linear(num_inputs, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, num_actions)
#         self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

#     def forward(self, state):
#         x = F.relu(self.linear1(state))
#         x = F.softmax(self.linear2(x), dim=1)
#         return x 
    
#     def get_action(self, state):
#         state = torch.from_numpy(state).float().unsqueeze(0)
#         probs = self.forward(Variable(state))
#         highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
#         log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
#         return highest_prob_action, log_prob

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_dist_params=2, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.linear = nn.Linear(num_inputs, num_dist_params)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = self.linear(state)
        return F.softmax(x)
    
    def get_action(self, state):
        # state = torch.from_numpy(state).float().unsqueeze(0)
        # probs = self.forward(Variable(state))
        # highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        # log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        dist_params = self.forward(state)
        b = tdist.Beta(dist_params[0],dist_params[1])
        action = b.sample()
        log_prob = b.log_prob(action)
        return action, log_prob

def update_policy(policy_network, costs, log_probs):
    discounted_costs = []

    for t in range(len(costs)):
        Gt = 0 
        pw = 0
        for c in costs[t:]:
            Gt = Gt + GAMMA**pw * c
            pw = pw + 1
        discounted_costs.append(Gt)
        
    discounted_costs = torch.tensor(discounted_costs)
    # discounted_costs = (discounted_costs - discounted_costs.mean()) / (discounted_costs.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_costs):
        policy_gradient.append(log_prob * Gt)
    
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()


if __name__ == "__main__":

    # b = tdist.Beta(torch.tensor([0.5]), torch.tensor([0.5]))
    # action = b.sample()
    # # next_state, reward = env.step(action)
    # # loss = -m.log_prob(action) * reward
    # loss = b.log_prob(action)
    # loss.backward()

    order = 1
    meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', 'star.mesh'))
    mesh = mfem.Mesh(meshfile, 1,1)
    mesh.UniformRefinement()

    poisson = fem_problem(mesh,order)
    # env = gym.make('CartPole-v0')
    policy_net = PolicyNetwork(4)
    
    max_episode_num = 500
    max_steps = 3
    numsteps = []
    all_costs = []

    for episode in range(max_episode_num):
        # state = env.reset()
        state = poisson.reset()
        log_probs = []
        costs = []

        for steps in range(max_steps):
    #         env.render()
            action, log_prob = policy_net.get_action(state)
            # new_state, cost, done, _ = env.step(action)
            new_state, cost, done, _ = poisson.step(action)
            log_probs.append(log_prob)
            costs.append(cost)

            if done or steps == max_steps:
                update_policy(policy_net, costs, log_probs)
                numsteps.append(steps)
                all_costs.append(np.sum(costs))
                if episode % 1 == 0:
                    sys.stdout.write("episode: {}, total cost: {}, average_cost: {}, length: {}\n".format(episode, np.round(np.sum(costs), decimals = 3),  np.round(np.mean(all_costs[-10:]), decimals = 3), steps))
                break
            
            state = new_state
        
    # plt.plot(numsteps)
    # plt.plot(avg_numsteps)
    # plt.xlabel('Episode')
    # plt.show()