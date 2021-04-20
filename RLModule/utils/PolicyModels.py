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

from scipy.stats import truncnorm

from StatisticsAndCost import StatisticsAndCost
# from problem_fem import fem_problem
from problem_fem import fem_problem

# Constants
GAMMA = 0.9
ORDER = 1

class TruncatedNormal():

    def __init__(self, mean, sd, lower_bound=0.0, upper_bound=1.0):
        self.mu = mean.detach().numpy()
        self.sigma = sd.detach().numpy()
        self.rv = truncnorm( (lower_bound - self.mu) / self.sigma, (upper_bound - self.mu) / self.sigma, loc=self.mu, scale=self.sigma )
        # 
        self.phi = lambda x: 1/np.sqrt(2*np.pi)*torch.exp(-0.5*x**2)
        self.Phi = lambda x: 0.5*(1+torch.erf(x/np.sqrt(2)))
        self.pdf = lambda theta: 1/sd * self.phi((theta - mean)/sd) /   \
        (self.Phi((upper_bound - mean)/sd) - self.Phi((lower_bound - mean)/sd))

    def sample(self):
        x = self.rv.rvs()
        return torch.from_numpy(x)
        # return torch.from_numpy(np.array([x]))

    def log_prob(self, x):
        return torch.log(self.pdf(x))


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, learning_rate=1e-4):
        super(PolicyNetwork, self).__init__()

        self.linear = nn.Linear(num_inputs, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = self.linear(state)
        x[1] = torch.exp(x[1])
        return x
    
    def get_action(self, state):
        dist_params = self.forward(state)
        b = tdist.Normal(dist_params[0], dist_params[1])
        # b = TruncatedNormal(dist_params[0], dist_params[1])
        action = b.sample()
        log_prob = b.log_prob(action)
        # return action, log_prob, b.mu, b.sigma
        return torch.sigmoid(action), log_prob, torch.sigmoid(dist_params[0]), dist_params[1].detach().numpy()

    def reset(self):
        self.linear.weight.data.fill_(0.01)
        self.linear.bias.data.fill_(0.5)

def update_policy(policy_network, costs, log_probs):
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
    
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()


if __name__ == "__main__":

    meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', 'l-shape.mesh'))
    # meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', 'star.mesh'))
    mesh = mfem.Mesh(meshfile, 1,1)
    mesh.UniformRefinement()

    # penalty = 1.0e1
    penalty = 0.0
    poisson = fem_problem(mesh,ORDER,penalty)
    # env = gym.make('CartPole-v0')
    policy_net = PolicyNetwork(4)
    # policy_net.reset()
    
    max_episode_num = 2000
    max_steps = 1
    numsteps = []
    all_costs = []
    actions = []
    means = []
    sds = []

    for episode in range(max_episode_num):
        # state = env.reset()
        state = poisson.reset()
        log_probs = []
        costs = []

        for steps in range(1,max_steps+1):
            # env.render()
            action, log_prob, mean, sd = policy_net.get_action(state)
            actions.append(action)
            # actions.append(action[0])
            means.append(mean.detach().numpy())
            sds.append(sd)
            # new_state, cost, done, _ = env.step(action)
            new_state, cost, done, _ = poisson.step(action)
            # if episode % 500 == 0 and episode < 5001:
            #     poisson.PlotSolution()

            log_probs.append(log_prob)
            costs.append(torch.tensor([cost]))
            # costs.append(torch.tensor([cost]) + 1e3*torch.maximum(torch.tensor([0]),mean-1) + 1e3*torch.maximum(torch.tensor([0]),-mean))
            # costs.append(torch.tensor([cost])+1e3*torch.maximum(torch.tensor([0]),mean-1))
            # costs.append(cost)

            if done or steps == max_steps:
                update_policy(policy_net, costs, log_probs)
                numsteps.append(steps)
                all_costs.append(np.sum(costs[0].detach().numpy()))
                if episode % 1 == 0:
                    # sys.stdout.write("episode: {}, episode/total cost: {}, average_cost: {}, length: {}\n".format(episode, np.round(np.sum(costs.detach().numpy()), decimals = 10),  np.round(np.mean(all_costs[-10:]), decimals = 10), steps))
                    sys.stdout.write("episode: {}, average_cost: {}, length: {}\n".format(episode, np.round(np.mean(all_costs[-10:]), decimals = 10), steps))
                break
            
            state = new_state
        
    fig, ax = plt.subplots(4, sharex=True)
    ax[0].plot(actions)
    ax[0].set_ylabel('Theta')
    ax[1].plot(means)
    ax[1].set_ylabel('mean')
    ax[2].semilogy(sds)
    ax[2].set_ylabel('st. dev.')
    ax[3].plot(all_costs)
    ax[3].set_ylabel('Cost')
    ax[3].set_xlabel('Episode')
    plt.show()




