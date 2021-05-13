import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

'''
    BASE CLASS for policy gradient algorithms
'''
class PolicyGradientMethod:

    def __init__(self, env, policy_net):
        self.env = env
        self.policy_net = policy_net

    def __call__(self, **kwargs):
        self.process_kwargs(**kwargs)
        self.optimize()
        self.plot_results()

    def process_kwargs(self, **kwargs):
        self.batch_size = kwargs.get('batch_size',1)
        self.max_steps = kwargs.get('max_steps',1)
        self.max_episode_num = kwargs.get('max_episode_num',100)
        self.GAMMA = kwargs.get('GAMMA',1.0)

    def optimize(self):
        self.numsteps = [1]
        self.actions = [0.0]
        self.all_costs = [0.0]

    def plot_results(self):

        if not self.all_costs:
            raise ValueError("Cannot plot empty results")

        fig, ax = plt.subplots(2, sharex=True)
        ax[0].plot(self.actions,'-o', linewidth=0.2, markersize=0.2)
        ax[0].set_ylabel('actions')
        ymin = np.min(self.actions)
        ymin = np.minimum(ymin,-0.1)
        ymax = np.max(self.actions)
        ymax = np.maximum(ymax,1.1)
        ax[0].set_ylim(ymin, ymax)
        ax[0].set_xlabel('Episodes')

        ax[1].plot(self.all_costs,'-o',linewidth=0.2, markersize=0.2)
        ax[1].set_ylabel('Cost')
        ax[1].set_xlabel('Episodes')

'''
    REINFORCE (with constant baseline)
'''
class REINFORCE(PolicyGradientMethod):

    def optimize(self):
        
        env = self.env
        policy_net = self.policy_net
        max_episode_num = self.max_episode_num
        batch_size = self.batch_size
        max_steps = self.max_steps
        numsteps = []
        all_costs = []
        actions = []
        dist_params = []
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
                action, log_prob, dist_param = policy_net.get_action(state)
                actions.append(action)
                new_state, cost, done, _ = env.step(action)

                log_probs.append(log_prob)
                costs.append(torch.tensor([cost]))
                # params = dist_param.detach().numpy()
                if episode == 1:
                    dist_params = dist_param
                else:
                    dist_params = np.vstack([dist_params,dist_param])
                # for i, param in enumerate(dist_param):
                    # params = 
                    # dist_params[i].append(param.item())

                if done or steps == max_steps:
                    self.update_policy(costs, log_probs, update=update)
                    numsteps.append(steps)
                    all_costs.append(np.sum(costs[0].detach().numpy()))
                    if episode % 1 == 0:
                        sys.stdout.write("episode: {}, average_cost: {}, length: {}\n".format(episode, np.round(np.mean(all_costs[-10:]), decimals = 10), steps))
                    break
                
                state = new_state
        
        self.numsteps = numsteps
        self.actions = actions
        self.all_costs = all_costs
        self.dist_params = dist_params
    
    def update_policy(self, costs, log_probs, update=True):

        GAMMA = self.GAMMA
        policy_net = self.policy_net
        batch_size = self.batch_size
        baseline = policy_net.update_baseline(costs)

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
            policy_gradient.append(log_prob * (Gt - baseline))
        
        policy_gradient = torch.stack(policy_gradient).sum()/batch_size

        # print('baseline = ', baseline)
        # policy_gradient = (costs[0] - baseline) * log_probs[0] / batch_size
        policy_gradient.backward()
        if update:
            policy_net.optimizer.step()


'''
    Actor-Critic (without eligibility traces)
'''
class ActorCritic(PolicyGradientMethod):

    def __init__(self, env, policy_net, value_net):
        super(ActorCritic, self).__init__(env, policy_net)
        self.value_net = value_net

    def optimize(self):
        env = self.env
        policy_net = self.policy_net
        value_net = self.value_net
        max_episode_num = self.max_episode_num
        batch_size = self.batch_size
        max_steps = self.max_steps
        GAMMA = self.GAMMA
        numsteps = []
        all_costs = []
        actions = []
        dist_params = []
        for episode in range(1,max_episode_num):
            state = env.reset()
            log_probs = []
            costs = []
            update=False
    
            if episode % batch_size == 0:
                update=True

            if batch_size == 1 or episode % batch_size == 1:
                policy_net.optimizer.zero_grad()
                value_net.optimizer.zero_grad()
            
            for steps in range(1,max_steps+1):
                action, log_prob, dist_param = policy_net.get_action(state)
                actions.append(action)
                new_state, cost, done, _ = env.step(action)

                if steps == max_steps:
                    # delta = cost + 7.255
                    delta = cost - value_net(state)
                else:
                    delta = cost + GAMMA*value_net(new_state) - value_net(state)

                v = value_net(state)
                v.backward()
                with torch.no_grad():
                    for p in value_net.parameters():
                        p.data += value_net.learning_rate * delta * p.grad

                log_prob.backward()
                with torch.no_grad():
                    for p in policy_net.parameters():
                        p.data -= policy_net.learning_rate * GAMMA**(steps-1) * delta * p.grad

                # log_probs.append(log_prob)
                print('\n')
                print('cost = ',cost)
                print('action = ',action)
                print('delta = ',delta)
                print('value function estimat = ',value_net(state).item())
                # print('value_net params')
                # for p in value_net.parameters():
                #     print(p)
                # print('policy_net params')
                # for p in policy_net.parameters():
                #     print(p)
                costs.append(torch.tensor([cost]))
                # if episode == 1:
                #     dist_params = dist_param
                # else:
                #     dist_params = np.vstack([dist_params,dist_param])

                # if done or steps == max_steps:
                #     self.update_policy(costs, log_probs, update=update)
                #     numsteps.append(steps)
                all_costs.append(np.sum(costs[0].detach().numpy()))
                #     if episode % 1 == 0:
                #         sys.stdout.write("episode: {}, average_cost: {}, length: {}\n".format(episode, np.round(np.mean(all_costs[-10:]), decimals = 10), steps))
                #     break
                
                state = new_state
        
        # self.numsteps = numsteps
        self.actions = actions
        self.all_costs = all_costs