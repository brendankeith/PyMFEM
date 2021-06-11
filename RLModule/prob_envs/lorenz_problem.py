'''
    Lorenz problem
'''
import os
import gym
from gym import spaces, utils
import numpy as np
from numpy.core.numeric import NaN

def lorenz(x, y, z, s=10, r=28, b=2.667):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


class lorenz_problem(gym.Env):

    # constructor
    def __init__(self, config):
        super().__init__()
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        self.n = 0
        self.cost = NaN
        self.reset()

    def reset(self):
        self.initial_obs = np.array([0.0]) # Bandit problem. Observation doesn't matter 
        return self.initial_obs

    def step(self, action):
        self.n += 1
        dt = 0.01
        num_steps = 10000

        # Need one more for the initial values
        xs = np.empty(num_steps + 1)
        ys = np.empty(num_steps + 1)
        zs = np.empty(num_steps + 1)

        # Set initial values
        xs[0], ys[0], zs[0] = (0., 1., 1.05)

        # Step through "time", calculating the partial derivatives at the current point
        # and using them to estimate the next point
        for i in range(num_steps):
            x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], r=(28.+action))
            xs[i + 1] = xs[i] + (x_dot * dt)
            ys[i + 1] = ys[i] + (y_dot * dt)
            zs[i + 1] = zs[i] + (z_dot * dt) 

        obs = self.initial_obs
        self.cost = -(np.mean(zs[1000:]) - 27.5)**2
        done = True
        info = {}
        return obs, self.cost, done, info

    def render(self):
        print("n =", self.n)
        print("cost =", self.cost)


if __name__ == "__main__":

    env = lorenz_problem(None)
    env.render()
    env.reset()
    obs, reward, done, _ = env.step(0.5)
    env.render()
    obs, reward, done, _ = env.step(1.0)
    env.render()
    obs, reward, done, _ = env.step(1.5)
    env.render()
    obs, reward, done, _ = env.step(2.5)
    env.render()
    obs, reward, done, _ = env.step(3.0)
    env.render()
    obs, reward, done, _ = env.step(3.5)
    env.render()


    from typing import Dict, TYPE_CHECKING
    import ray
    import ray.rllib.agents.ppo as ppo
    import ray.rllib.agents.dqn as dqn
    from ray.rllib.env import BaseEnv
    from ray.rllib.policy import Policy
    from ray.rllib.policy.sample_batch import SampleBatch
    from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
    from ray.rllib.agents.callbacks import DefaultCallbacks

    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    total_episodes = 500
    batch_size = 16
    nbatches = int(total_episodes/batch_size)

    config = ppo.DEFAULT_CONFIG.copy()
    config['train_batch_size'] = batch_size
    config['sgd_minibatch_size'] = batch_size
    config['rollout_fragment_length'] = batch_size
    config['num_workers'] = 3
    config['num_gpus'] = 0
    config['lr'] = 1e-4
    # config['num_envs_per_worker'] = 1

    os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"
    agent = ppo.PPOTrainer(config, env=lorenz_problem)
    policy = agent.get_policy()
    model = policy.model

    cor = []
    ref = []

    checkpoint_period = 100

    episode = 0
    checkpoint_episode = 0
    for n in range(nbatches):
        print("training batch %d of %d batches" % (n+1,nbatches))
        agent.train()
        episode += config['train_batch_size']
        checkpoint_episode += config['train_batch_size']
        if (checkpoint_episode >= checkpoint_period):
            checkpoint_episode = 0
            checkpoint_path = agent.save()
            print(checkpoint_path)
            
    import matplotlib.pyplot as plt
    import pandas as pd
    root_path, _ = os.path.split(checkpoint_path)
    root_path, _ = os.path.split(root_path)
    csv_path = root_path + '/progress.csv'
    df = pd.read_csv(csv_path)
    cost = -df.episode_reward_mean.to_numpy()
    plt.plot(cost,'r',lw=1.3)
    # plt.semilogy(cost,'r',lw=1.3)
    plt.ylabel("cost")
    plt.xlabel("iteration")
    plt.show()    