"""
    
    Script to train an optimal marking policy starting from a fixed initial mesh

    * In order for this to work with rllib, you need to have RLModule in your PYTHONPATH

"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from prob_envs.DerefVariableInitMesh import DerefVariableInitMesh
# from prob_envs.VariableInitialMesh import VariableInitialMesh
# from prob_envs.FixedInitialMesh import FixedInitialMesh

prob_config = {
    'mesh_name'         : 'star.mesh',
    # 'mesh_name'         : 'l-shape.mesh',
    'num_unif_ref'      : 2,
    'num_random_ref'    : 0,
    'order'             : 2,
    'optimization_type' : 'A',
    # 'optimization_type' : 'B',
}

# total_episodes = 10000
# batch_size = 32
# checkpoint_period = 100

total_episodes = 5000
batch_size = 64
checkpoint_period = 200

# # short run - for debugging
# total_episodes = 10
# batch_size = 5
# checkpoint_period = 5

nbatches = int(total_episodes/batch_size)

config = ppo.DEFAULT_CONFIG.copy()
config['train_batch_size'] = batch_size
config['sgd_minibatch_size'] = batch_size
config['rollout_fragment_length'] = batch_size
config['num_workers'] = 6
config['num_gpus'] = 0
config['lr'] = 1e-4


ray.shutdown()
ray.init(ignore_reinit_error=True)

register_env("my_env", lambda config : DerefVariableInitMesh(**prob_config))
agent = ppo.PPOTrainer(env="my_env", config=config)
policy = agent.get_policy()
model = policy.model

cor = []
ref = []

episode = 0
checkpoint_episode = 0
for n in range(nbatches):
    print("training batch %d of %d batches" % (n+1,nbatches))
    result = agent.train()
    episode += config['train_batch_size']
    checkpoint_episode += config['train_batch_size']
    episode_score = -result["episode_reward_mean"]
    print ("Episode cost", episode_score)
    if (checkpoint_episode >= checkpoint_period and n > 0.9*(nbatches-1)):
        checkpoint_episode = 0
        checkpoint_path = agent.save()
        print(checkpoint_path)

root_path, _ = os.path.split(checkpoint_path)
root_path, _ = os.path.split(root_path)
csv_path = root_path + '/progress.csv'
df = pd.read_csv(csv_path)
cost = -df.episode_reward_mean.to_numpy()

fig, ax = plt.subplots(2)
ax[0].plot(cost,'r',lw=1.3)
# ax.semilogy(cost,'r',lw=1.3)
ax[0].set_ylabel("cost")
ax[0].set_xlabel("iteration")

agent.restore(checkpoint_path)

## print
import time
prob_config['num_random_ref'] = 0
episode_cost = 0
env = DerefVariableInitMesh(**prob_config)
done = False
obs = env.reset()
print("Num. Elems. = ", env.mesh.GetNE())
env.render()

while not done:
    action = agent.compute_action(obs,explore=False)
    obs, reward, done, info = env.step(action)
    episode_cost -= reward
    rlcost = episode_cost
    print("step = ", env.n)
    print("refine action   = ", action[0].item())
    print("derefine action = ", action[1].item())
    print("Num. Elems. = ", env.mesh.GetNE())
    print("episode cost = ", episode_cost)
    time.sleep(0.5)
env.render()
# env.render_mesh()

costs = []
rlcosts = []
actions = []
nth = 11
for i in range(1, nth):
    action = np.array([i/(nth-1),0])
    actions.append(action[0].item())
    rlcosts.append(rlcost)
    env.reset()
    done = False
    episode_cost = 0
    while not done:
        _, reward, done, info = env.step(action)
        episode_cost -= reward 
        print("step = ", env.n)
        print("refine action   = ", action[0].item())
        print("derefine action = ", action[1].item())
        print("Num. Elems. = ", env.mesh.GetNE())
        print("episode cost = ", episode_cost)
    env.render()    
    costs.append(episode_cost)

    
ax[1].plot(actions,costs,'-or',lw=1.3)
ax[1].plot(actions,rlcosts,'-b',lw=1.3)
# ax.semilogy(cost,'r',lw=1.3)
ax[1].set_ylabel("cost")
ax[1].set_xlabel("Constant Actions (theta)")
plt.show()