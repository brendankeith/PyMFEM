"""
    
    Script to train an optimal marking policy starting from a fixed initial mesh

    * In order for this to work with rllib, you need to have RLModule in your PYTHONPATH

"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from prob_envs.StationaryProblem import StationaryProblem
# from prob_envs.VariableInitialMesh import VariableInitialMesh
import numpy as np
# from prob_envs.FixedInitialMesh import FixedInitialMesh
from drawnow import drawnow, figure

scores = []
avg_scores = []
def draw_fig():
  plt.title('reward')
  plt.plot(scores, 'r-')
  plt.plot(avg_scores, 'b-')

prob_config = {
    # 'mesh_name'         : 'star.mesh',
    'mesh_name'         : 'l-shape.mesh',
    'num_unif_ref'      : 1,
    # 'num_random_ref'    : 2,
    'order'             : 2,
    'optimization_type' : 'step_threshold', # 'error_threshold', 'dof_threshold', 'step_threshold'
    # 'random_mesh'       : True
}

total_episodes = 4000
batch_size = 16
nbatches = int(total_episodes/batch_size)
checkpoint_period = 200

config = ppo.DEFAULT_CONFIG.copy()
config['train_batch_size'] = batch_size
config['sgd_minibatch_size'] = batch_size
config['rollout_fragment_length'] = batch_size
config['num_workers'] = 3
config['num_gpus'] = 0
config['gamma'] = 1.0
config['lr'] = 1e-4


ray.shutdown()
ray.init(ignore_reinit_error=True)
env = StationaryProblem(**prob_config)
register_env("my_env", lambda config : StationaryProblem(**prob_config))
agent = ppo.PPOTrainer(env="my_env", config=config)

episode = 0
checkpoint_episode = 0
for n in range(nbatches):
    print("training batch %d of %d batches" % (n+1,nbatches))
    result = agent.train()
    episode += config['train_batch_size']
    checkpoint_episode += config['train_batch_size']
    episode_score = -result["episode_reward_mean"]
    print ("Episode cost", episode_score)
    # scores.append(episode_score)
    # if n == 0:
    #     avg_scores.append(episode_score)
    # else:
    #     avg_scores.append(avg_scores[-1] * 0.99 + episode_score*0.01)
    
    # drawnow(draw_fig)
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
# plt.show()

agent.restore(checkpoint_path)


prob_config['random_mesh'] = False

## print
import time
prob_config['num_random_ref'] = 0
episode_cost = 0
done = False
obs = env.reset()
print("Num. Elems. = ", env.mesh.GetNE())
env.render()
rlcost = 0
rlactions = []
while not done:
    action = agent.compute_action(obs,explore=False)
    rlactions.append(action)
    obs, reward, done, info = env.step(action)
    episode_cost -= reward 
    rlcost = episode_cost
    print("step = ", env.k)
    print("action = ", action.item())
    print("Num. Elems. = ", env.mesh.GetNE())
    print("episode cost = ", episode_cost)
    time.sleep(0.05)
    env.RenderMesh()


costs = []
rlcosts = []
actions = []
nth = 11
for i in range(1, nth):
    action = np.array([i/(nth-1)])
    actions.append(action.item())
    rlcosts.append(rlcost)
    env.reset()
    done = False
    episode_cost = 0
    while not done:
        _, reward, done, info = env.step(action)
        episode_cost -= reward 
        print("step = ", env.k)
        print("action = ", action.item())
        print("Num. Elems. = ", env.mesh.GetNE())
        print("episode cost = ", episode_cost)
    costs.append(episode_cost)

ax[1].plot(actions,costs,'-or',lw=1.3)
ax[1].plot(actions,rlcosts,'-b',lw=1.3)
# ax.semilogy(cost,'r',lw=1.3)
ax[1].set_ylabel("cost")
ax[1].set_xlabel("Constant Actions (theta)")

plt.show()