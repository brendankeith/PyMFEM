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
from prob_envs.StationaryProblem import DeRefStationaryProblem, DeRefStationaryProblemBob

train = True
prob_config = {
    # 'mesh_name'         : 'star.mesh',
    # 'mesh_name'         : 'l-shape.mesh',
    'mesh_name'         : 'inline-quad.mesh',
    'num_unif_ref'      : 2,
    'num_random_ref'    : 0,
    'order'             : 2,
    'optimization_type' : 'error_threshold',
    # 'optimization_type' : 'dof_threshold',
    'problem_type' : 'wavefront',
    # 'problem_type' : 'laplace',
}

# env = DeRefStationaryProblem(**prob_config)
# env.reset()
# env.step(np.array([0.5,0.5]))
# env.step(np.array([0.5,0.0]))
# env.step(np.array([0.5,0.0]))
# env.step(np.array([0.5,0.0]))
# env.step(np.array([0.5,0.0]))
# env.step(np.array([0.5,0.0]))
# env.step(np.array([0.5,0.0]))
# env.render()

total_episodes = 10000
batch_size = 64
checkpoint_period = 200

# # short run - for debugging
# total_episodes = 100
# batch_size = 10
# checkpoint_period = 20

nbatches = int(total_episodes/batch_size)

config = ppo.DEFAULT_CONFIG.copy()
config['train_batch_size'] = batch_size
config['sgd_minibatch_size'] = batch_size
config['rollout_fragment_length'] = batch_size
config['num_workers'] = 6
config['num_gpus'] = 0
config['gamma'] = 1.0
config['lr'] = 1e-4

ray.shutdown()
ray.init(ignore_reinit_error=True)

register_env("my_env", lambda config : DeRefStationaryProblem(**prob_config))
agent = ppo.PPOTrainer(env="my_env", config=config)
policy = agent.get_policy()
model = policy.model

if train:
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
else:
    # checkpoint_path = '/Users/keith10/ray_results/PPO_my_env_2021-06-17_11-52-02npl_aius/checkpoint_000153/checkpoint-153'
    # checkpoint_path = '/Users/keith10/ray_results/PPO_my_env_2021-06-17_12-33-12wojlg07o/checkpoint_000153/checkpoint-153'
    # checkpoint_path = '/Users/keith10/ray_results/PPO_my_env_2021-06-21_13-52-2843jvi5fv/checkpoint_000153/checkpoint-153'
    checkpoint_path = '/Users/keith10/ray_results/PPO_my_env_2021-06-21_14-32-16_bpybyev/checkpoint_000153/checkpoint-153'

root_path, _ = os.path.split(checkpoint_path)
root_path, _ = os.path.split(root_path)
csv_path = root_path + '/progress.csv'
df = pd.read_csv(csv_path)
cost = -df.episode_reward_mean.to_numpy()

fig, ax = plt.subplots(4)
ax[0].plot(cost,'r',lw=1.3)
# ax.semilogy(cost,'r',lw=1.3)
ax[0].set_ylabel("cost")
ax[0].set_xlabel("iteration")

agent.restore(checkpoint_path)

## print
import time
prob_config['num_random_ref'] = 0
episode_cost = 0
env = DeRefStationaryProblem(**prob_config)
done = False
obs = env.reset()
print("Num. Elems. = ", env.mesh.GetNE())
# env.render()

ref_thetas = []
deref_thetas = []
max_local_errors = []
while True:
    action = agent.compute_action(obs,explore=False)
    obs, reward, done, info = env.step(action)
    if done:
        break
    episode_cost -= reward
    rlcost = episode_cost
    print("step = ", env.k)
    print("refine action   = ", action[0].item())
    print("derefine action = ", action[1].item())
    print("Num. Elems. = ", env.mesh.GetNE())
    print("Num dofs", info['num_dofs'])
    print("episode cost = ", episode_cost)
    print("Error estimate", info['global_error'])
    ref_thetas.append(action[0].item())
    deref_thetas.append(action[0].item()*action[1].item())
    max_local_errors.append(info['max_local_errors'])
env.RenderMesh()
ref_thetas = np.array(ref_thetas)
deref_thetas = np.array(deref_thetas)
max_local_errors = np.array(max_local_errors)

env.render()

ax[1].plot(ref_thetas,'r',lw=1.3,label='ref. param.')
ax[1].plot(deref_thetas,'b',lw=1.3,label='deref. param.')
ax[1].set_ylabel("Parameter")
ax[1].set_xlabel("Iteration")
ax[1].legend()

ax[2].plot(ref_thetas*max_local_errors,'r',lw=1.3,label='ref. thresh.')
ax[2].plot(deref_thetas*max_local_errors,'b',lw=1.3,label='deref. thresh.')
ax[2].set_ylabel("Threshold")
ax[2].set_xlabel("Iteration")
ax[2].legend()

costs = []
actions = []
nth = 11
for theta in np.linspace(0,0.9999,num=11):
    action = np.array([theta,0])
    actions.append(action[0].item())
    env.reset()
    done = False
    episode_cost = 0
    while not done:
        _, reward, done, info = env.step(action)
        episode_cost -= reward 
        if (int(env.k) == 1):
            print("")
        print("step = ", env.k)
        print("refine action   = ", action[0].item())
        print("derefine action = ", action[1].item())
        print("Num. Elems. = ", env.mesh.GetNE())
        print("episode cost = ", episode_cost)
    env.RenderMesh()    
    costs.append(episode_cost)

ax[3].plot(actions,costs,'-or',lw=1.3)
ax[3].plot([0,1],[rlcost,rlcost],'-b',lw=1.3)
# ax.semilogy(cost,'r',lw=1.3)
ax[3].set_ylabel("cost")
ax[3].set_xlabel("Constant Actions (theta)")


#############
## Bob's test
#############

# env_Bob = DeRefStationaryProblemBob(**prob_config)
# env_Bob.reset()
# null_action = np.zeros((2,))
# episode_cost = 0
# while True:
#     obs, reward, done, info = env_Bob.step(null_action)
#     if done:
#         break
#     episode_cost -= reward
#     bobcost = episode_cost
#     print("step = ", env_Bob.k)
#     print("Num. Elems. = ", env_Bob.mesh.GetNE())
#     print("Num dofs", info['num_dofs'])
#     print("episode cost = ", episode_cost)
#     print("Error estimate", info['global_error'])

# ax[3].plot([0,1],[bobcost,bobcost],'--g',lw=1.3)

plt.show()