"""
    
    Script to train an optimal marking policy starting from a fixed initial mesh

    * In order for this to work with rllib, you need to have RLModule in your PYTHONPATH

"""

import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from prob_envs.MovingLoad import MovingLoadProblem

# prob_config = {
#     'optimization_type' : 'dof_threshold',
#     'mesh_name'         : 'inline-quad.mesh',
#     'num_unif_ref'      : 3,
#     'order'             : 2,
#     'error_target'      : 1e-4,
#     'dof_threshold'     : 1e4,
# }

# prob_config = {
#     'optimization_type' : 'error_threshold',
#     'mesh_name'         : 'inline-quad.mesh',
#     'num_unif_ref'      : 3,
#     'order'             : 2,
#     'error_threshold'   : 1e-4,
#     'dof_threshold'     : 1e4,
#     'penalty_rate'      : 0.95,
# }

prob_config = {
    'optimization_type' : 'convex_combination',
    'mesh_name'         : 'inline-quad.mesh',
    'num_unif_ref'      : 3,
    'order'             : 1,
    'dof_threshold'     : 5e4,
    'convex_coeff'      : 0.20, # E[ alpha*log_num_dofs/d + (1-alpha)*log_global_error ]
}

# env = MovingLoadProblem(**prob_config)
# env.reset()
# env.step(np.array([0.9,0.1]))
# env.RenderRHS()
# env.step(np.array([0.9,0.1]))
# env.RenderRHS()
# env.step(np.array([0.9,0.1]))
# env.RenderRHS()
# env.step(np.array([0.9,0.1]))
# env.RenderRHS()

total_episodes = 1000
batch_size = 64
checkpoint_period = 200

nbatches = int(total_episodes/batch_size)

config = ppo.DEFAULT_CONFIG.copy()
config['train_batch_size'] = batch_size
config['sgd_minibatch_size'] = batch_size
config['rollout_fragment_length'] = batch_size
config['num_workers'] = 6
config['num_gpus'] = 0
config['gamma'] = 1.0
config['lr'] = 1e-3
config['no_done_at_end'] = True

ray.shutdown()
ray.init(ignore_reinit_error=True)

register_env("my_env", lambda config : MovingLoadProblem(**prob_config))
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

fig, ax = plt.subplots(3)
ax[0].plot(cost,'r',lw=1.3)
ax[0].set_ylabel("cost")
ax[0].set_xlabel("iteration")

agent.restore(checkpoint_path)

## print
wait = input("Press any key to continue.")
time.sleep(0.5)
prob_config['num_random_ref'] = 0
episode_cost = 0
env = MovingLoadProblem(**prob_config)
done = False
obs = env.reset()
print("Num. Elems. = ", env.mesh.GetNE())
# env.render()

ref_thetas = []
deref_thetas = []
max_local_errors = []
for _ in range(200):
    action = agent.compute_action(obs,explore=False)
    obs, reward, done, info = env.step(action)
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
    time.sleep(0.1)
    env.RenderMesh()
ref_thetas = np.array(ref_thetas)
deref_thetas = np.array(deref_thetas)
max_local_errors = np.array(max_local_errors)

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

# costs = []
# rlcosts = []
# actions = []
# nth = 11
# for i in range(1, nth):
#     action = np.array([i/(nth-1),0])
#     actions.append(action[0].item())
#     rlcosts.append(rlcost)
#     env.reset()
#     done = False
#     episode_cost = 0
#     while not done:
#         _, reward, done, info = env.step(action)
#         episode_cost -= reward 
#         print("step = ", env.k)
#         print("refine action   = ", action[0].item())
#         print("derefine action = ", action[1].item())
#         print("Num. Elems. = ", env.mesh.GetNE())
#         print("episode cost = ", episode_cost)
#     env.RenderMesh()    
#     costs.append(episode_cost)

    
# ax[1].plot(actions,costs,'-or',lw=1.3)
# ax[1].plot(actions,rlcosts,'-b',lw=1.3)
# # ax.semilogy(cost,'r',lw=1.3)
# ax[1].set_ylabel("cost")
# ax[1].set_xlabel("Constant Actions (theta)")
plt.show()