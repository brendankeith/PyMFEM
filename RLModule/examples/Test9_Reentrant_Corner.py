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
from prob_envs.pacman_benchmark_problem import PacmanProblem
# from prob_envs.VariableInitialMesh import VariableInitialMesh
import numpy as np
# from prob_envs.FixedInitialMesh import FixedInitialMesh
from drawnow import drawnow, figure
import csv

scores = []
avg_scores = []
def draw_fig():
  plt.title('reward')
  plt.plot(scores, 'r-')
  plt.plot(avg_scores, 'b-')

prob_config = {
    # 'mesh_name'         : 'star.mesh',
    'mesh_name'         : 'l-shape-benchmark.mesh', #'l-shape-benchmark.mesh','circle_3_4.mesh'
    'mesh_name_two'     : 'circle_3_4.mesh',
    'num_unif_ref'      : 1, 
    # 'num_random_ref'    : 2,
    'refinement_strategy' : 'max', #'max', 'quantile', 'dorfler'
    'mode' : 'hp', #'hp', 'h'
    'order'             : 1,
    'optimization_type' : 'dof_threshold', # 'error_threshold', 'dof_threshold', 'step_threshold'
    'problem_type' : 'Homogeneous', #Homogeneous, Exact, Random
    'mesh_type'         :  'RandomAngle', #RandomAngle, Fixed
    # 'random_mesh'       : True
    #'error_threshold' : 2e-2,  #default is 1e-3
    'dof_threshold' : 1e4 #default is 1e4
}

total_episodes = 4000
batch_size = 16
nbatches = int(total_episodes/batch_size)
checkpoint_period = 0

config = ppo.DEFAULT_CONFIG.copy()
config['train_batch_size'] = 200
config['sgd_minibatch_size'] = 20
config['rollout_fragment_length'] = 5
config['num_workers'] = 4
config['num_gpus'] = 0
config['gamma'] = 1.0
config['lr'] = 1e-4


ray.shutdown()
ray.init(ignore_reinit_error=True)
env = PacmanProblem(**prob_config)
register_env("my_env", lambda config : PacmanProblem(**prob_config))
agent = ppo.PPOTrainer(env="my_env", config=config)


#checkpoint_path = "/home/justin/ray_results/PPO_my_env_2021-07-22_11-34-0626sk6yja/checkpoint_000250/checkpoint-250"

#env.RenderMesh()

#env.hpDeterministicPolicy(0.5)
#env.RenderHPmesh()
#env.reset()
#env.RenderHPmesh()
#env.step(np.array([0.5, 1.0]))
#env.RenderHPmesh()
#env.step(np.array([1.0, 0.25]))
#env.RenderHPmesh()

#env.reset()
#env.RenderHPmesh()
#env.reset()
#env.RenderHPmesh()
#env.reset()
#env.RenderHPmesh()

for j in range(1):
    #env.Continuation(j)
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
#ax.semilogy(cost,'r',lw=1.3)
ax[0].set_ylabel("cost")
ax[0].set_xlabel("iteration")
plt.show()




agent.restore(checkpoint_path)
prob_config['random_mesh'] = False

## print
import time
prob_config['num_random_ref'] = 0
episode_cost = 0
done = False
#obs = env.reset()
#env.RenderHPmesh()
obs = env.reset_to_new_mesh()
#obs = env.SwapProblemType()
env.RenderHPmesh()
print("Num. Elems. = ", env.mesh.GetNE())
env.render()
rlcost = 0
rlactions = []



headers = ['theta', 'rho', 'N', 'DoFs', 'Total_DoFs', 'Error_Estimate', 'L2_Error', 'H1_Error']
rows = []
obs_header = ['N', 'Mean', 'Variance', 'Skewness', 'Kurtosis', 'Average_Order']
obs_rows = []
while not done:
    action = agent.compute_action(obs,explore=False)
    rlactions.append(action)
    obs, reward, done, info = env.step(action)
    episode_cost -= reward 
    rlcost = episode_cost
    print("step = ", env.k)
    print("action = ", action)
    print("Num. Elems. = ", env.mesh.GetNE())
    print("episode cost = ", episode_cost)
    print("Num of dofs = ", env.sum_of_dofs)
    print("Global Error = ", env.global_error)
    env.compute_error_values()
    rows.append([action[0].item(), action[1].item(), env.mesh.GetNE(), env.fespace.GetTrueVSize(), 
                 env.sum_of_dofs, env.global_error, env.L2error, env.H1error])
    obs_rows.append(obs)
    time.sleep(0.05)
    #env.RenderMesh()
    env.RenderHPmesh()

with open('datafile', 'w') as datafile:
    write = csv.writer(datafile)
    write.writerow(headers)
    write.writerows(rows)
with open('statsfile', 'w') as statsfile:
    write = csv.writer(statsfile)
    write.writerow(obs_header)
    write.writerows(obs_rows)

"""
agent.restore(checkpoint_path)
prob_config['random_mesh'] = False

## print
import time
prob_config['num_random_ref'] = 0
episode_cost = 0
done = False
obs = env.reset_to_new_mesh()
#obs = env.SwapProblemType()
print("Num. Elems. = ", env.mesh.GetNE())
env.render()
rlcost = 0
rlactions = []



headers = ['theta', 'rho', 'N', 'DoFs', 'Total_DoFs', 'Error_Estimate', 'L2_Error', 'H1_Error']
rows = []
obs_header = ['N', 'Mean', 'Variance', 'Skewness', 'Kurtosis', 'Average_Order']
obs_rows = []
while not done:
    action = agent.compute_action(obs,explore=False)
    rlactions.append(action)
    obs, reward, done, info = env.step(action)
    episode_cost -= reward 
    rlcost = episode_cost
    print("step = ", env.k)
    print("action = ", action)
    print("Num. Elems. = ", env.mesh.GetNE())
    print("episode cost = ", episode_cost)
    print("Num of dofs = ", env.sum_of_dofs)
    print("Global Error = ", env.global_error)
    env.compute_error_values()
    rows.append([action[0].item(), action[1].item(), env.mesh.GetNE(), env.fespace.GetTrueVSize(), 
                 env.sum_of_dofs, env.global_error, env.L2error, env.H1error])
    obs_rows.append(obs)
    time.sleep(0.05)
    #env.RenderMesh()
    env.RenderHPmesh()

with open('datafile2', 'w') as datafile2:
    write = csv.writer(datafile2)
    write.writerow(headers)
    write.writerows(rows)
with open('statsfile2', 'w') as statsfile2:
    write = csv.writer(statsfile2)
    write.writerow(obs_header)
    write.writerows(obs_rows)

"""

"""
costs = []
#rlcosts = []
actions = []
nth = 11
headers = ['theta', 'N', 'DoFs', 'Total_DoFs', 'Error_Estimate', 'L2_Error', 'H1_Error']
#rows = []
with open('datafile', 'w') as datafile:
    write = csv.writer(datafile)
    write.writerow(headers)
    for i in range(0, nth - 1):
        #for j in range(0, 2):
        action = np.array([i/(nth-1)])
        #action = {'order' : j, 'space' : i / (nth-1)}
        # actions.append(action.item())
        #rlcosts.append(rlcost)
        env.hpDeterministicPolicy(action)
        env.reset()
        #write.writerow(headers)
        write.writerows(env.rows)
    done = False
    episode_cost = 0
    while not done:
        _, reward, done, info = env.step(action)
        episode_cost -= reward 
        print("step = ", env.k)
        #print("action = ", action.item())
        print("Num. Elems. = ", env.mesh.GetNE())
        print("episode cost = ", episode_cost)
        #costs.append(episode_cost)
        rows.append([action[0].item(), env.mesh.GetNE(), env.fespace.GetTrueVSize(), env.sum_of_dofs, env.global_error, env.L2error, env.H1error])
    

#with open('datafile', 'w') as datafile:
#    write = csv.writer(datafile)
#    write.writerow(headers)
#    write.writerows(rows)

#ax[1].plot(costs,'-or',lw=1.3)
#ax[1].plot(rlcosts,'-b',lw=1.3)
# ax.semilogy(cost,'r',lw=1.3)
#ax[1].set_ylabel("cost")
#ax[1].set_xlabel("Constant Actions (theta)")

#plt.show()
"""
