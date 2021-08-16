"""
    
    2D hp experiments to be used in paper
    code by Justin Crum, Andrew Gillette, Brendan Keith, Socratis Petrides
    June - Sep 2021
    
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

from optparse import OptionParser



def run_experiment():

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
        'BC_type' : 'Exact', #Homogeneous, Exact, Random
        'mesh_type'         :  'RandomAngle', #RandomAngle, Fixed
        # 'random_mesh'       : True
        #'error_threshold' : 2e-2,  #default is 1e-3
        'dof_threshold' : 1e4 #default is 1e4
    }

    data_deterministic_no_flagging = False  #Set this to True if you want to do a deterministic policy data collection with no flagging.
    data_deterministic = True              #Set this to True if you want to do a deterministic policy data collection with flagging.
                                            #Note that the deterministic policy with flagging is currently set up to run 100 values of theta
                                            #from 0.0 to 0.99, so it will override any theta value that is passed into it.
    training = True                         #Set this to True if you want to train a policy.
    evaluation = True                       #Set this to True if you want to evaluate a trained policy.  You must have training set to True as well or give a checkpoint path.
    distribution = False                     #Set this to True if you want to do either a distribution of deterministic policies or if you want to do
                                            #evaluations on on many meshes where you use the average episode cost as the value of interest.
    save_mesh_files = True


    total_episodes = 100 # 1000 in Test9
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

    checkpoint_path = "/home/justin/ray_results/PPO_my_env_2021-08-06_13-02-15djgz1kc7/checkpoint_000500/checkpoint-500"

    if data_deterministic:
        env.hpDeterministicPolicy(0.6, Distribution = distribution)

    if training:
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

    if evaluation == True:
        #agent.restore(checkpoint_path)
        prob_config['random_mesh'] = False

        import time
        prob_config['num_random_ref'] = 0
        rows = []

        if distribution:
            total_cost = 0.0
            num_episodes = 20
            headers = ['Average Cost']
            for _ in range(num_episodes):
                episode_cost = 0
                done = False
                obs = env.reset_to_new_mesh(newmesh=True, mesh_type='RandomAngle')
                rlcost = 0
                rlactions = []
                while not done:
                    action = agent.compute_action(obs,explore=False)
                    rlactions.append(action)
                    obs, reward, done, info = env.step(action)
                    episode_cost -= reward 
                    rlcost = episode_cost
                total_cost += episode_cost
            average_episode_cost = total_cost / num_episodes
            rows.append([average_episode_cost])
            env.RenderHPmesh()
            print("average episode cost = ", average_episode_cost)
            with open('datafile', 'a') as datafile:
                write = csv.writer(datafile)
                write.writerow(headers)
                write.writerows(rows)

        else:
            headers = ['theta', 'rho', 'N', 'DoFs', 'Total_DoFs', 'Error_Estimate', 'Cost']#, 'L2_Error', 'H1_Error']
            episode_cost = 0
            done = False
            obs = env.reset_to_new_mesh(newmesh=True, mesh_type='RandomAngle')
            rlcost = 0
            rlactions = []
            while not done:
                action = agent.compute_action(obs,explore=False)
                #rows.append([action[0].item(), action[1].item(), env.mesh.GetNE(), env.fespace.GetTrueVSize(), 
                #            env.sum_of_dofs, env.global_error, episode_cost])#, env.L2error, env.H1error])
                rlactions.append(action)
                obs, reward, done, info = env.step(action)
                episode_cost -= reward 
                rlcost = episode_cost
                if save_mesh_files:
                    env.RenderHPmesh()
                    env.mesh.Save('mesh_file_step_' + str(env.k))
                # print("step = ", env.k)
                # print("action = ", action)
                # print("Num. Elems. = ", env.mesh.GetNE())
                # print("episode cost = ", episode_cost)
                # print("Num of dofs = ", env.sum_of_dofs)
                # print("Global Error = ", env.global_error)
                # env.compute_error_values()
            rows.append([action[0].item(), action[1].item(), env.mesh.GetNE(), env.fespace.GetTrueVSize(), 
                        env.sum_of_dofs, env.global_error, episode_cost])
            #env.RenderHPmesh()
            print("episode cost = ", episode_cost)
            with open('datafile', 'w') as datafile:
                write = csv.writer(datafile)
                write.writerow(headers)
                write.writerows(rows)

    if data_deterministic_no_flagging == True:
        headers = ['theta', 'rho', 'N', 'DoFs', 'Total_DoFs', 'Error_Estimate', 'Cost']#, 'L2_Error', 'H1_Error']
        rows = []
        for j in range(10):
            for k in range(10):
                done = False
                env.reset_to_new_mesh(newmesh=True)
                theta = j / 10
                rho = k / 10
                episode_cost = 0.0
                while not done:
                    action = np.array([theta, rho])
                    obs, reward, done, info = env.step(action)
                    episode_cost -= reward 
                    rlcost = episode_cost
                
                rows.append([action[0].item(), action[1].item(), env.mesh.GetNE(), env.fespace.GetTrueVSize(), 
                                env.sum_of_dofs, env.global_error, episode_cost])#, env.L2error, env.H1error])

        with open('datafile', 'w') as datafile:
            write = csv.writer(datafile)
            write.writerow(headers)
            write.writerows(rows)



if __name__ == '__main__':

    #==================================================================================================#
    # Provide help screen documentation. Let the user define options. Also define defaults.            #
    #==================================================================================================#

    doc_string = "\nSpecify function from test_functions.py.  Compare neural net model to DelaunaySparse model"
    usage = doc_string + "\nUsage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option( "--jobid", help="Job ID (from SLURM).", 
        dest="jobid", type=int, default=999999)   
    parser.add_option("--detnoflag", dest="do_detnoflag", action="store_true", default=False,
	    help="Deterministic policy data collection with no flagging. Default False.")
    parser.add_option("--detwithflag", dest="do_detflag", action="store_true", default=False,
	    help="Deterministic policy data collection with flagging. Currently set up to run 100 values of theta from 0.0 to 0.99; will override any theta value that is passed in. Default False.")
    parser.add_option("--train", dest="do_train", action="store_true", default=False,
	    help="Do RL training. Default False.")
    parser.add_option("--eval", dest="do_eval", action="store_true", default=False,
	    help="Evaluate a trained policy. You must have training set to True as well or give a checkpoint path. Default False. ***** MODIFY THIS OPTION")
    parser.add_option("--dist", dest="do_dist", action="store_true", default=False,
	    help="Do either a distribution of deterministic policies or evaluations on on many meshes where you use the average episode cost as the value of interest. Default False. ***** MODIFY THIS OPTION")
    parser.add_option("--savemesh", dest="save_mesh", action="store_true", default=False,
	    help="Save mesh files at each step.  Default False")
    
    # total_episodes = 100 # 1000 in Test9
    # batch_size = 16

    # parser.add_option("--exp", dest="exptype", type=int, default=1,
    #     help="Experiment type: 1 = compare DS and NN.  2 = increase sample and DS only.  Default = 1.")
    # parser.add_option( "--fn", help="Test function to use: options include griewank, rosenbrock, paraboloid, michalewicz (default).", 
    #     dest="fn_name", type=str, default="michalewicz")   
    # parser.add_option( "--dim", dest="dim", type=int, default=0, 
    #     help="Dimension of input space.  Required input")
    # parser.add_option("--viz", dest="do_viz", action="store_true", default=False,
	#     help="Do visualization. Default False.")
    (options, args) = parser.parse_args()
    

    def echo_options(options):
        print("User-input options:")
        print("Job ID      :", options.jobid)
        print("Detwithflag?:", options.do_detflag)
        # print("Exprmt  type:", options.exptype)
        # print("Learn ratio: ", options.learnratio)
        # print("Num hdn lyrs:", options.numhiddenlyrs)
        # print("Loss func:   ", options.loss_fn)
        
        
        # if (options.exptype == -1):
        #     print("Need to specify experiment type with --exp")
        #     exit()

    echo_options(options)
    run_experiment()

