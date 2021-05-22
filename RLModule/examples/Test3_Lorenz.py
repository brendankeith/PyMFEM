import gym
from gym import spaces, utils
import numpy as np
import os
from os.path import expanduser, join
import mfem.ser as mfem
from mfem.ser import intArray

import torch
import sys
sys.path.append("./RLModule")
sys.path.append("..")
from utils.StatisticsAndCost import StatisticsAndCost
# from glvis import glvis, to_stream

from ray.util import inspect_serializability


"""
    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed
    
    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

"""

class RefineAndEstimate(gym.Env):
    one  = mfem.ConstantCoefficient(1.0)
    zero = mfem.ConstantCoefficient(0.0)

    def __init__(self,config,**kwargs):
        super().__init__()
        mesh_name = kwargs.get('mesh_name','l-shape.mesh')
        num_unif_ref = kwargs.get('num_unif_ref',1)
        order = kwargs.get('order',1)

        meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', mesh_name))
        mesh = mfem.Mesh(meshfile)
        # for _ in range(num_unif_ref):
        #     mesh.UniformRefinement()
        self.initial_mesh = mesh
        self.order = order
        print("Number of Elements in mesh = " + str(self.initial_mesh.GetNE()))

        penalty = kwargs.get('penalty',0.0)
        self.stats = StatisticsAndCost(penalty)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
        self.n = 0
        
        # self.reset()
    
    # def reset(self):
    #     self.mesh = mfem.Mesh(self.initial_mesh)
    #     self.setup()
    #     self.AssembleAndSolve()
    #     errors = self.GetLocalErrors()
    #     return  self.errors2obs(errors)
    
    # def step(self, theta):
    #     th_temp = theta
    #     if th_temp < 0. :
    #       th_temp = 0.
    #     if th_temp > 0.99 :
    #       th_temp = 0.99 

    #     self.RefineAndUpdate(th_temp)
    #     self.AssembleAndSolve()
    #     errors = self.GetLocalErrors()
    #     obs = self.errors2obs(errors)
    #     cost = self.errors2cost(errors)
    #     done = True if (self.mesh.GetNE() > 1e3) else False
    #     info = None
    #     return obs, -cost, done, info
    
    # def render(self):
    #     sol_sock = mfem.socketstream("localhost", 19916)
    #     sol_sock.precision(8)
    #     sol_sock.send_solution(self.mesh,  self.x)

    # def errors2obs(self, errors):
    #     stats = self.stats(errors)
    #     obs = [stats.nobs, stats.mean, stats.variance, stats.skewness, stats.kurtosis]
    #     return torch.tensor(obs).float()
    
    # def errors2cost(self, errors):
    #     stats = self.stats(errors)
    #     return stats.cost

    # def setup(self):
    #     # print("Setting up Poisson problem ")
    #     dim = self.mesh.Dimension()
    #     fec = mfem.H1_FECollection(self.order, dim)
    #     self.fespace = mfem.FiniteElementSpace(self.mesh, fec)
    #     self.a = mfem.BilinearForm(self.fespace)
    #     self.b = mfem.LinearForm(self.fespace)
    #     integ = mfem.DiffusionIntegrator(self.one)
    #     self.a.AddDomainIntegrator(integ)
    #     self.b.AddDomainIntegrator(mfem.DomainLFIntegrator(self.one))
    #     self.x = mfem.GridFunction(self.fespace)
    #     self.x.Assign(0.0)
    #     self.ess_bdr = intArray(self.mesh.bdr_attributes.Max())
    #     self.ess_bdr.Assign(1)
    #     self.flux_fespace = mfem.FiniteElementSpace(self.mesh, fec, dim)
    #     self.estimator =  mfem.ZienkiewiczZhuEstimator(integ, self.x, self.flux_fespace,
    #                                                    own_flux_fes = False)

    #     self.refiner = mfem.ThresholdRefiner(self.estimator)
    #     self.refiner.SetTotalErrorFraction(0.7)

    #   #   print("Poisson problem setup finished ")

    # def AssembleAndSolve(self):
    #     self.a.Assemble()
    #     self.b.Assemble()
    #     ess_tdof_list = intArray()
    #     self.fespace.GetEssentialTrueDofs(self.ess_bdr, ess_tdof_list)
    #     A = mfem.OperatorPtr()
    #     B = mfem.Vector();  X = mfem.Vector()
    #     self.a.FormLinearSystem(ess_tdof_list, self.x, self.b, A, X, B, 1)
    #     AA = mfem.OperatorHandle2SparseMatrix(A)
    #     M = mfem.GSSmoother(AA)
    #     mfem.PCG(AA, M, B, X, -1, 200, 1e-12, 0.0)
    #     self.a.RecoverFEMSolution(X,self.b,self.x)
    # #   return None

    # def GetLocalErrors(self):
    #     mfem_errors = self.estimator.GetLocalErrors()
    #     errors = np.array([mfem_errors[i] for i in range(self.mesh.GetNE())])
    #     return errors

    # def RefineAndUpdate(self, theta):
    #     self.refiner.SetTotalErrorFraction(theta)
    #     self.refiner.Apply(self.mesh)
    #     self.fespace.Update()
    #     self.x.Update()
    #     self.a.Update()
    #     self.b.Update()


env = RefineAndEstimate(None)
inspect_serializability(env.initial_mesh)

env.render()

# env.reset()
# obs, reward, done, _ = env.step(0.5)
# obs, reward, done, _ = env.step(0.5)
# obs, reward, done, _ = env.step(0.5)
# env.render()


# from typing import Dict, TYPE_CHECKING
# import ray
# import ray.rllib.agents.ppo as ppo
# import ray.rllib.agents.dqn as dqn
# from ray.rllib.env import BaseEnv
# from ray.rllib.policy import Policy
# from ray.rllib.policy.sample_batch import SampleBatch
# from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
# from ray.rllib.agents.callbacks import DefaultCallbacks

# ray.shutdown()
# ray.init(ignore_reinit_error=True)

# total_episodes = 1.e6
# batch_size = 800
# nbatches = int(total_episodes/batch_size)

# config = ppo.DEFAULT_CONFIG.copy()
# config['train_batch_size'] = batch_size
# config['num_workers'] = 1
# config['num_gpus'] = 0

# os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"
# agent = ppo.PPOTrainer(config, env=RefineAndEstimate)
# policy = agent.get_policy()
# model = policy.model

# cor = []
# ref = []

# checkpoint_period = 5000

# neval = 500
# eval_period = 1000

# episode = 0
# checkpoint_episode = 0
# eval_episode = 0
# for n in range(nbatches):
#     print("training batch %d of size %d" % (n,config['train_batch_size']))
#     agent.train()
#     episode += config['train_batch_size']
#     checkpoint_episode += config['train_batch_size']
#     if (checkpoint_episode >= checkpoint_period):
#         checkpoint_episode = 0
#         checkpoint_path = agent.save()
#         print(checkpoint_path)
        
#     eval_episode += config['train_batch_size']
#     if (eval_period and eval_episode >= eval_period):
#         print("begin evaluation...")
#         eval_episode = 0
#         cor1, cor_ref1, cor_noref1, ref1 = eval_ensemble(model, neval)
#         cor.append(cor1)
#         ref.append(ref1)