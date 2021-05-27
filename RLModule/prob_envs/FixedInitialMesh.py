import os
from os.path import expanduser, join
import gym
from gym import spaces
import numpy as np
import mfem.ser as mfem
from mfem.ser import intArray
from utils.StatisticsAndCost import StatisticsAndCost

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

class FixedInitialMesh(gym.Env):

    def __init__(self,**kwargs):
        super().__init__()
        self.one = mfem.ConstantCoefficient(1.0)
        self.zero = mfem.ConstantCoefficient(0.0)
        mesh_name = kwargs.get('mesh_name','l-shape.mesh')
        num_unif_ref = kwargs.get('num_unif_ref',1)
        order = kwargs.get('order',1)

        meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', mesh_name))
        mesh = mfem.Mesh(meshfile)
        for _ in range(num_unif_ref):
            mesh.UniformRefinement()
        self.initial_mesh = mesh
        self.order = order
        print("Number of Elements in mesh = " + str(self.initial_mesh.GetNE()))
        self.stats = StatisticsAndCost()

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
        self.previous_cost = 0.0
        self.reset()
    
    def reset(self):
        self.n = 0
        self.mesh = mfem.Mesh(self.initial_mesh)
        self.setup()
        self.AssembleAndSolve()
        errors = self.GetLocalErrors()
        return  self.errors2obs(errors)
    
    def step(self, action):
        self.n += 1
        th_temp = action.item()
        if th_temp < 0. :
          th_temp = 0.
        if th_temp > 0.99 :
          th_temp = 0.99 

        self.RefineAndUpdate(th_temp)
        self.AssembleAndSolve()
        errors = self.GetLocalErrors()
        obs = self.errors2obs(errors)
        cost = self.errors2cost(errors)
        if self.mesh.GetNE() > 100:
            cost = self.previous_cost
            done = True
        else:
            self.previous_cost = cost
            cost = 0.0
            done = False
        info = {}
        return obs, -cost, done, info
    
    def render(self):
        sol_sock = mfem.socketstream("localhost", 19916)
        sol_sock.precision(8)
        sol_sock.send_solution(self.mesh,  self.x)
        title = "step " + str(self.n)
        sol_sock.send_text("window_title '" + title)

    def errors2obs(self, errors):
        stats = self.stats(errors)
        obs = [stats.nels, stats.mean, stats.variance, stats.skewness, stats.kurtosis]
        return np.array(obs)
    
    def errors2cost(self, errors):
        stats = self.stats(errors)
        return stats.cost

    def setup(self):
        # print("Setting up Poisson problem ")
        dim = self.mesh.Dimension()
        fec = mfem.H1_FECollection(self.order, dim)
        self.fespace = mfem.FiniteElementSpace(self.mesh, fec)
        self.a = mfem.BilinearForm(self.fespace)
        self.b = mfem.LinearForm(self.fespace)
        integ = mfem.DiffusionIntegrator(self.one)
        self.a.AddDomainIntegrator(integ)
        self.b.AddDomainIntegrator(mfem.DomainLFIntegrator(self.one))
        self.x = mfem.GridFunction(self.fespace)
        self.x.Assign(0.0)
        self.ess_bdr = intArray(self.mesh.bdr_attributes.Max())
        self.ess_bdr.Assign(1)
        self.flux_fespace = mfem.FiniteElementSpace(self.mesh, fec, dim)
        self.estimator =  mfem.ZienkiewiczZhuEstimator(integ, self.x, self.flux_fespace,
                                                       own_flux_fes = False)

        self.refiner = mfem.ThresholdRefiner(self.estimator)
        self.refiner.SetTotalErrorFraction(0.7)

    def AssembleAndSolve(self):
        self.a.Assemble()
        self.b.Assemble()
        ess_tdof_list = intArray()
        self.fespace.GetEssentialTrueDofs(self.ess_bdr, ess_tdof_list)
        A = mfem.OperatorPtr()
        B = mfem.Vector();  X = mfem.Vector()
        self.a.FormLinearSystem(ess_tdof_list, self.x, self.b, A, X, B, 1)
        AA = mfem.OperatorHandle2SparseMatrix(A)
        M = mfem.GSSmoother(AA)
        mfem.PCG(AA, M, B, X, -1, 200, 1e-12, 0.0)
        self.a.RecoverFEMSolution(X,self.b,self.x)

    def GetLocalErrors(self):
        mfem_errors = self.estimator.GetLocalErrors()
        errors = np.array([mfem_errors[i] for i in range(self.mesh.GetNE())])
        return errors

    def RefineAndUpdate(self, theta):
        self.refiner.SetTotalErrorFraction(theta)
        self.refiner.Apply(self.mesh)
        self.fespace.Update()
        self.x.Update()
        self.a.Update()
        self.b.Update()