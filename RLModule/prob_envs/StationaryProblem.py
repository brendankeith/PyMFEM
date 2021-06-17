import os
from os.path import expanduser, join
import gym
from gym import spaces
import numpy as np
import mfem.ser as mfem
from mfem.ser import intArray
from utils.StatisticsAndCost import Statistics, GlobalError

class StationaryProblem(gym.Env):

    def __init__(self,**kwargs):
        super().__init__()
        self.BC = mfem.ConstantCoefficient(0.0)
        self.RHS = mfem.ConstantCoefficient(1.0)
        self.coeff = mfem.ConstantCoefficient(1.0)

        self.optimization_type = kwargs.get('optimization_type','error_threshold')
        self.error_threshold = kwargs.get('error_threshold',5e-4)
        self.dof_threshold = kwargs.get('dof_threshold',1e4)
        self.step_threshold = kwargs.get('step_threshold',10)
        mesh_name = kwargs.get('mesh_name','l-shape.mesh')
        num_unif_ref = kwargs.get('num_unif_ref',1)
        order = kwargs.get('order',1)    
        meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', mesh_name))
        mesh = mfem.Mesh(meshfile)
        mesh.EnsureNCMesh()
        for _ in range(num_unif_ref):
            mesh.UniformRefinement()
        self.dim = mesh.Dimension()
        self.initial_mesh = mesh
        self.order = order
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
    
    def reset(self):
        self.k = 0
        self.mesh = mfem.Mesh(self.initial_mesh)
        self.Setup()
        self.AssembleAndSolve()
        self.errors = self.GetLocalErrors()
        obs = self.Errors2Observation(self.errors)
        self.global_error = GlobalError(self.errors)
        self.sum_of_dofs = self.fespace.GetTrueVSize()
        return obs
    
    def step(self, action):
        self.k += 1
        self.UpdateMesh(action)
        self.AssembleAndSolve()
        self.errors = self.GetLocalErrors()
        obs = self.Errors2Observation(self.errors)
        num_dofs = self.fespace.GetTrueVSize()
        if self.optimization_type == 'error_threshold':
            self.global_error = GlobalError(self.errors)
            cost = np.log(1.0 + num_dofs/self.sum_of_dofs)
            self.sum_of_dofs += num_dofs
            if self.global_error < self.error_threshold:
                done = True
            else:
                done = False
            if self.sum_of_dofs > 1e6 or self.k > 100:
                cost = 0.0
                done = True
        elif self.optimization_type == 'dof_threshold':
            self.sum_of_dofs += self.fespace.GetTrueVSize()
            if self.sum_of_dofs > self.dof_threshold:
                cost = 0.0
                done = True
            else:
                global_error = GlobalError(self.errors)
                cost = np.log(global_error/self.global_error)
                self.global_error = global_error
                done = False
        else:
            if self.k == 1:
                accumulated_cost = 0
            else:
                accumulated_cost = np.log(self.global_error*self.sum_of_dofs)
            self.global_error = GlobalError(self.errors)
            self.sum_of_dofs += self.fespace.GetTrueVSize()
            cost = np.log(self.global_error*self.sum_of_dofs) - accumulated_cost
            if self.k >= self.step_threshold:
                done = True
            elif self.sum_of_dofs > self.dof_threshold:
                cost = 10.0
                done = True
            else:
                done = False
        info = {'global_error':self.global_error, 'num_dofs':num_dofs, 'max_local_errors':np.amax(self.errors)}
        return obs, -cost, done, info
    
    def render(self):
        sol_sock = mfem.socketstream("localhost", 19916)
        sol_sock.precision(8)
        sol_sock.send_solution(self.mesh,  self.x)
        title = "step " + str(self.k)
        sol_sock.send_text("window_title '" + title)

    def Setup(self):
        # print("Setting up Poisson problem ")
        dim = self.mesh.Dimension()
        fec = mfem.H1_FECollection(self.order, dim)
        self.fespace = mfem.FiniteElementSpace(self.mesh, fec)
        self.a = mfem.BilinearForm(self.fespace)
        self.b = mfem.LinearForm(self.fespace)
        integ = mfem.DiffusionIntegrator(self.coeff)
        self.a.AddDomainIntegrator(integ)
        self.b.AddDomainIntegrator(mfem.DomainLFIntegrator(self.RHS))
        self.x = mfem.GridFunction(self.fespace)
        self.x.Assign(0.0)
        self.ess_bdr = intArray(self.mesh.bdr_attributes.Max())
        self.ess_bdr.Assign(1)
        self.x.ProjectBdrCoefficient(self.BC, self.ess_bdr)
        self.flux_fespace = mfem.FiniteElementSpace(self.mesh, fec, dim)
        self.estimator =  mfem.ZienkiewiczZhuEstimator(integ, self.x, self.flux_fespace,
                                                       own_flux_fes = False)

        self.refiner = mfem.ThresholdRefiner(self.estimator)

    def Errors2Observation(self, errors):
        stats = Statistics(errors)
        obs = [stats.nels, stats.mean, stats.variance, stats.skewness, stats.kurtosis]
        return np.array(obs)

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

    def RenderMesh(self):
        sol_sock = mfem.socketstream("localhost", 19916)
        sol_sock.precision(8)
        zerogf = mfem.GridFunction(self.fespace)
        zerogf.Assign(0.0)
        sol_sock.send_solution(self.mesh, zerogf)
        title = "step " + str(self.k)
        sol_sock.send_text('keys ARjlmp*******' + " window_title '" + title)

    def UpdateMesh(self, action):
        theta = action.item() # refinement threshold
        if theta < 0. :
          theta = 0.
        if theta > 0.999 :
          theta = 0.999 
        self.Refine(theta)

    def Refine(self, theta):
        # self.refiner.Reset()
        self.refiner.SetTotalErrorFraction(theta)
        self.refiner.Apply(self.mesh)
        self.fespace.Update()
        self.x.Update()
        # self.fespace.UpdatesFinished()
        self.a.Update()
        self.b.Update()


class DeRefStationaryProblem(StationaryProblem):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    def UpdateMesh(self, action):
        theta1 = action[0].item() # refine threshold
        if theta1 < 0. :
          theta1 = 0.
        if theta1 > 0.999 :
          theta1 = 0.999 
        theta2 = action[1].item() # derefine threshold
        theta2 *= theta1 # enforces deref < ref threshold 
        self.Refine(theta1)
        self.Derefine(theta2)
    
    def Derefine(self, theta):
        threshold = theta * np.max(self.errors)
        rtransforms = self.mesh.GetRefinementTransforms()
        coarse_to_fine = mfem.Table()
        coarse_to_ref_type = mfem.intArray()
        ref_type_to_matrix = mfem.Table()
        ref_type_to_geom = mfem.GeometryTypeArray()
        rtransforms.GetCoarseToFineMap(self.mesh, coarse_to_fine, coarse_to_ref_type, ref_type_to_matrix, ref_type_to_geom)
        new_errors = mfem.doubleArray(coarse_to_fine.Width())
        tmp = mfem.intArray(1)
        for i in range(coarse_to_fine.Width()):
            new_errors[i] = mfem.infinity()
        for i in range(coarse_to_fine.Size()):
            if coarse_to_fine.RowSize(i) == 1:
                tmp_data = coarse_to_fine.GetRow(i)
                tmp.Assign(tmp_data)
                index = tmp[0]
                new_errors[index] = self.errors[i]
        self.mesh.DerefineByError(new_errors,threshold)
        
        # self.refiner.Reset()
        self.fespace.Update()
        self.x.Update()
        # self.fespace.UpdatesFinished()
        self.a.Update()
        self.b.Update()