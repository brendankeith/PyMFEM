from mfem._ser.coefficient import ConstantCoefficient
import os
from os.path import expanduser, join
import gym
from gym import spaces
import numpy as np
from tensorflow.python.ops.gen_array_ops import Const
import mfem.ser as mfem
from mfem.ser import intArray
from utils.StatisticsAndCost import Statistics, GlobalError
from utils.solution_wavefront import *
from utils.Solution_LShaped import *

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import dsolve

import pandas as pd

from utils.Solution_LShaped import *
from utils.Solution_SinSin import *
from utils.ExactErrorEstimators import *

class StationaryProblem(gym.Env):

    def __init__(self,**kwargs):
        super().__init__()
        self.nc_limit = 1
        self.problem_type = kwargs.get('problem_type','laplace')
        self.estimator_type = kwargs.get('estimator_type', 'ZZ')

        if (self.problem_type == 'laplace'):
            self.BC = mfem.ConstantCoefficient(0.0)
            self.RHS = mfem.ConstantCoefficient(1.0)
            self.coeff = mfem.ConstantCoefficient(1.0)
        elif (self.problem_type == 'wavefront'):
            self.BC = WavefrontSolutionCoefficient()
            self.RHS = WavefrontRHSCoefficient()
            self.coeff = mfem.ConstantCoefficient(1.0)
        elif (self.problem_type == 'lshaped'):
            self.BC = mfem.NumbaFunction(LShapedExact, 2).GenerateCoefficient()
            self.RHS = mfem.ConstantCoefficient(0.0)
            self.GradSoln = mfem.VectorNumbaFunction(LShapedExactGrad, 2, 2).GenerateCoefficient()
            self.coeff = mfem.ConstantCoefficient(1.0)
        elif (self.problem_type == 'sinsin'):
            self.BC = mfem.NumbaFunction(SinSinExact, 2).GenerateCoefficient()
            self.RHS = mfem.NumbaFunction(SinSinExactLaplace, 2).GenerateCoefficient()
            self.GradSoln = mfem.VectorNumbaFunction(SinSinExactGrad, 2, 2).GenerateCoefficient()
            self.coeff = mfem.ConstantCoefficient(1.0)
        else:
            print("Problem type not recognized.  Exiting.")
            exit()
        
        if (self.problem_type != 'ZZ') and (not hasattr(self, 'GradSoln')):
            print("Cannot measure H1 error for this problem.  Exiting.")
            exit()

        self.optimization_type = kwargs.get('optimization_type','error_threshold')
        self.error_threshold = kwargs.get('error_threshold',1e-3)
        self.dof_threshold = kwargs.get('dof_threshold',5e4)
        self.step_threshold = kwargs.get('step_threshold',20)
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
        self.action_space = spaces.Box(low=0.0, high=0.999, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
        self.error_file_name = kwargs.get('error_file_name','./RLModule/out/errors.csv')

        self.zero = mfem.ConstantCoefficient(0.0)
        self.zero_vector = mfem.Vector(self.dim)
        self.zero_vector.Assign(0.0)
        self.zerovector = mfem.VectorConstantCoefficient(self.zero_vector)

        self.global_errors = [ [] for _ in range(4)]
    
    def reset(self, save_errors=False):
        if save_errors:
            self.save_errors = True
            self.df_ErrorHistory = pd.DataFrame()
        else:
            self.save_errors = False
        self.k = 0
        self.mesh = mfem.Mesh(self.initial_mesh)
        self.Setup()
        self.AssembleAndSolve()
        self.errors = self.GetLocalErrors()
        self.global_error = max(GlobalError(self.errors),1e-8)
        self.sum_of_dofs = self.fespace.GetTrueVSize()
        obs = self.GetObservation()
        if self.save_errors:
            self.SaveErrorsToFile()
        return obs
    
    def step(self, action):
        self.k += 1
        self.UpdateMesh(action)
        self.AssembleAndSolve()
        self.errors = self.GetLocalErrors()
        num_dofs = self.fespace.GetTrueVSize()
        if self.optimization_type == 'error_threshold':
            self.global_error = GlobalError(self.errors)
            cost = np.log(1.0 + num_dofs/self.sum_of_dofs)
            self.sum_of_dofs += num_dofs
            if self.global_error < self.error_threshold:
                cost = 0.0
                done = True
            else:
                done = False
            if self.sum_of_dofs > self.dof_threshold:
                cost += 10.0
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
        obs = self.GetObservation()
        info = {'global_error':self.global_error, 'num_dofs':num_dofs, 'max_local_errors':np.amax(self.errors)}
        if self.save_errors:
            self.SaveErrorsToFile()
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
        if self.estimator_type == 'ZZ':
            self.estimator =  mfem.ZienkiewiczZhuEstimator(integ, self.x, self.flux_fespace,
                                                        own_flux_fes = False)
        else:
            self.estimator = H10ErrorEstimator(self.x, self.GradSoln)
            # print("Error estimator not recognized.  Exiting.")
            # exit()
        # self.refiner = mfem.ThresholdRefiner(self.estimator)

    def GetObservation(self):
        num_dofs = self.fespace.GetTrueVSize()
        stats = Statistics(self.errors, num_dofs=num_dofs)
        # rel_dof_threshold = (np.log(self.dof_threshold) - np.log(self.sum_of_dofs))/np.log(self.dof_threshold)
        # rel_error_threshold = (np.log(self.error_threshold) - np.log(self.global_error))/np.log(self.error_threshold)
        # obs = [rel_dof_threshold, rel_error_threshold, stats.mean, stats.variance, stats.skewness, stats.kurtosis]
        obs = [stats.nels, stats.mean, stats.variance, stats.skewness, stats.kurtosis]
        return np.array(obs)

    def AssembleAndSolve(self):
        self.a.Assemble()
        self.b.Assemble()
        self.x.Assign(0.0)
        self.x.ProjectBdrCoefficient(self.BC, self.ess_bdr)
        ess_tdof_list = intArray()
        self.fespace.GetEssentialTrueDofs(self.ess_bdr, ess_tdof_list)
        A = mfem.OperatorPtr()
        B = mfem.Vector();  X = mfem.Vector()
        self.a.FormLinearSystem(ess_tdof_list, self.x, self.b, A, X, B, 1)
        AA = mfem.OperatorHandle2SparseMatrix(A)
        M = mfem.GSSmoother(AA)
        mfem.CG(AA, B, X, -1, 2000, 1e-30, 0.0)
        # mfem.PCG(AA, M, B, X, -1, 200, 1e-12, 0.0)
        self.a.RecoverFEMSolution(X,self.b,self.x)
        self.solution_norm = self.x.ComputeGradError(self.zerovector) + 1e-12
        # self.solution_norm = self.x.ComputeH1Error(self.zero, self.zerovector)

    def GetLocalErrors(self):
        self.estimator.Reset()
        self.mfem_errors = self.estimator.GetLocalErrors()
        errors = np.array([self.mfem_errors[i] for i in range(self.mesh.GetNE())])# / self.solution_norm
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
        action = np.clip(action, 0.0, 1.0)
        theta = action.item() # refinement threshold
        self.Refine(theta)

    # def Refine(self, theta):
    #     self.refiner.Reset()
    #     self.refiner.SetTotalErrorFraction(theta)
    #     self.refiner.Apply(self.mesh)
    #     self.fespace.Update()
    #     self.x.Update()
    #     # self.fespace.UpdatesFinished()
    #     self.a.Update()
    #     self.b.Update()

    def Refine(self, theta):
        threshold = theta * np.max(self.errors)
        self.mesh.RefineByError(self.mfem_errors,threshold, -1, self.nc_limit)
        self.fespace.Update()
        self.x.Update()
        self.a.Update()
        self.b.Update()

    def SaveErrorsToFile(self):
        num_dofs = self.fespace.GetTrueVSize()
        df_tmp = pd.DataFrame({str(num_dofs):self.errors})
        self.df_ErrorHistory = pd.concat([self.df_ErrorHistory,df_tmp], axis=1)
        self.df_ErrorHistory.to_csv(self.error_file_name, index=False)

    def GlobalErrorEstimator(self):
        alpha = 1
        HDIVfec = mfem.RT_FECollection(self.order-1, self.dim)
        HDIVfespace = mfem.FiniteElementSpace(self.mesh, HDIVfec)
        b = mfem.LinearForm(HDIVfespace)
        a = mfem.BilinearForm(HDIVfespace)
        y = mfem.GridFunction(HDIVfespace)

        grad_u_h = mfem.GradientGridFunctionCoefficient(self.x)
        b1 = mfem.VectorFEDomainLFIntegrator(grad_u_h)
        minus_f = mfem.ProductCoefficient(-alpha,self.RHS)
        b2 = mfem.VectorFEDomainLFDivIntegrator(minus_f)
        b.AddDomainIntegrator(b1)
        b.AddDomainIntegrator(b2)
        b.Assemble()

        alpha_coeff = mfem.ConstantCoefficient(alpha)
        one = mfem.ConstantCoefficient(1.0)
        a.AddDomainIntegrator(mfem.DivDivIntegrator(alpha_coeff))
        a.AddDomainIntegrator(mfem.VectorFEMassIntegrator(one))
        a.Assemble()

        A = mfem.OperatorPtr()
        B = mfem.Vector()
        X = mfem.Vector()
        ess_tdof_list = intArray()
        a.FormLinearSystem(ess_tdof_list, y, b, A, X, B)

        AA = mfem.OperatorHandle2SparseMatrix(A)
        M = mfem.GSSmoother(AA)
        mfem.PCG(AA, M, B, X, 0, 10000, 1e-12, 0.0);

        a.RecoverFEMSolution(X, b, y)

        global_error_estimate = y.ComputeL2Error(grad_u_h)
        print("global error estimate = ", global_error_estimate)

        grad_u = mfem.VectorNumbaFunction(ExactGrad, self.mesh.SpaceDimension(), self.mesh.Dimension()).GenerateCoefficient()
        global_error = self.x.ComputeH1Error(self.BC,grad_u)
        print("global error = ", global_error)

        ZZ_error_estimate = sqrt(sum(np.array(self.errors) ** 2))
        print(" ZZ error estimate = ", ZZ_error_estimate)

        self.global_errors[0].append(self.fespace.GetTrueVSize())
        self.global_errors[1].append(global_error)
        self.global_errors[2].append(global_error_estimate/6)
        self.global_errors[3].append(ZZ_error_estimate)
        

class DeRefStationaryProblem(StationaryProblem):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    def UpdateMesh(self, action):
        action = np.clip(action, 0.0, 1.0)
        theta1 = action[0].item() # refine threshold
        theta2 = action[1].item() # derefine threshold
        theta2 *= theta1 # enforces deref < ref threshold 
        self.Refine(theta1)
        self.Derefine(theta2)

    def GetNewErrors(self):
        if self.mesh.GetLastOperation() == self.mesh.REFINE:
            self.rtransforms = self.mesh.GetRefinementTransforms()
            coarse_to_fine = mfem.Table()
            coarse_to_ref_type = mfem.intArray()
            ref_type_to_matrix = mfem.Table()
            ref_type_to_geom = mfem.GeometryTypeArray()
            self.rtransforms.GetCoarseToFineMap(self.mesh, coarse_to_fine, coarse_to_ref_type, ref_type_to_matrix, ref_type_to_geom)
            self.new_errors = mfem.doubleArray(coarse_to_fine.Width())
            tmp = mfem.intArray(1)
            for i in range(coarse_to_fine.Width()):
                self.new_errors[i] = mfem.infinity()
            for i in range(coarse_to_fine.Size()):
                if coarse_to_fine.RowSize(i) == 1:
                    tmp_data = coarse_to_fine.GetRow(i)
                    tmp.Assign(tmp_data)
                    index = tmp[0]
                    self.new_errors[index] = self.errors[i]
        else:
            nel = len(self.errors)
            self.new_errors = mfem.doubleArray(nel)
            for i in range(nel):
                self.new_errors[i] = self.errors[i]
    
    def Derefine(self, theta2):
        threshold = theta2 * np.max(self.errors)
        # if self.mesh.GetLastOperation() == self.mesh.REFINE:
        #     self.rtransforms = self.mesh.GetRefinementTransforms()
        #     coarse_to_fine = mfem.Table()
        #     coarse_to_ref_type = mfem.intArray()
        #     ref_type_to_matrix = mfem.Table()
        #     ref_type_to_geom = mfem.GeometryTypeArray()
        #     self.rtransforms.GetCoarseToFineMap(self.mesh, coarse_to_fine, coarse_to_ref_type, ref_type_to_matrix, ref_type_to_geom)
        #     new_errors = mfem.doubleArray(coarse_to_fine.Width())
        #     tmp = mfem.intArray(1)
        #     for i in range(coarse_to_fine.Width()):
        #         new_errors[i] = mfem.infinity()
        #     for i in range(coarse_to_fine.Size()):
        #         if coarse_to_fine.RowSize(i) == 1:
        #             tmp_data = coarse_to_fine.GetRow(i)
        #             tmp.Assign(tmp_data)
        #             index = tmp[0]
        #             new_errors[index] = self.errors[i]
        # else:
        #     nel = len(self.errors)
        #     new_errors = mfem.doubleArray(nel)
        #     for i in range(nel):
        #         new_errors[i] = self.errors[i]
        self.GetNewErrors()
        self.mesh.DerefineByError(self.new_errors,threshold)
        
        # self.refiner.Reset()
        self.fespace.Update()
        self.x.Update()
        # self.fespace.UpdatesFinished()
        self.a.Update()
        self.b.Update()

class DeRefStationaryProblemBob(DeRefStationaryProblem):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.factor = 5e-5

    def UpdateMesh(self, action):
        emax = np.max(self.errors)
        theta1 = self.factor / max(emax, self.factor)
        self.Refine(theta1)
