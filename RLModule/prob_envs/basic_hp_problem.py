from threading import current_thread
from mfem._ser.gridfunc import GridFunction, ProlongToMaxOrder
import os
from os.path import expanduser, join
import gym
from gym import spaces
import numpy as np
import mfem.ser as mfem
from mfem.ser import intArray
from utils.StatisticsAndCost import Statistics, GlobalError
import math

class HpProblem(gym.Env):

    def __init__(self,**kwargs):
        super().__init__()
        self.BC = mfem.ConstantCoefficient(0.0)
        self.RHS = mfem.ConstantCoefficient(1.0)
        self.coeff = mfem.ConstantCoefficient(1.0)

        self.optimization_type = kwargs.get('optimization_type','error_threshold')
        self.error_threshold = kwargs.get('error_threshold',1e-3)
        self.dof_threshold = kwargs.get('dof_threshold',1e4)
        self.step_threshold = kwargs.get('step_threshold',10)
        mesh_name = kwargs.get('mesh_name','l-shape.mesh')
        num_unif_ref = kwargs.get('num_unif_ref',1)
        order = kwargs.get('order',1)
        self.average_order = order
        meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', mesh_name))
        mesh = mfem.Mesh(meshfile)
        mesh.EnsureNCMesh()
        for _ in range(num_unif_ref):
            mesh.UniformRefinement()
        self.dim = mesh.Dimension()
        self.initial_mesh = mesh
        self.order = order
#        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Dict({"space" : spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32), 
                                         "order" : spaces.Discrete(2)})
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        self.THETA = 0.7
    
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
        if self.optimization_type == 'error_threshold':
            global_error = GlobalError(self.errors)
            num_dofs = self.fespace.GetTrueVSize()
            cost = np.log(1.0 + num_dofs/self.sum_of_dofs) #- self.ComputeAverageOrder()
            self.sum_of_dofs += num_dofs
            if self.global_error < self.error_threshold:
                done = True
            else:
                done = False
                self.global_error = global_error
            if self.sum_of_dofs > self.dof_threshold or self.k > 50:
                cost = 10.0
                done = True
                #print("Optimal Policy reached termination condition.")
        elif self.optimization_type == 'dof_threshold':
            self.sum_of_dofs += self.fespace.GetTrueVSize()
            if self.sum_of_dofs > self.dof_threshold:
                cost = 0.0
                done = True
            else:
                global_error = GlobalError(self.errors)
                cost = np.log(global_error/self.global_error) #- self.ComputeAverageOrder()
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
        info = {}

        #print("Optimal Policy reached termination condition.")
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
        average_order = self.ComputeAverageOrder()
        obs = [stats.nels, stats.mean, stats.variance, stats.skewness, stats.kurtosis, average_order]
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
        self.estimator.Reset()
        mfem_errors = self.estimator.GetLocalErrors()
        errors = np.array([mfem_errors[i] for i in range(self.mesh.GetNE())])
        return errors

    def RenderMesh(self):
        sol_sock = mfem.socketstream("localhost", 19916)
        sol_sock.precision(8)
        zerogf = mfem.GridFunction(self.fespace)
        zerogf.Assign(0.0)
        prolonged = ProlongToMaxOrder(self.x)
        sol_sock.send_solution(self.mesh, prolonged)
        title = "step " + str(self.k)
        sol_sock.send_text('keys ARjlmp*******' + " window_title '" + title)

    def UpdateMesh(self, action):
        rho = action['order'] # determine if we want to refine the order this time
        theta = action['space'].item() #refinement threshold
        #theta = action[0].item() #refinement threshold for h
        #rho = action[1].item() 
        #theta = self.THETA
        if theta < 0. :
          theta = 0.
        if theta > 0.999 :
          theta = 0.999 
        #if rho < 0. :
        #  rho = 0.
        #if rho > 0.999 :
        #  rho = 0.999 
        if rho == 1:
            self.Prefine(theta)
        #self.Prefine(theta, rho)
        self.Refine(theta)

    def Refine(self, theta):
        # self.refiner.Reset()
        self.refiner.SetTotalErrorFraction(theta)
        self.refiner.Apply(self.mesh)
        self.fespace.Update(False)
        self.x.Update()
        self.x.Assign(0.0)
        self.x.ProjectBdrCoefficient(self.BC, self.ess_bdr)
        # self.fespace.UpdatesFinished()
        self.a.Update()
        self.b.Update()

    def Prefine(self, theta):
        #mark_to_p_refine = []
        threshold = theta * np.max(self.errors)
        for i in range(self.mesh.GetNE()):
            if threshold >= self.errors[i]:
                #mark_to_p_refine.append((i, self.errors[i]))
                current_order = self.fespace.GetElementOrder(i)
                self.fespace.SetElementOrder(i, current_order + 1)
        """
        number_elements_to_refine = math.floor(rho * len(mark_to_p_refine))
        mark_to_p_refine.sort(key=lambda x:x[1], reverse=True)
        for i in range(0, number_elements_to_refine):
            current_element = mark_to_p_refine[i][0]
            current_order = self.fespace.GetElementOrder(current_element)
            self.fespace.SetElementOrder(current_element, current_order + 1)
        """
        self.fespace.Update(False)
        self.x.Update()
        self.x.Assign(0.0)
        self.x.ProjectBdrCoefficient(self.BC, self.ess_bdr)
        # self.fespace.UpdatesFinished()
        self.a.Update()
        self.b.Update()

    def RenderHPmesh(self):
        ordersfec = mfem.L2_FECollection(0, self.dim)
        ordersfes = mfem.FiniteElementSpace(self.mesh, ordersfec)
        orders = mfem.GridFunction(ordersfes)
        for i in range(0, self.mesh.GetNE()):
            elem_dofs = 0
            elem_dofs = ordersfes.GetElementDofs(i)
            orders[elem_dofs[0]] = self.fespace.GetElementOrder(i)
        sol_sock = mfem.socketstream("localhost", 19916)
        sol_sock.precision(8)
        sol_sock.send_solution(self.mesh, orders)
        title = "step " + str(self.k)
        sol_sock.send_text('keys ARjlmp*******' + " window_title '" + title)

    def ComputeAverageOrder(self):
        total_els = self.fespace.GetNE()
        for i in range(0, total_els):
            element_and_order_dict = {}
            ord = self.fespace.GetElementOrder(i)
            if ord in element_and_order_dict:
                val = element_and_order_dict[ord]
                element_and_order_dict[ord] = val+1
            else:
                element_and_order_dict[ord] = 1
        running_average = 0
        for key in element_and_order_dict:
            running_average = running_average + key * element_and_order_dict[key] / total_els
        return running_average



    
"""
L2_FECollection ordersfec(0,dim);
FiniteElementSpace ordersfes(mesh,&ordersfec);
GridFunction orders(&ordersfes);
for (int i = 0;i<mesh->GetNE(); i++)
{
  Array<int> elem_dofs;
  ordersfes.GetElementDofs(i,elem_dofs);
  MFEM_VERIFY(elem_dofs.Size() == 1,"Wrong elem_dofs size");
  orders[elem_dofs[0]] = fespace->GetElementOrder(i);
}
socketstream orders_sock(vishost, visport);
orders_sock.precision(8);
orders_sock << "solution\n" << *mesh << orders << flush;
sol_sock.send_solution(self.mesh, prolonged)
"""