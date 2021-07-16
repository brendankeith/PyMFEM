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
from math import atan2, sqrt, sin, cos
import csv

from utils.RandomFunction import RandomFunction

def Exact(pt):
    x = pt[0]
    y = pt[1]
    r = sqrt(x*x + y*y)
    alpha = 2. / 3.

    theta = atan2(y, x)
    if y == 0 and x < 0:
        theta += 2 * np.pi
    if y < 0:
        theta += 2 * np.pi
    if y == 0 and x == -1:
        theta = np.pi
    return r**alpha * sin(alpha * theta)

def ExactGrad(pt):
    x = pt[0]
    y = pt[1]
    alpha = 2. / 3.
    if (x == 0 and y == 0):
        x+=1e-12
        y+=1e-12
    r = sqrt(x*x + y*y)
    theta = atan2(y, x)
    if y == 0 and x < 0:
       theta += 2 * np.pi
    if y < 0:
       theta += 2 * np.pi
    if y == 0 and x == -1:
       theta = np.pi
    rx = x/r
    ry = y/r
    thetax = - y / r**2
    thetay =   x / r**2
    fx = alpha * r**(alpha - 1.) *(rx*sin(alpha*theta) + r*thetax * cos(alpha*theta))
    fy = alpha * r**(alpha - 1.) *(ry*sin(alpha*theta) + r*thetay * cos(alpha*theta))
    return (fx, fy)

##### BK: New class called RandomCoefficient()

class RandomCoefficient(mfem.PyCoefficient):

    def __init__(self, omega=np.pi/2, scale=1.0):
        self.omega = omega
        self.scale = scale
        self.fluctuations = RandomFunction()
        super().__init__()

    def EvalValue(self, pt):
        x = pt[0]
        y = pt[1]
        r = sqrt(x**2 + y**2)
        if r < 1.0:
            return Exact(pt)
        else:
            theta = atan2(y, x)
            if x > 0 and abs(y) < 1e-6:
                theta = 0.0
            elif y < 0:
                theta += 2*np.pi
            s = theta/(2*np.pi - self.omega)
            return Exact(pt) + self.scale * self.fluctuations(s)

#####

class ExactCoefficient(mfem.PyCoefficient):
    def EvalValue(self, pt):
        return Exact(pt)

class ExactGradCoefficient(mfem.VectorPyCoefficient):
    def EvalValue(self, pt):
        return ExactGrad(pt)

class DoubleHpProblem(gym.Env):

    def __init__(self,**kwargs):
        super().__init__()
        #self.BC = mfem.ConstantCoefficient(0.0)
        self.BC = ExactCoefficient()
        #delattr(self, 'RHS')
        self.RHS = mfem.ConstantCoefficient(0.0)
        #self.RHS = RHSCoefficient()
        #self.RHS = mfem.ConstantCoefficient(1.0)
        self.coeff = mfem.ConstantCoefficient(1.0)

        #self.ExactVal = ExactCoefficient()
        #self.ExactGrad = ExactGradCoefficient(2)

        self.optimization_type = kwargs.get('optimization_type','error_threshold')
        self.error_threshold = kwargs.get('error_threshold',1e-3)
        self.dof_threshold = kwargs.get('dof_threshold',1e4)
        self.step_threshold = kwargs.get('step_threshold',10)
        self.refinement_strategy = kwargs.get('refinement_strategy','max')
        self.mode = kwargs.get('mode', 'hp')
        mesh_name = kwargs.get('mesh_name','l-shape.mesh')
        num_unif_ref = kwargs.get('num_unif_ref',1)
        order = kwargs.get('order',1)
        self.average_order = order
        meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', mesh_name))
        mesh = mfem.Mesh(meshfile)
        self.mesh = mesh
        #self.SetBoundaryAttributes()
        mesh.EnsureNCMesh()
        for _ in range(num_unif_ref):
            mesh.UniformRefinement()
        self.dim = mesh.Dimension()
        self.initial_mesh = mesh
        self.order = order
        self.alpha = 0.05
        if self.mode == 'h':
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        elif self.mode == 'hp':
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
    
    def reset(self):
        self.k = 0
        self.mesh = mfem.Mesh(self.initial_mesh)

        ##### BK: Example of use of RandomCoefficient()

        omega = np.pi/2
        scale = 1.0
        self.BC = RandomCoefficient(omega=omega, scale=scale)

        #####

        self.Setup()
        self.AssembleAndSolve()
        self.errors = self.GetLocalErrors()
        obs = self.Errors2Observation(self.errors)
        self.global_error = GlobalError(self.errors)
        self.initial_error_estimate = self.global_error
        self.sum_of_dofs = self.fespace.GetTrueVSize()
        return obs
    
    def step(self, action):
        self.k += 1
        self.UpdateMesh(action)
        self.AssembleAndSolve()
        self.errors = self.GetLocalErrors()
        obs = self.Errors2Observation(self.errors)
        if self.optimization_type == 'error_threshold':
            self.global_error = GlobalError(self.errors)
            num_dofs = self.fespace.GetTrueVSize()
            cost = np.log(1.0 + num_dofs/self.sum_of_dofs)
            self.sum_of_dofs += num_dofs
            if self.global_error < self.alpha * self.initial_error_estimate:
                cost = 0.0
                done = True
            else:
                done = False
                #self.global_error = global_error
            if self.sum_of_dofs > self.dof_threshold or self.k > 50:
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
        info = {}
        
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
        #self.ess_bdr.Print()
        #print(self.ess_bdr[0], self.ess_bdr[1])
        #print(self.mesh.bdr_attributes.Max())
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
        #rho = action['order'] # determine if we want to refine the order this time
        #theta = action['space'].item() #refinement threshold
        theta = action[0].item() #refinement threshold for h
        if self.mode == 'hp':
            rho = action[1].item() 
            rho = theta * rho
        if self.mode == 'hp':
            if self.refinement_strategy == 'quantile':
                h_refine_list, p_refine_list = self.MarkForRefinement(theta, rho)
                self.PrefineQ(p_refine_list)
                self.RefineQ(h_refine_list)
            if self.refinement_strategy == 'max':
                self.Prefine(theta, rho)
                self.Refine(theta)
        if self.mode == 'h':
            self.Refine(theta)

    def MarkForRefinement(self, theta, rho):
        mark_to_h_refine = []
        mark_to_p_refine = []
        element_error_list = []
        for i in range(self.mesh.GetNE()):
            element_error_list.append((i, self.errors[i]))
        element_error_list.sort(key=lambda x:x[1])#, reverse=True)
        cutoff_number_h = math.ceil(theta * (self.mesh.GetNE() - 1))
        cutoff_error_h = element_error_list[cutoff_number_h][1]*(1 - 1e-4)
        cutoff_number_p = math.ceil(rho * (self.mesh.GetNE() - 1))
        cutoff_error_p = element_error_list[cutoff_number_p][1]*(1 - 1e-4)

        if theta == 1:
            cutoff_error_h =  np.max(self.errors)
        if rho == 1:
            cutoff_error_p = cutoff_error_h

        for i in range(self.mesh.GetNE()):
            curr_err = self.errors[i]
            if curr_err > cutoff_error_h:
                mark_to_h_refine.append(i)
        for i in range(self.mesh.GetNE()):
            if i not in mark_to_h_refine:
                curr_err = self.errors[i]
                if curr_err > cutoff_error_p:
                    mark_to_p_refine.append(i)
        return mark_to_h_refine, mark_to_p_refine

    def RefineQ(self, h_refine_list):
        elements_to_h_refine = intArray(h_refine_list)
        self.mesh.GeneralRefinement(elements_to_h_refine)
        self.fespace.Update(False)
        self.x.Update()
        self.x.Assign(0.0)
        self.x.ProjectBdrCoefficient(self.BC, self.ess_bdr)
        # self.fespace.UpdatesFinished()
        self.a.Update()
        self.b.Update()

    def Refine(self, theta):
        #self.refiner.Reset()
        self.refiner.SetTotalErrorFraction(theta)
        self.refiner.Apply(self.mesh)
        self.fespace.Update(False)
        self.x.Update()
        self.x.Assign(0.0)
        self.x.ProjectBdrCoefficient(self.BC, self.ess_bdr)
        # self.fespace.UpdatesFinished()
        self.a.Update()
        self.b.Update()


    def Prefine(self, theta, rho):   
        mark_to_p_refine = []
        threshold = theta * np.max(self.errors)
        for i in range(self.mesh.GetNE()):
            if threshold >= self.errors[i]:
                mark_to_p_refine.append((i, self.errors[i]))
        mark_to_p_refine.sort(key=lambda x:x[1], reverse=True)
        for i in range(len(mark_to_p_refine)):
            if mark_to_p_refine[i][1] > rho * np.max(self.errors):
                current_element = mark_to_p_refine[i][0]
                current_order = self.fespace.GetElementOrder(current_element)
                self.fespace.SetElementOrder(current_element, current_order + 1)
        
        self.fespace.Update(False)
        self.x.Update()
        self.x.Assign(0.0)
        self.x.ProjectBdrCoefficient(self.BC, self.ess_bdr)
        # self.fespace.UpdatesFinished()
        self.a.Update()
        self.b.Update()

    def PrefineQ(self, p_refine_list):
        for j in range(len(p_refine_list)):
            current_order = self.fespace.GetElementOrder(p_refine_list[j])
            self.fespace.SetElementOrder(p_refine_list[j], current_order + 1)
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
        sol_sock.send_text("valuerange 1.0 8.0 \n")

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

    def SetBoundaryAttributes(self):
        for i in range(self.mesh.GetNBE()):
            verts = self.mesh.GetBdrElementVertices(i)
            vert_in_boundary = False
            for j in range(len(verts)):
                temp_arr = mfem.doubleArray(2)
                coords = self.mesh.GetVertex(verts[j])
                temp_arr.Assign(coords)
                if temp_arr[0] == 0.0 and temp_arr[1] == 0.0:
                   vert_in_boundary = True
            if vert_in_boundary:
                self.mesh.SetBdrAttribute(i,2)
        self.mesh.SetAttributes()
    
    def hpDeterministicPolicy(self, thetaDet):
        self.reset()
        #self.rows = []   
        while self.sum_of_dofs < self.dof_threshold:
            self.k += 1
            elements_to_h_refine = []
            elements_to_p_refine = []
            neighbor_table = self.mesh.ElementToElementTable()
            element_error_list = []

            for i in range(self.mesh.GetNE()):
                element_error_list.append((i, self.errors[i]))
            element_error_list.sort(key=lambda x:x[1], reverse=True)
            cutoff_number = math.ceil(thetaDet * (self.mesh.GetNE() - 1))
            cutoff_error = element_error_list[cutoff_number][1]*(1-1e-4)

            for i in range(self.mesh.GetNE()):
                curr_verts = self.mesh.GetElementVertices(i)
                element_touching_corner = False
                curr_error = self.errors[i]

                if self.refinement_strategy == 'max':
                    threshold = thetaDet * np.max(self.errors)
                else:
                    threshold = cutoff_error
                if threshold < curr_error:
                    for j in range(len(curr_verts)):
                        temp_arr = mfem.doubleArray(2)
                        coords = self.mesh.GetVertex(curr_verts[j])
                        temp_arr.Assign(coords)
                        if temp_arr[0] == 0.0 and temp_arr[1] == 0.0:
                            element_touching_corner = True
                    if(element_touching_corner):    
                        elements_to_h_refine.append(i)
                    else:
                        elements_to_p_refine.append(i)
                        # neighbor_row = neighbor_table.GetRow(i)
                        # row_size = neighbor_table.RowSize(i)
                        # neighbor_array = intArray(row_size)
                        # neighbor_array.Assign(neighbor_row)
                        # for l in range(row_size):
                        #     neighbor_order = self.fespace.GetElementOrder(neighbor_array[l])
                        #     if neighbor_order <= self.fespace.GetElementOrder(i):
                        #         elements_to_p_refine.append(neighbor_array[l])

            p_refine_elements = np.unique(elements_to_p_refine).tolist()
            for k in range(len(p_refine_elements)):
                current_element = p_refine_elements[k]
                current_order = self.fespace.GetElementOrder(current_element)
                self.fespace.SetElementOrder(current_element, current_order + 1)
            
            self.fespace.Update(False)
            self.x.Update()
            self.x.Assign(0.0)
            self.x.ProjectBdrCoefficient(self.BC, self.ess_bdr)
            # self.fespace.UpdatesFinished()
            self.a.Update()
            self.b.Update()

            elements_to_h_refine = intArray(elements_to_h_refine)
            self.mesh.GeneralRefinement(elements_to_h_refine)
            
            self.fespace.Update(False)
            self.x.Update()
            self.x.Assign(0.0)
            self.x.ProjectBdrCoefficient(self.BC, self.ess_bdr)
            # self.fespace.UpdatesFinished()
            self.a.Update()
            self.b.Update()

            self.CloseMesh()

            self.AssembleAndSolve()
            self.errors = self.GetLocalErrors()
            self.global_error = GlobalError(self.errors)
            self.sum_of_dofs += self.fespace.GetTrueVSize()
            self.RenderHPmesh()
            #self.rows.append([thetaDet, self.mesh.GetNE(), self.fespace.GetTrueVSize(), self.sum_of_dofs, self.global_error, self.L2error, self.H1error])
        #with open('datafile', 'w') as datafile:
        #    write = csv.writer(datafile)
        #    write.writerow(headers)
        #    write.writerows(rows)    
        print("dofs = ", self.sum_of_dofs)
        print("Global error = ", self.global_error)
    
    def compute_error_values(self):
        self.ExactVal = ExactCoefficient()
        self.ExactGrad = ExactGradCoefficient(2)
        self.L2error = self.x.ComputeL2Error(self.ExactVal) 
        self.H1error = self.x.ComputeH1Error(self.ExactVal,self.ExactGrad)

    def CloseMesh(self, delta_p = 1):
        # Loop through all elements in mesh until the maximum difference in polynomial
        # orders across all edges is no more than delta_p
        neighbor_table = self.mesh.ElementToElementTable()
        while True:
            mesh_closed = True
            elements_to_p_refine = []
            for i in range(self.mesh.GetNE()):
                neighbor_row = neighbor_table.GetRow(i)
                row_size = neighbor_table.RowSize(i)
                neighbor_array = intArray(row_size)
                neighbor_array.Assign(neighbor_row)
                for l in range(row_size):
                    neighbor_order = self.fespace.GetElementOrder(neighbor_array[l])
                    if neighbor_order - self.fespace.GetElementOrder(i) > delta_p:
                        elements_to_p_refine.append(i)
                        mesh_closed = False
            p_refine_elements = np.unique(elements_to_p_refine).tolist()
            for k in range(len(p_refine_elements)):
                current_element = p_refine_elements[k]
                current_order = self.fespace.GetElementOrder(current_element)
                self.fespace.SetElementOrder(current_element, current_order + 1)

            if mesh_closed:
                break

        self.fespace.Update(False)
        self.x.Update()
        self.x.Assign(0.0)
        self.x.ProjectBdrCoefficient(self.BC, self.ess_bdr)
        # self.fespace.UpdatesFinished()
        self.a.Update()
        self.b.Update()


                
                    








    
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


"""
        for i in range(self.mesh.GetNE()):
            curr_err = self.errors[i]
            if curr_err >= cutoff_error_h:
                mark_to_h_refine.append(i)
        for i in range(self.mesh.GetNE()):
            if i not in mark_to_h_refine:
                curr_err = self.errors[i]
                if curr_err >= cutoff_error_p:
                    mark_to_p_refine.append(i)
        return mark_to_h_refine, mark_to_p_refine
"""


"""

"""