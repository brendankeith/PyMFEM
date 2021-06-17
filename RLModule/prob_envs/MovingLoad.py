from math import atan, sqrt, cos, sin
import numpy as np
import mfem.ser as mfem
from mfem.ser import intArray
from utils.StatisticsAndCost import Statistics, GlobalError
from prob_envs.StationaryProblem import DeRefStationaryProblem
import random

def ball(pt, t):
    C = 1.0
    alpha = 0.01
    r0 = 0.25
    r00 = 0.1
    x0 = r0*cos(t) + 0.5
    y0 = r0*sin(t) + 0.5
    x = pt[0] - x0
    y = pt[1] - y0
    r = sqrt(x**2 + y**2)
    return 1/2-atan((r - r00)/alpha)/np.pi

class RHSCoefficient(mfem.PyCoefficientT):
    def EvalValue(self, pt, t):
        return ball(pt, t)

class MovingLoadProblem(DeRefStationaryProblem):
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.error_target = kwargs.get('error_target',1e-3)
        self.penalty_rate = kwargs.get('penalty_rate',1.0)
        self.convex_coeff = kwargs.get('convex_coeff',0.5)
        self.delta_t = 0.05
        delattr(self, 'RHS')
        self.RHS = RHSCoefficient()

    def reset(self):
        self.time = random.uniform(-np.pi, np.pi)
        self.rolling_average = 0.0
        obs = super().reset()
        return obs

    def step(self, action):
        self.k += 1
        self.UpdateMesh(action)
        self.AssembleAndSolve()
        self.errors = self.GetLocalErrors()
        obs = self.Errors2Observation(self.errors)
        global_error = GlobalError(self.errors)
        num_dofs = self.fespace.GetTrueVSize()
        done = False
        if self.optimization_type == 'dof_threshold':
            distance_from_target_error = np.log(global_error/self.error_target)**2
            # distance_from_target_error = np.maximum(np.log(np.abs(global_error-self.error_target)), -10)/10
            if self.k == 1:
                cost = distance_from_target_error
                self.rolling_average = distance_from_target_error
            else:
                cost = (distance_from_target_error - self.rolling_average)/self.k
                self.rolling_average *= (self.k-1)/self.k
                self.rolling_average += distance_from_target_error/self.k
            if num_dofs > self.dof_threshold:
                cost += np.log(num_dofs/self.dof_threshold)/(self.k**self.penalty_rate)
                # cost = np.abs(self.rolling_average)
                if num_dofs > 10*self.dof_threshold:
                    done = True
        elif self.optimization_type == 'error_threshold':
            log_num_dofs = np.log(num_dofs)
            if self.k == 1:
                cost = log_num_dofs
                self.rolling_average = log_num_dofs
            else:
                cost = (log_num_dofs - self.rolling_average)/self.k
                self.rolling_average *= (self.k-1)/self.k
                self.rolling_average += log_num_dofs/self.k
            if global_error > self.error_threshold:
                cost += np.log(global_error/self.error_threshold)/(self.k**self.penalty_rate)
            if num_dofs > self.dof_threshold:
                done = True
        else:
            alpha = self.convex_coeff
            d = self.mesh.Dimension()
            log_num_dofs = np.log(num_dofs)
            log_global_error = np.log(global_error)
            if self.k == 1:
                cost = alpha*log_num_dofs/d + (1-alpha)*log_global_error
                self.rolling_average = alpha*log_num_dofs/d + (1-alpha)*log_global_error
            else:
                # beta = max(1/self.k,0.1)
                # cost = beta*(alpha*log_num_dofs/d + (1-alpha)*log_global_error - self.rolling_average)
                # cost = (alpha*log_num_dofs/d + (1-alpha)*log_global_error - self.rolling_average)/self.k
                # self.rolling_average *= (self.k-1)/self.k
                # self.rolling_average += (alpha*log_num_dofs/d + (1-alpha)*log_global_error)/self.k
                cost = alpha*log_num_dofs/d + (1-alpha)*log_global_error - self.rolling_average
                self.rolling_average = alpha*log_num_dofs/d + (1-alpha)*log_global_error
            if num_dofs > self.dof_threshold:
                cost += 10.0/self.k
                done = True
            # if random.uniform(0,1) < 0.05:
                # done = True
                # self.reset()
        info = {'global_error':global_error, 'num_dofs':num_dofs, 'max_local_errors':np.amax(self.errors)}
        return obs, -cost, done, info

    def AssembleAndSolve(self):
        self.time += self.delta_t
        self.RHS.SetTime(self.time)
        super().AssembleAndSolve()

    def render(self):
        if not hasattr(self, 'sol_sock_soln'):
            self.sol_sock_soln = mfem.socketstream("localhost", 19916)
        self.sol_sock_soln.precision(8)
        self.sol_sock_soln.send_solution(self.mesh, self.x)

    def RenderMesh(self):
        flag = False
        if not hasattr(self, 'sol_sock_mesh'):
            flag = True
            self.sol_sock_mesh = mfem.socketstream("localhost", 19916)
        self.sol_sock_mesh.precision(8)
        zerogf = mfem.GridFunction(self.fespace)
        zerogf.Assign(0.0)
        self.sol_sock_mesh.send_solution(self.mesh, zerogf)
        if flag:
            self.sol_sock_mesh.send_text('keys ARjlmp*******')

    def RenderRHS(self):
        if not hasattr(self, 'sol_sock_RHS'):
            self.sol_sock_RHS = mfem.socketstream("localhost", 19916)
        self.sol_sock_RHS.precision(8)
        rhs_coeff = mfem.GridFunction(self.fespace)
        rhs_coeff.ProjectCoefficient(self.RHS)
        self.sol_sock_RHS.send_solution(self.mesh, rhs_coeff)
        