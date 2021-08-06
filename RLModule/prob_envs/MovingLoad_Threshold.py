from math import atan, sqrt, cos, sin
import numpy as np
import mfem.ser as mfem
from mfem.ser import intArray
from utils.StatisticsAndCost import Statistics, GlobalError
from prob_envs.StationaryProblem import DeRefStationaryProblem
import random
from gym import spaces

def ball(pt, t):
    alpha = 0.01
    r0 = 0.25
    r00 = 0.1
    # r0 = 0.15 + 0.1 * cos(2*t)**2
    # r00 = 0.05 + 0.05 * cos(t)**2
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
        self.strict_dof_threshold = kwargs.get('strict_dof_threshold',10*self.dof_threshold)
        self.delta_t = 0.05
        delattr(self, 'RHS')
        self.RHS = RHSCoefficient()
        # self.observation_space = spaces.Box(low=np.array([-10,0.0,-np.inf,-np.inf,-np.inf,-np.inf]), high=np.array([10.0,1.0,np.inf,np.inf,np.inf,np.inf]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        self.action_space = spaces.Box(low=np.array([-2.0, -1.0]), high=np.array([-1.0, 0.0]))

    def reset(self):
        self.time = random.uniform(-np.pi, np.pi)
        self.rolling_average_cost = 0.0
        self.rolling_average_error = 0.0
        self.rolling_average_dofs = 0.0
        obs = super().reset()
        return obs

    def step(self, action):
        ## set-up
        self.k += 1
        self.UpdateMesh(action)
        self.AssembleAndSolve()
        self.errors = self.GetLocalErrors()
        obs = self.GetObservation()
        global_error = GlobalError(self.errors)
        num_dofs = self.fespace.GetTrueVSize()
        done = False
        log_num_dofs = np.log(num_dofs)
        log_global_error = np.log(global_error)
        log_dof_threshold = np.log(self.dof_threshold)
        log_error_threshold = np.log(self.error_threshold)
        ## compute instantaneous cost
        if self.optimization_type == 'dof_threshold':
            log_error_target = np.log(self.error_target)
            if self.k == 1:
                self.rolling_average_dofs = log_num_dofs
            else:
                self.rolling_average_dofs *= (self.k-1)/self.k
                self.rolling_average_dofs += log_num_dofs/self.k
            pen = 1e2
            # instantaneous_cost = (log_global_error - log_error_target)**2 + pen*max(0,self.rolling_average_dofs-log_dof_threshold)**2
            instantaneous_cost = log_global_error + pen*max(0,self.rolling_average_dofs-log_dof_threshold)
            # instantaneous_cost = log_global_error + (self.rolling_average_dofs-log_dof_threshold)**2
        elif self.optimization_type == 'error_threshold':
            if self.k == 1:
                self.rolling_average_error = log_global_error
            else:
                self.rolling_average_error *= (self.k-1)/self.k
                self.rolling_average_error += log_global_error/self.k
            pen = 1e0
            instantaneous_cost = log_num_dofs + pen*max(0,self.rolling_average_error-log_error_threshold)
        else:
            alpha = self.convex_coeff
            d = self.mesh.Dimension()
            instantaneous_cost = alpha*log_num_dofs/d + (1-alpha)*log_global_error
        ## compute incremental cost
        if self.k == 1:
            cost = instantaneous_cost
            self.rolling_average_cost = cost
        else:
            cost = (instantaneous_cost - self.rolling_average_cost)/self.k
            self.rolling_average_cost *= (self.k-1)/self.k
            self.rolling_average_cost += instantaneous_cost/self.k
        ## exit if taking too long
        # if num_dofs > self.strict_dof_threshold:
        #     cost += 1e3/max(1,self.k-10)**2
        #     done = True
        if random.uniform(0,1) < 0.05:
            done = True
        info = {'global_error':global_error, 'num_dofs':num_dofs, 'max_local_errors':np.amax(self.errors)}
        return obs, -cost, done, info

    def AssembleAndSolve(self):
        self.time += self.delta_t
        self.RHS.SetTime(self.time)
        super().AssembleAndSolve()

    def GetObservation(self):
        stats = Statistics(self.errors)
        # obs = [stats.nels, stats.mean, stats.variance, stats.skewness, stats.kurtosis]
        obs = [self.rolling_average_cost, stats.nels, stats.mean, stats.variance, stats.skewness, stats.kurtosis]
        # if self.optimization_type == 'dof_threshold':
        #     log_dof_threshold = np.log(self.dof_threshold)
        #     rel_constraint = (self.rolling_average_dofs-log_dof_threshold)/log_dof_threshold
        #     obs = [self.rolling_average_cost, rel_constraint, stats.mean, stats.variance, stats.skewness, stats.kurtosis]
        return np.array(obs)

    # def UpdateMesh(self, action):
    #     thresh1 = action[0].item() # refine threshold
    #     thresh2 = action[1].item() # derefine threshold
    #     thresh2 *= thresh1 # enforces deref < ref threshold 
    #     self.Refine(theta1)
    #     self.Derefine(theta1, theta2)

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

    def UpdateMesh(self, action):
        num_elements = self.mesh.GetNE()
        ref_threshold = 10 ** action[0].item() / sqrt(num_elements)
        deref_threshold = 10 ** (action[0].item() + action[1].item()) / sqrt(num_elements)
        self.Refine(ref_threshold)
        self.Derefine(deref_threshold)

    def Refine(self,threshold):
        self.GetNewErrors()
        self.mesh.RefineByError(self.new_errors, threshold, -1, self.nc_limit)
        self.fespace.Update()
        self.x.Update()
        self.a.Update()
        self.b.Update()
    
    def Derefine(self,threshold):
        self.GetNewErrors()
        self.mesh.DerefineByError(self.new_errors, threshold, self.nc_limit)
        self.fespace.Update()
        self.x.Update()
        self.a.Update()
        self.b.Update()