import sys
# sys.path.append("~/Work/PyMFEM/RLModule/examples")
# sys.path.append("~/Work/PyMFEM/")
sys.path.append("..")
import os
from os.path import expanduser, join
import mfem.ser as mfem

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar

from problem_fem import fem_problem

# Constants
ORDER = 1
GAMMA = 0.9

meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', 'l-shape.mesh'))
mesh = mfem.Mesh(meshfile, 1,1)
mesh.UniformRefinement()
# mesh.UniformRefinement()

# penalty = 1e1
penalty = 0.
poisson = fem_problem(mesh,ORDER,penalty)

max_steps = 1

def f(action):
    cost = 0.0
    poisson.reset()
    for steps in range(max_steps):
        _, tmp_cost, _, _ = poisson.step(action)
        # cost = tmp_cost
        cost += GAMMA**steps * tmp_cost
    nel = poisson.mesh.GetNE()
    return cost

res = minimize_scalar(f, bounds=(0, 1), method='bounded')

print("Optimal theta = ", res.x)
print("Optimal cost  = ", f(res.x))

xx = np.linspace(0,1,500)
ff = np.array([f(x) for x in xx])

plt.plot(xx,ff)
plt.xlabel('theta')
plt.ylabel('cost')
plt.show()