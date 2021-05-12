### Configuration
import sys
sys.path.append("/Users/keith10/Work/PyMFEM/RLModule")
import os
from os.path import expanduser, join
import mfem.ser as mfem
from utils.PolicyModels import PolicyNetwork
from utils.PolicyGradient import REINFORCE
from utils.problem_fem import fem_problem
from utils.toy_problem import toy_problem
import matplotlib.pyplot as plt

ORDER = 1
config = {
    'batch_size'        : 1,
    'max_steps'         : 1,
    'max_episode_num'   : 1000,
}

if __name__ == "__main__":

    meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', 'l-shape.mesh'))
    # meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', 'star.mesh'))
    mesh = mfem.Mesh(meshfile, 1,1)
    mesh.UniformRefinement()

    penalty = 0.0
    env = fem_problem(mesh,ORDER,penalty)
    policy_net = PolicyNetwork(4)
    policy_net.reset()

    REINFORCE = REINFORCE(env, policy_net)
    REINFORCE(**config)
    plt.show()