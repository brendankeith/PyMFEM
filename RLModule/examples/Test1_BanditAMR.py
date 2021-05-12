### Configuration
import sys
sys.path.append("/Users/keith10/Work/PyMFEM/RLModule")
import os
from os.path import expanduser, join
import mfem.ser as mfem
from utils.PolicyNetworks import TwoParamNormal, TwoParamTruncatedNormal
from utils.PolicyGradientMethods import REINFORCE
from prob_envs.problem_fem import fem_problem
from prob_envs.toy_problem import toy_problem
import matplotlib.pyplot as plt



ORDER = 1
config = {
    'batch_size'        : 1,
    'max_steps'         : 1,
    'max_episode_num'   : 1000,
    'learning_rate'     : 1e-1,
    # 'learning_rate'     : 1e-2,
}

if __name__ == "__main__":

    meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', 'l-shape.mesh'))
    # meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', 'star.mesh'))
    mesh = mfem.Mesh(meshfile, 1,1)
    mesh.UniformRefinement()

    penalty = 0.0
    env = fem_problem(mesh,ORDER,penalty)
    policy_net = TwoParamNormal(**config)
    # policy_net = TwoParamTruncatedNormal(**config)
    policy_net.reset()

    REINFORCE = REINFORCE(env, policy_net)
    REINFORCE(**config)

    means = REINFORCE.dist_params[0]
    sds = REINFORCE.dist_params[1]
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(means)
    # ax[0].set_ylabel('means')
    ax[1].set_ylabel('sigmoid(mean)')
    ax[1].semilogy(sds)
    ax[1].set_ylabel('std. dev.')

    plt.show()