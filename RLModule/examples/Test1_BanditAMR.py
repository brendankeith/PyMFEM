### Configuration
import sys
sys.path.append("/Users/keith10/Work/PyMFEM/RLModule")
from utils.PolicyNetworks import TwoParamNormal, TwoParamTruncatedNormal
from utils.PolicyGradientMethods import REINFORCE
from prob_envs.problem_fem import fem_problem
from prob_envs.toy_problem import toy_problem
import matplotlib.pyplot as plt

prob_config = {
    'mesh_name'         : 'l-shape.mesh',
    'num_unif_ref'      : 1,
    'order'             : 1,
}
DRL_config = {
    'batch_size'        : 1,
    'max_steps'         : 1,
    'max_episode_num'   : 1000,
    'learning_rate'     : 1e-1,
    # 'learning_rate'     : 1e-2,
}

if __name__ == "__main__":

    env = fem_problem(**prob_config)
    policy_net = TwoParamNormal(**DRL_config)
    # policy_net = TwoParamTruncatedNormal(**config)
    policy_net.reset()

    REINFORCE = REINFORCE(env, policy_net)
    REINFORCE(**DRL_config)

    means = REINFORCE.dist_params[0]
    sds = REINFORCE.dist_params[1]
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(means)
    # ax[0].set_ylabel('means')
    ax[0].set_ylabel('sigmoid(mean)')
    ax[1].semilogy(sds)
    ax[1].set_ylabel('std. dev.')

    plt.show()