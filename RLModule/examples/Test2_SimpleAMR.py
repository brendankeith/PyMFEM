### Configuration
import sys

from torch.distributions import distribution
sys.path.append("./RLModule")
sys.path.append("..")
from utils.PolicyNetworks import LinearNormal, LinearCritic
from utils.PolicyGradientMethods import ActorCritic, REINFORCE
from prob_envs.problem_fem import fem_problem
import matplotlib.pyplot as plt

prob_config = {
    'mesh_name'         : 'l-shape.mesh',
    'num_unif_ref'      : 1,
    'order'             : 1,
}

DRL_config = {
    'batch_size'        : 1,
    'max_steps'         : 3,
    'max_episodes'   : 1000,
    'learning_rate'     : 1e-3,
    'learning_rate_critic'     : 1e-3,
}

if __name__ == "__main__":

    env = fem_problem(**prob_config)
    policy_net = LinearNormal(**DRL_config)
    value_net = LinearCritic(**DRL_config)
    policy_net.reset()
    value_net.reset()

    ActorCritic = ActorCritic(env, policy_net, value_net)
    ActorCritic(**DRL_config)

    # params = ActorCritic.dist_params
    # num_plots = len(params[0])

    # fig, ax = plt.subplots({2, sharex=True)
    # ax[0].plot(params[:,0])
    # # ax[0].set_ylabel('means')
    # ax[0].set_ylabel('sigmoid(mean)')
    # ax[1].semilogy(params[:,1])
    # ax[1].set_ylabel('std. }dev.')



    plt.show()