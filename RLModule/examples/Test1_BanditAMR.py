### Configuration
import sys

from torch.distributions import distribution
sys.path.append("/Users/keith10/Work/PyMFEM/RLModule")
sys.path.append("..")
from utils.PolicyNetworks import Categorical, TwoParamNormal, TwoParamTruncatedNormal, LinearNormal
from utils.PolicyGradientMethods import REINFORCE
from prob_envs.problem_fem import fem_problem
from prob_envs.toy_problem import toy_problem
import matplotlib.pyplot as plt

prob_config = {
    'mesh_name'         : 'l-shape.mesh',
    'num_unif_ref'      : 1,
    'order'             : 1,
}

dist_choice = 'TwoParamNormal'
# dist_choice = 'TwoParamTruncatedNormal'
# dist_choice = 'LinearNormal'
# dist_choice = 'Categorical'

DRL_config = {
    'batch_size'        : 1,
    'max_steps'         : 1,
    'max_episode_num'   : 1000,
    'learning_rate'     : 5e-2,
    'update_rule'       :'SGD',
    'num_actions'       : 5
}

if __name__ == "__main__":

    env = fem_problem(**prob_config)
    if dist_choice == 'TwoParamNormal':
        policy_net = TwoParamNormal(**DRL_config)
    elif dist_choice == 'TwoParamTruncatedNormal':    
        policy_net = TwoParamTruncatedNormal(**DRL_config)
    elif dist_choice == 'LinearNormal':    
        policy_net = LinearNormal(**DRL_config)
    elif dist_choice == 'Categorical':    
        policy_net = Categorical(**DRL_config)
    else:
        print("Wrong choice of distribution")
        quit()


    policy_net.reset()

    REINFORCE = REINFORCE(env, policy_net)
    REINFORCE(**DRL_config)

    params = REINFORCE.dist_params
    num_plots = len(params[0])

    if dist_choice == 'Categorical':
        fig, ax = plt.subplots(1, sharex=True)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probabilities")
        for i in range(0,num_plots):
            ax.plot(params[:,i],label="Action: {:.2f}".format(i/(num_plots-1)))
        ax.legend()
    else:
        fig, ax = plt.subplots(2, sharex=True)
        ax[0].plot(params[:,0])
        # ax[0].set_ylabel('means')
        ax[0].set_ylabel('sigmoid(mean)')
        ax[1].semilogy(params[:,1])
        ax[1].set_ylabel('std. dev.')



    plt.show()