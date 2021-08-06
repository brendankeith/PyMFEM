import numpy as np
import pandas as pd
from pylab import *
import seaborn as sns
from prob_envs.StationaryProblem import StationaryProblem

num_refs = 6
file_name = "./RLModule/out/errors.csv"

prob_config = {
    'problem_type'      : 'exact',
    'mesh_name'         : 'l-shape-benchmark.mesh',
    'num_unif_ref'      : 1,
    'order'             : 1,
}

# prob_config = {
#     'problem_type'      : 'laplace',
#     'mesh_name'         : 'inline-quad.mesh',
#     'num_unif_ref'      : 1,
#     'order'             : 1,
# }

env = StationaryProblem(**prob_config)
env.reset(save_errors=True)
for _ in range(num_refs):
      env.step(0.0)

df = pd.read_csv(file_name)
for i, col in enumerate(df.columns):
      num_dofs = float(col)
      num_non_zeros = len(df[col]) - df[col].isna().sum()
      # df[col] *= df[col]
      df[col] *= num_non_zeros
      # df[col] = - np.log(df[col]) / np.log(num_non_zeros**2)
      # df[col] = - np.log(df[col]) / np.log(num_dofs)
      df.rename(columns={col:str(i)}, inplace=True)

ax = sns.boxenplot(data=df, width=.6, palette="coolwarm")
ax.set_yscale('log')
ax.set_ylabel('Element errors (normalized)')
ax.set_xlabel('Refinement')

plt.show()




# print(np.log(env.global_errors[1][-1]/env.global_errors[1][-5])/np.log(env.global_errors[0][-5]/env.global_errors[0][-1]))
# print(np.log(env.global_errors[2][-1]/env.global_errors[2][-5])/np.log(env.global_errors[0][-5]/env.global_errors[0][-1]))
# print(np.log(env.global_errors[3][-1]/env.global_errors[3][-5])/np.log(env.global_errors[0][-5]/env.global_errors[0][-1]))


# plt.title('Errors')
# plt.loglog(env.global_errors[0], env.global_errors[1], 'r-', label='|| grad(u - u_h)||')
# plt.loglog(env.global_errors[0], env.global_errors[2], 'b-', label='|| y - grad(u_h)||')
# plt.loglog(env.global_errors[0], env.global_errors[3], 'k-', label='|| y_ZZ - grad(u_h)||')
# plt.legend()