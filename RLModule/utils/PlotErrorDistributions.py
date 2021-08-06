from math import sqrt
import numpy as np
import pandas as pd
from pylab import *
import seaborn as sns
from prob_envs.StationaryProblem import StationaryProblem

recompute = False
num_refs = 6
# num_refs = 3
# file_name = "./RLModule/out/H1errors.csv"
file_name = "./RLModule/out/H10errors.csv"
# file_name = "./RLModule/out/ZZerrors.csv"

prob_config = {
    'estimator_type'    : 'exact',
    'problem_type'      : 'lshaped',
    'mesh_name'         : 'l-shape-benchmark.mesh',
    'num_unif_ref'      : 1,
    'order'             : 1,
    'error_file_name'   : file_name
}

# prob_config = {
#     'estimator_type'    : 'exact',
#     'problem_type'      : 'sinsin',
#     'mesh_name'         : 'inline-quad.mesh',
#     'num_unif_ref'      : 0,
#     'order'             : 1,
#     'error_file_name'   : file_name
# }

if recompute:
      env = StationaryProblem(**prob_config)
      env.reset(save_errors=True)
      for _ in range(num_refs):
            env.step(0.0)
            print(np.sum(env.errors**2))

df = pd.read_csv(file_name)
for i, col in enumerate(df.columns):
      num_dofs = float(col)
      num_non_zeros = len(df[col]) - df[col].isna().sum()
      df[col] *= df[col]
      df[col] *= num_non_zeros
      # df[col] *= num_non_zeros
      # df[col] = - np.log(df[col]) / np.log(num_non_zeros**2)
      # df[col] = np.log(df[col]) / np.log(num_dofs)
      # df[col] = np.exp(df[col])
      df.rename(columns={col:str(i)}, inplace=True)

means = df.mean().to_numpy()
proxy_df = pd.DataFrame(columns = df.columns)
for i, col in enumerate(proxy_df.columns):
      proxy_df[col] = [means[i]]

ax = sns.stripplot(data=proxy_df, zorder=10,  color="white", linewidth=1, jitter=False, edgecolor="black")
ax = sns.boxenplot(data=df, width=.6, palette="coolwarm")
            # showmeans=True,
            # meanprops={"marker":"o",
            #            "markerfacecolor":"white", 
            #            "markeredgecolor":"black",
            #            "markersize":"10"})
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