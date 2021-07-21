import numpy as np
import pandas as pd
from pylab import *
import seaborn as sns

file_name = "./RLModule/out/errors.csv"
df = pd.read_csv(file_name)
for i, col in enumerate(df.columns):
      num_dofs = float(col)
      num_non_zeros = len(df[col]) - df[col].isna().sum()
      df[col] *= df[col]
      df[col] *= num_non_zeros
      df[col] = - np.log(df[col]) / np.log(num_non_zeros**2)
      # df[col] = - np.log(df[col]) / np.log(num_dofs)

ax = sns.boxenplot(data=df, width=.6, palette="coolwarm")
ax.set_ylabel('Element errors (normalized)')
ax.set_xlabel('Refinement')

plt.show()

