import numpy as np
import pandas as pd
from pylab import *
import seaborn as sns

file_name = "./RLModule/hp_sample_data/Deterministic_Policy_with_marking.csv"
file_name_2 = "./RLModule/hp_sample_data/No_Marking_Deterministic_data.csv"
file_name_3 = "./RLModule/hp_sample_data/Random_Angle_Training.csv"
file_name_4 = './RLModule/hp_sample_data/Exact_Random_Angle_Training.csv'
file_name_5 = './RLModule/hp_sample_data/Exact_Deterministic_Data.csv'
file_name_6 = './RLModule/hp_sample_data/Exact_Deterministic_Data_No_Flagging.csv'
file_name_7 = './RLModule/hp_sample_data/Exact_Random_Angle_Averages.csv'
file_name_8 = './RLModule/hp_sample_data/Exact_Pacman_Random_Angle_Training.csv'
file_name_9 = './RLModule/hp_sample_data/Exact_Pacman_Deterministic_No_Flagging.csv'
file_name_10 = './RLModule/hp_sample_data/Exact_Pacman_Deterministic_Flagging.csv'
file_name_11 = './RLModule/hp_sample_data/L_Shaped_Random_Angle_Training_No_Knowledge.csv'
file_name_12 = './RLModule/hp_sample_data/Pacman_Random_Angle_Training_No_Knowledge.csv'
file_name_13 = './RLModule/hp_sample_data/Random_L_Shape_Random_Angle_Exact_No_Angle_Observation.csv'
file_name_14 = './RLModule/hp_sample_data/Random_Pacman_Random_Angle_Exact_No_Angle_Observation.csv'
df1 = pd.read_csv(file_name)['Cost'].to_numpy()
df2 = pd.read_csv(file_name_2)['Cost'].to_numpy()
df3 = pd.read_csv(file_name_3)['Cost'].to_numpy()
df4 = pd.read_csv(file_name_4)['Cost'].to_numpy()
df5 = pd.read_csv(file_name_5)['Cost'].to_numpy()
df6 = pd.read_csv(file_name_6)['Cost'].to_numpy()
df7 = pd.read_csv(file_name_7)['Cost'].to_numpy()
df8 = pd.read_csv(file_name_8)['Cost'].to_numpy()
df9 = pd.read_csv(file_name_9)['Cost'].to_numpy()
df10 = pd.read_csv(file_name_10)['Cost'].to_numpy()
df11 = pd.read_csv(file_name_11)['Cost'].to_numpy()
df12 = pd.read_csv(file_name_12)['Cost'].to_numpy()
df13 = pd.read_csv(file_name_13)['Cost'].to_numpy()
df14 = pd.read_csv(file_name_14)['Cost'].to_numpy()
#cdf = pd.DataFrame([df1['Cost'].to_numpy(), df2['Cost'].to_numpy(), df3['Cost'].to_numpy()], columns=['Deterministic Policy with Marking', 'Deterministic Policy without Marking',
                    #'Learned Policy'])
#df1 = pd.DataFrame(data = df1, columns=['Deterministic Policy with Flagging'])
#df2 = pd.DataFrame(data = df2, columns=['Deterministic Policy without Flagging'])
#df3 = pd.DataFrame(data = df3, columns=['Learned Policy'])
df4 = pd.DataFrame(data = df4, columns=['Exact BCs Learned Policy'])
df5 = pd.DataFrame(data = df5, columns=['Exact BCs Deterministic with Flagging'])
df6 = pd.DataFrame(data = df6, columns=['Exact BCs Deterministic without Flagging'])
df7 = pd.DataFrame(data = df7, columns=['Exact BCs Learned Policy on Random Angles'])
df8 = pd.DataFrame(data = df8, columns=['Exact BCs Fixed Pacman Learned Policy'])
df9 = pd.DataFrame(data = df9, columns=['Exact BCs Fixed Pacman Deterministic without Flagging'])
df10 = pd.DataFrame(data = df10, columns=['Exact BCs Fixed Pacman with Flagging'])
df11 = pd.DataFrame(data = df11, columns=['Exact BCs Learned Policy (No Angle Info)'])
df12 = pd.DataFrame(data = df12, columns=['Exact BCs Fixed Pacman Learned Policy (No Angle Info)'])
df13 = pd.DataFrame(data = df13, columns=['Exact BCs Random L Shaped Learned Policy (No Angle Info)'])
df14 = pd.DataFrame(data = df14, columns=['Exact BCs Random Pacman Learned Policy (No Angle Info)'])
cdf = pd.concat([df8, df12, df14, df10, df9])
#cdf = pd.concat([df4, df7, df11, df13, df5, df6])#, df8, df12, df10, df9])
#print(df1)
#print(df2)
#print(df3)
#print(cdf)
mdf = pd.melt(cdf)

# for i, col in enumerate(df.columns):
#       num_dofs = float(col)
#       num_non_zeros = len(df[col]) - df[col].isna().sum()
#       df[col] *= df[col]
#       df[col] *= num_non_zeros
#       df[col] = - np.log(df[col]) / np.log(num_non_zeros**2)
#       # df[col] = - np.log(df[col]) / np.log(num_dofs)

ax = sns.boxenplot(x='variable', y = 'value', data=mdf, width=.4, palette="coolwarm")
#ax = sns.boxenplot(x=df['Cost'])
ax.set_ylabel('Episode Cost')
ax.set_xlabel('Policy Type')
ax.set_title('Comparison of policies on benchmark problem using the max refinement strategy')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)
#ax.axhline(-6.772727935088968)
#ax.axhline(-6.424731126623283, color = 'red')

#ax.set_xlabel('Episode Cost')

plt.show()

