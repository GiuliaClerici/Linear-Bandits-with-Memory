import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker
import pickle
import sys
import matplotlib.ticker as mt
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline, BSpline

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



regret_OM = np.array([])
regret_OMBlock = np.array([])
horizons_OM = np.array([])
horizons_OMBlock = np.array([])

n_files = 19
for n_file in range(n_files):
    with open(r"regret_rising/regret_rising_" + str(n_file) + ".pkl", "rb") as input_file:
        n_runs, T_overopt, T_block, alpha, m, regret_OM, regret_OMBlock = pickle.load(input_file)
        #print("right after: ", regret_OM)
        #regret_OM = np.append(regret_OM, totoveropt_regret)
        #regret_OMBlock = np.append(regret_OMBlock, totblock_regret)
        horizons_OM = np.append(horizons_OM, T_overopt)
        horizons_OMBlock = np.append(horizons_OMBlock, T_block)
        #print(n_file)
        #print(horizons_OM.shape)
        #print(regret_OM.shape)
        if n_file == 16:
            regOM_old = np.copy(regret_OM)
            regOMBlock_old = np.copy(regret_OMBlock)
        if n_file > 16:
            #print("here: ", regret_OM)
            regret_OM = np.append(regOM_old, regret_OM)
            regret_OMBlock = np.append(regOMBlock_old, regret_OMBlock)


with open(r"regret_rising_greedy18.pkl", "rb") as input_file:
    n_runs, T_overopt, T_block, alpha, m, tot_rwds_greedy_a, tot_rwds_greedy_b = pickle.load(input_file)

tot_rwds_O3M = np.zeros(n_files)
tot_rwds_OMBlock = np.zeros(n_files)
tot_rwds_O3M = tot_rwds_greedy_a - regret_OM # I get the cumulative rwds of O3M
tot_rwds_OMBlock = tot_rwds_greedy_b - regret_OMBlock # I get the cumulative rwds of OMBlock
regret_OM_new = (tot_rwds_greedy_a / 2) - tot_rwds_O3M
regret_OMBlock_new = (tot_rwds_greedy_b / 2) - tot_rwds_OMBlock

ax = plt.figure(figsize=(10, 8))

'''
y1_f = interp1d(horizons_OM, regret_OM_new, 'linear')
y2_f = interp1d(horizons_OMBlock[:10], regret_OMBlock_new[:10], 'linear')
y1 = y1_f(horizons_OM)
y2 = y2_f(horizons_OMBlock[:10])
ax = plt.plot(horizons_OM, y1,'x-')
ax = plt.plot(horizons_OMBlock[:10], y2,'x-')
'''

'''
spl = make_interp_spline(horizons_OM, regret_OM_new, k=3)
power_smooth = spl(horizons_OM)
ax = plt.plot(horizons_OM, power_smooth)
'''

ax = plt.scatter(horizons_OM, regret_OM, c="#029e73", marker="o") # plt.plot(horizons_OM, regret_OM, "-o")
ax = plt.scatter(horizons_OMBlock, regret_OMBlock, c="#1E88E5", marker="s")
ax = sns.lineplot(x=horizons_OM, y=regret_OM, color="#029e73")
ax = sns.lineplot(x=horizons_OMBlock[:10], y=regret_OMBlock[:10], color="#1E88E5")
# Regression
#ax = sns.regplot(x=horizons_OM, y=regret_OM, color="green", logx=True)
#ax = sns.regplot(x=horizons_OMBlock[:10], y=regret_OMBlock[:10], color="royalblue", logx=True)

ax = sns.set_style("ticks")
#ax = plt.yscale('log')
#ax = plt.xscale('log')

ax = plt.xlabel('Time', fontsize=30)
ax = plt.ylabel('Regret', fontsize=30)
ax = plt.legend(labels=["O3M", "OM-Block"], fontsize=20, title=r"$m=1, \gamma = 2$", title_fontsize=20)  # , "Mid", "Block", "Greedy", "LinUCB"], fontsize=25)
ax = plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax = plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# labels = [r"$T_1^0$", r"$T_1^{'}$", r"$T_2^0$", r"$T_2^{'}$", r"$T_3^0$", r"$T_3^{'}$", r"$T_4^0$", r"$T_5^{0}$"]
# ax = plt.xticks(np.arange(1, 9), labels)

ax = plt.grid()

name_fig = "regret_rising.pdf"
ax = plt.savefig(name_fig, format="pdf", bbox_inches='tight', transparent=False)

plt.show()

print("Tot regret O3M: ", regret_OM)
print("Tot regret OM-Block: ", regret_OMBlock[:10])



