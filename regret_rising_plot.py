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
with open(r"regret_rising.pkl", "rb") as input_file:
    n_runs, horizons_OM, horizons_OMBlock, alpha, m, regret_OM, regret_OMBlock, tot_greedy_a, tot_greedy_b = pickle.load(input_file)

ax = plt.figure(figsize=(10, 8))
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



