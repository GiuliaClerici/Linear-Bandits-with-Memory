import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker
import pickle
import sys

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#with open(r"plots/plots_definitive/greedy_subopt/data_rwds_greedy_subopt.pkl", "rb") as input_file:
with open(r"greedy_subopt_benchmark_new.pkl", "rb") as input_file:
    n_runs, T, alpha, m, runs_rwds, runs_rwds_greedy, runs_rwds_e2, runs_rwds_lin = pickle.load(input_file)

runs_rwds = runs_rwds.reshape(n_runs, T)
runs_rwds_greedy = runs_rwds_greedy.reshape(n_runs, T)
runs_rwds_e2 = runs_rwds_e2.reshape(n_runs, T)
runs_rwds_lin = runs_rwds_lin.reshape(n_runs, T)


for run in range(n_runs):
    runs_rwds[run, :] = np.cumsum(runs_rwds[run, :])
    runs_rwds_greedy[run, :] = np.cumsum(runs_rwds_greedy[run, :])
    runs_rwds_e2[run, :] = np.cumsum(runs_rwds_e2[run, :])
    runs_rwds_lin[run, :] = np.cumsum(runs_rwds_lin[run, :])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
time_ev = np.arange(0, T, 1)
runs_rwds = runs_rwds.flatten()
runs_rwds_greedy = runs_rwds_greedy.flatten()
runs_rwds_e2 = runs_rwds_e2.flatten()
runs_rwds_lin = runs_rwds_lin.flatten()

'''
ax = plt.figure(figsize=(10, 8))
ax = sns.lineplot(x=time_ev, y=runs_rwds, color="green")
#ax = sns.lineplot(x="Time steps", y="Mid_totrwd", estimator="mean", errorbar="sd", lw=1., data=pd_df, color="C1")
ax = sns.lineplot(x=time_ev, y=runs_rwds_greedy, color="red")
ax = sns.lineplot(x=time_ev, y=runs_rwds_e2, color="purple")
ax = sns.lineplot(x=time_ev, y=runs_rwds_lin, color="gold")
ax = sns.set_style("ticks")
'''

OverOpt_plot = runs_rwds.ravel()
E2_plot = runs_rwds_e2.ravel()
Greedy_plot = runs_rwds_greedy.ravel()
LinUCB_plot = runs_rwds_lin.ravel()

time_ax = np.arange(0, T, 1)
time_ax_plot = np.tile(time_ax, n_runs)

pd_df = {'Time steps': time_ax_plot,
         'OverOpt_totrwd': OverOpt_plot,
         'E2_totrwd': E2_plot,
         'LinUCB_totrwd': LinUCB_plot,
         'Greedy_totrwd': Greedy_plot}



ax = plt.figure(figsize=(10, 8))
ax = sns.lineplot(x="Time steps", y="OverOpt_totrwd", estimator="mean", errorbar="sd", lw=1., data=pd_df, color="#029e73")
ax = sns.lineplot(x="Time steps", y="E2_totrwd", estimator="mean", errorbar="sd", lw=1., data=pd_df, color="#cc78bc")
ax = sns.lineplot(x="Time steps", y="LinUCB_totrwd", estimator="mean", errorbar="sd", lw=1., data=pd_df, color="#ece133")
ax = sns.lineplot(x="Time steps", y="Greedy_totrwd", estimator="mean", errorbar="sd", lw=1., data=pd_df, color="#d55e00")

ax = sns.set_style("ticks")

ax = plt.xlabel('Time', fontsize=30)
ax = plt.ylabel('Cumulative rewards', fontsize=30)
ax = plt.legend(labels=["O3M", r"Policy $\pi_2$", "OFUL", "Oracle Greedy"], fontsize=20, title=r"$ m=2, \gamma = 1$", title_fontsize=20)
ax = plt.yticks(fontsize=17)
ax = plt.xticks(fontsize=17)
ax = plt.title("")
ax = plt.grid()

name_fig = "Performance_greedy_subopt.pdf"
ax = plt.savefig(name_fig, format="pdf", bbox_inches='tight', transparent=False)
plt.show()
