import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker
import pickle
import sys

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

OverOpt_list = []
Mid_list = []
Block_list = []
Greedy_list = []
LinUCB_list = []

with open(r"data_rwds_exp.pkl", "rb") as input_file:
    n_runs, T, alpha, m, runs_rwds, runs_rwds_greedy, runs_rwds_b, runs_block_rwds, runs_rwds_lin = pickle.load(input_file)

with open("data_combiner_alpha.pkl", "rb") as input_file3:
    n_runs, T, alpha_star, m_star, runs_rwds_a = pickle.load(input_file3)

with open("data_combiner_m.pkl", "rb") as input_file2:
    n_runs, T, alpha_star, m_star, runs_rwds_m = pickle.load(input_file2)



T = 1200
# Reshape where every row corresponds to one run
runs_rwds = runs_rwds.reshape(n_runs, T)
# runs_rwds_b = runs_rwds_b.reshape(n_runs, T)
runs_block_rwds = runs_block_rwds.reshape(n_runs, T)
runs_rwds_greedy = runs_rwds_greedy.reshape(n_runs, T)
runs_rwds_lin = runs_rwds_lin.reshape(n_runs, T)
runs_rwds_a = runs_rwds_a.reshape(n_runs, 1232)
runs_rwds_a = runs_rwds_a[:, :1200]
runs_rwds_m = runs_rwds_m.reshape(n_runs, T)

# Compute cumulative rewards
for run in range(n_runs):
    runs_rwds[run, :] = np.cumsum(runs_rwds[run, :])
    runs_rwds_greedy[run, :] = np.cumsum(runs_rwds_greedy[run, :])
    runs_block_rwds[run, :] = np.cumsum(runs_block_rwds[run, :])
    runs_rwds_lin[run, :] = np.cumsum(runs_rwds_lin[run, :])
    runs_rwds_a[run, :] = np.cumsum(runs_rwds_a[run, :])
    runs_rwds_m[run, :] = np.cumsum(runs_rwds_m[run, :])

time_ax = np.arange(0, T, 1)
time_ax_plot = np.tile(time_ax, 5)

OverOpt_plot = runs_rwds.ravel()
Block_plot = runs_block_rwds.ravel()
Greedy_plot = runs_rwds_greedy.ravel()
LinUCB_plot = runs_rwds_lin.ravel()
Comba_plot = runs_rwds_a.ravel()
Combm_plot = runs_rwds_m.ravel()

pd_df = {'Time steps': time_ax_plot,
         'OverOpt_totrwd': OverOpt_plot,
         'Block_totrwd': Block_plot,
         'Greedy_totrwd': Greedy_plot,
         'LinUCB_totrwd': LinUCB_plot,
         'Comba_totrwd': Comba_plot,
         'Combm_totrwd': Combm_plot}

ax = plt.figure(figsize=(10, 8))
ax = sns.lineplot(x="Time steps", y="OverOpt_totrwd", estimator="mean", errorbar="sd", lw=1., data=pd_df,
                  color="#029e73")
# ax = sns.lineplot(x="Time steps", y="Mid_totrwd", estimator="mean", errorbar="sd", lw=1., data=pd_df, color="C1")
#ax = sns.lineplot(x="Time steps", y="Block_totrwd", estimator="mean", errorbar="sd", lw=1., data=pd_df,
#                  color="#0173b2")
ax = sns.lineplot(x="Time steps", y="Comba_totrwd", estimator="mean", errorbar="sd", lw=1., data=pd_df,
                  color="#fc0fc0")
ax = sns.lineplot(x="Time steps", y="Combm_totrwd", estimator="mean", errorbar="sd", lw=1., data=pd_df,
                  color="#1E88E5")
ax = sns.lineplot(x="Time steps", y="LinUCB_totrwd", estimator="mean", errorbar="sd", lw=1., data=pd_df,
                  color="#ece133")
ax = sns.lineplot(x="Time steps", y="Greedy_totrwd", estimator="mean", errorbar="sd", lw=1., data=pd_df,
                  color="#d55e00")
ax = sns.set_style("ticks")

# y = 10 * np.arange(0, 9)
# labels = np.arange(0, 17, 2)
# ax = plt.yticks(100 * labels, labels)

# ax = plt.tick_params(labelsize=24)
# plt.xlabel("t", fontsize=15)
# plt.ylabel(r"$\sum_t ~ X_t$", fontsize=15)

# sns.set_palette("colorblind")
ax = plt.xlabel('Time', fontsize=30)
ax = plt.ylabel('Cumulative rewards', fontsize=30)
ax = plt.legend(labels=["O3M", r"Combiner $\gamma$", "Combiner m", "OFUL", "Oracle Greedy"], fontsize=20,
                title=r"$ m=2, \gamma =" + str(-alpha) + "$", title_fontsize=20)
ax = plt.yticks(fontsize=17)
ax = plt.xticks(fontsize=17)
#ax = plt.xscale('log')
#ax = plt.yscale('log')
# ax = plt.title("Experiments for m = " + str(m) + " and alpha = " + str(-alpha))

name_fig = "plots/Performance.pdf"
ax = plt.grid()

ax = plt.savefig(name_fig, format="pdf", bbox_inches='tight', transparent=False)

plt.show()
