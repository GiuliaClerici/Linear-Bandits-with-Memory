import time
import numpy as np
import torch
from sklearn.preprocessing import normalize
from combiner_alpha import BanditCombiner
from shared_features import gen_A, compute_rwd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from MACbanditsOverOpt import MAC_Bandit
import pickle
#from LinUCB import LinUCB

# set seeds
run = 0 # 0
np.random.seed(run)
torch.manual_seed(run)

# dimensionality
d = 3
# size of the sliding window
m_star = 2
alpha_star = -3
# size L
m_range = np.array([m_star])  # range of values for m
alpha_range = np.array([0., -1., -2., alpha_star, -4.]) # np.array([0., 0.5, alpha_star, 1., 2., 3.]) # np.array([-2, 0, 2])  # range of values for alpha
# horizon
T = 1250  # must be divisible by every (m + L) with m in m_range and L in L_range
n_models = len(m_range) * len(alpha_range)
print("Number of models: ", n_models)

theta = torch.rand(d)
theta /= np.linalg.norm(theta)
print(theta)

# theta^*
theta = torch.rand(d)
theta /= np.linalg.norm(theta) # n**(1/d)
#heta = torch.zeros(d)
#theta[0] = 1.0

noise_level = 0.1
lambda_A = 1
delta_regret = 0.01
m_best = 0
alpha_best = 0.0
torch.autograd.set_detect_anomaly(True)
m_max = m_range[-1]
preactions = torch.zeros((m_max, d))  # different m in every roun so need to have
prev_block = [None] * n_models
init_blocks = prev_block = [None] * n_models
#np.random.seed(0)

tau = np.zeros((n_models))
#L_range = np.ones((n_models), dtype=int) * int(10)  # TO BE SET AS THE Ls FOR THE DIFFERENT m
#for i in range(n_models):
#    L_range[i] = m_star * (i + 1)

L_range = np.array([20])

n_runs = 5
runs_rwds = np.array([])
runs_rwds_o = np.array([])
runs_rwds_lin = np.array([])


for run in range(n_runs):

    combiner = BanditCombiner(d, T, m_range, alpha_range, L_range, n_models, noise_level, delta_regret, lambda_A)
    beta_c = np.sqrt(lambda_A) + np.sqrt(2 * np.log(1 / delta_regret) + d * np.log(1 + 1 / (d * lambda_A)))

    # In LBwM list I store all the instances of each sub-bandit
    LBwM = []
    for idx_model in range(n_models):
        i, j = np.unravel_index(idx_model, [len(m_range), len(alpha_range)])
        LBwM.append(MAC_Bandit(d, m_range[i], L_range[i], theta, alpha_range[j], lambda_A, noise_level, flag_rising=0))
        #LBwM.append(BaseAlg(d, m_range[i], L_range[i], theta, alpha_range[j], lambda_A, noise_level, beta_c))

    combiner.target_regret()
    combiner.putative_regret()

    if run == 0:
        for init in range(n_models):
            block = torch.rand(
                (LBwM[init].m + LBwM[init].L, d))  # block = torch.rand((LBwM[idx_arm].m + LBwM[idx_arm].L, d))
            for index in range(LBwM[init].m + LBwM[init].L):
                block[index, :] /= np.linalg.norm(block[index,
                                                  :])  # making sure that the norm of each action inside the block is bounded by 1 (before applying A)
            block.requires_grad = True
            prev_block[init] = block.detach().clone()
            init_blocks[init] = block.detach().clone()
    else:
        for init in range(n_models):
            block = init_blocks[init]
            block.requires_grad = True
            prev_block[init] = block.detach().clone()

        block.requires_grad = True
        prev_block[init] = block.detach().clone()
        init_blocks[init] = block.detach().clone()

    rwds = np.array([])
    print("Tot rounds: ", T // (m_star + L_range[0]))
    for t in range(T // (m_star + L_range[0])):
        print("t: ", t)
        # Compute UCBs and select the base algorithm
        #print("UCBs: ", combiner.u_inds) # (at first all UCBs are ones)
        idx_arm = combiner.UCB_selection()  # select base algorithm to play
        #print("Arm selected: ", idx_arm)  # print index of chosen sub-bandit
        #print(LBwM[idx_arm].m)
        #print(LBwM[idx_arm].L)
        #print(LBwM[idx_arm].alpha)
        # if you need, to get the row and column number to retrieve values in an array of size (m_range) * (alpha_range)
        idx_r, idx_c = np.unravel_index([idx_arm], [len(m_range), len(alpha_range)])

        #Play the base algorithm selected in this round
        #block = torch.rand((LBwM[idx_arm].m + LBwM[idx_arm].L, d))  # block = torch.rand((LBwM[idx_arm].m + LBwM[idx_arm].L, d))
        #for index in range(LBwM[idx_arm].m + LBwM[idx_arm].L):
        #    block[index, :] /= np.linalg.norm(block[index, :])  # making sure that the norm of each action inside the block is bounded by 1 (before applying A)
        #block.requires_grad = True
        lr = 0.8  #1 / (tau[idx_arm] + 1) # lr_set[idx_arm] # selecting the proper learning rate for the sub-bandit
        #tau[idx_arm] += 1  # counter for keeping track of how many times the sub-bandit has been played
        block = prev_block[idx_arm]
        block.requires_grad = True
        block = LBwM[idx_arm].oracle(block, lr)
        actions = block.detach().clone()
        #print("Actions: ", actions)
        prev_block[idx_arm] = block.detach().clone()

        # PLAY BLOCK AND GET REWARD of m and L actions
        rwd = 0.0
        rwd_seq = 0.0
        avg_rwd = 0.0
        #preactions_block = torch.vstack((preactions, actions))
        for i in range(LBwM[idx_arm].m + LBwM[idx_arm].L):
            A_t = gen_A(preactions[-m_star:, :], alpha_star, delta=1, dim=d)
            rwd = compute_rwd(actions, i, A_t, theta_star=theta, noise=noise_level)
            if run == 4:
                print(rwd)
            rwds = np.append(rwds, rwd.item())
            if i >= LBwM[idx_arm].m:
                rwd_seq += rwd
                LBwM[idx_arm].update_coeff(rwd, actions, i, A_t)
            preactions = torch.vstack((preactions[1:, :], actions[i, :]))

        avg_rwd = rwd_seq / (LBwM[idx_arm].L)
        #avg_rwd = (avg_rwd + 1) / 2
        #avg_rwd = (avg_rwd - (float(1 +LBwM[idx_arm].m) ** (-alpha_range[idx_c]))) / (1 + (float(1 + LBwM[idx_arm].m)**(-alpha_range[idx_c])))  # normalizing between [0,1]
        #print("Avg rwd: ", avg_rwd)
        #preactions = actions[- LBwM[idx_arm].m:, :]  # select only last m actions that I use

        s = combiner.t_plays[idx_arm]
        LBwM[idx_arm].update_beta(s)
        LBwM[idx_arm].est_theta_hat()

        #Update the Bandit Combiner
        combiner.update_stats_and_discard(idx_arm, rwd_seq)  # update also UCBs
        #print("I set: ", combiner.i_set)
        if np.sum(combiner.i_set) <= 1.0:
            idx = int(np.where(combiner.i_set == 1)[0])
            m_best, alpha_best = idx % len(m_range), idx % len(alpha_range)
            #print("The model selection has selected one base algorithm, where its m* and alpha* are respectively ", m_best,
            #      "and", alpha_best)
        #print("Avgs: ", combiner.mean_rwds)
        #print("Number of plays for each arm: ", tau)
        #print("Theta hat: ", LBwM[idx_arm].theta_hat)
        #print("Putative regrets: ", combiner.put_regs)
        print("Total rwd combiner: ", rwds.sum())
    runs_rwds = np.append(runs_rwds, rwds)

    '''
    # -------------------------------------------------- MAC bandits over-opt--------------------------------------------
    print("Over-opt MAC bandits algorithm ------------------------------------ ")
    start_1 = time.time()
    torch.autograd.set_detect_anomaly(True)
    L_o = L_range[0]
    mab_o = MAC_Bandit(d, m_star, L_o, theta, alpha_star, lambda_A, noise_level, flag_rising=0)
    #print(mab_o.theta_hat.shape)
    rwds_o = np.array([])
    rwd_o = 0.0
    pre_actions = torch.zeros((m_star, d))
    obj_plot = np.array([])

    # block = torch.ones((m + L, d)) / torch.sqrt(torch.tensor(d))
    #block_o = torch.rand((m_star + L_o, d))
    #for index in range(m_star + L_o):
    #    block_o[index, :] /= np.linalg.norm(block_o[index, :])
    #block_o.requires_grad = True

    block_o = torch.tensor([[0.5060, 0.3613, 0.7832], [0.5377, 0.6223, 0.5688], [0.6844, 0.7116, 0.1585], [0.5310, 0.6178, 0.5800], [0.2619, 0.9287, 0.2623], [0.3047, 0.2829, 0.9095]])
    block_o.requires_grad = True
    # for each time step
    print("Tot rounds: ", T // (m_star + L_range[0]))
    for tau in range(T // (m_star + L_range[0])):
        print("round: ", tau)

        # GD
        lr = 0.8  # 0.8  # 1 / (tau + 1)
        block_o = mab_o.oracle(block_o, lr)  # , obj_plot)
        actions = block_o.detach().clone()
        #print("Actions: ", actions)

        # PLAY BLOCK AND GET REWARD of m and L actions
        # for m actions, get last m actions in previous block
        rwd_o_temp = 0.0
        rwd_o_seq = 0.00
        rwd_o = 0.0
        preactions_block = torch.vstack((pre_actions, actions))
        for i in range(m_star + L_o):
            A_t = mab_o.gen_A(preactions_block[i:i + m_star, :])
            #print(A_t)
            rwd_o = mab_o.compute_rwd(actions, i, A_t)
            rwds_o = np.append(rwds_o, rwd_o.item())
            rwd_o_seq += rwd_o
            if i >= m_star:
                mab_o.update_coeff(rwd_o, actions, i, A_t)
                rwd_o_temp += rwd_o

        # print("rwd temp star: ", rwd_temp)
        # print("rwd block: ", rwd_seq)
        pre_actions = actions[- m_star:, :]

        mab_o.update_beta(tau)
        #print("V: ", mab_o.V_tau)
        #print("U: ", mab_o.U_tau)
        mab_o.est_theta_hat()

    # print(rwd_block(theta, actions, m, L, alpha, delta_A))

    print("--- %s seconds ---" % (time.time() - start_1))
    print("Real theta *: ", theta)
    print("Theta hat estimate: ", mab_o.theta_hat)
    print("Theta hat estimate rescaled: ", mab_o.theta_hat / torch.norm(mab_o.theta_hat))
    print("Total reward: ", rwds_o.sum())

    runs_rwds_o = np.append(runs_rwds_o, rwds_o)

    # -------------------------------------------------------------------------------------
    print("LinUCB algorithm ---------------------------------------------------")
    mab_lin = LinUCB(d, theta, lambda_A, noise_level)
    pre_actions = torch.zeros((m_star, d))
    rwds_lin = np.array([])
    rwd_lin = 0.0
    arm = torch.rand((d))
    arm.requires_grad = True
    for tau in range(T):
        # compute A matrix
        # A_t = gen_A(pre_actions, alpha, delta_A, d)
        lr = 1
        arm = mab_lin.oracle(arm, lr)
        action = arm.detach().clone()

        A_t = gen_A(pre_actions, alpha_star, lambda_A, d)
        a_tilde = torch.matmul(A_t, action)
        y = torch.dot(a_tilde, theta)
        rwd_lin = np.random.normal(loc=y, scale=noise_level, size=1)
        rwds_lin = np.append(rwds_lin, rwd_lin.item())

        if m_star <= 1:
            pre_actions = action
        else:
            pre_actions = torch.vstack((pre_actions[1:, :], action.reshape(1, -1)))

        mab_lin.update_beta(tau)
        mab_lin.est_theta_hat()
    runs_rwds_lin = np.append(runs_rwds_lin, rwds_lin)
    # -------------------------------------------------------------------------------------
    '''

with open('data_combiner_' + str(alpha_star) +  '_LinUCB.pkl', 'wb') as f:
    pickle.dump([n_runs, T, alpha_star, m_star, runs_rwds], f)  # runs_rwds_o, runs_rwds_lin

'''
runs_rwds = runs_rwds.reshape(n_runs, T)
#runs_rwds_o = runs_rwds_o.reshape(n_runs, T)
#runs_rwds_lin = runs_rwds_lin.reshape(n_runs, T)
print(runs_rwds)
#print(runs_rwds_o)
#print(runs_rwds_lin)


for run in range(n_runs):
    runs_rwds[run, :] = np.cumsum(runs_rwds[run, :])
    #runs_rwds_o[run, :] = np.cumsum(runs_rwds_o[run, :])
    #runs_rwds_lin[run, :] = np.cumsum(runs_rwds_lin[run, :])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
time_ev = np.arange(0, T, 1)
runs_rwds = runs_rwds.flatten()
#runs_rwds_o = runs_rwds_o.flatten()
#runs_rwds_lin = runs_rwds_lin.flatten()

#OverOpt_plot = runs_rwds_o.ravel()
Combiner_plot = runs_rwds.ravel()
#LinUCB_plot = runs_rwds_lin.ravel()

time_ax = np.arange(0, T, 1)
time_ax_plot = np.tile(time_ax, n_runs)

#pd_df = {'Time steps': time_ax_plot,
#         'OverOpt_totrwd': OverOpt_plot,
#         'Comb_totrwd': Combiner_plot,
#         'LinUCB_totrwd': LinUCB_plot}
pd_df = {'Time steps': time_ax_plot,
         'Comb_totrwd': Combiner_plot}

ax = plt.figure(figsize=(10, 8))
#ax = sns.lineplot(x="Time steps", y="OverOpt_totrwd", estimator="mean", errorbar="sd", lw=1., data=pd_df, color="green")
ax = sns.lineplot(x="Time steps", y="Comb_totrwd", estimator="mean", errorbar="sd", lw=1., data=pd_df, color="orange")
#ax = sns.lineplot(x="Time steps", y="LinUCB_totrwd", estimator="mean", errorbar="sd", lw=1., data=pd_df, color="yellow")
ax = sns.set_style("ticks")
#ax = plt.yscale('log')

ax = plt.xlabel('Time') #, weight='bold')
ax = plt.ylabel('Cumulative rewards') #, weight='bold')
ax = plt.legend(labels=["Combiner"], fontsize=20)  # "OverOpt", "Combiner", "OFUL"], fontsize=20)
ax = plt.title("Model selection on alpha")
ax = plt.grid()

name_fig = "plots/Performance_model_selection_alpha_OFUL.pdf"
ax = plt.savefig(name_fig, format="pdf", bbox_inches='tight', transparent=False)
plt.show()
'''