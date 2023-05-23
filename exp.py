import numpy as np 
import torch
from MACbanditsOverOpt import MAC_Bandit
from BlockMACbandits import BlockMAC_Bandit
import time
import matplotlib
import matplotlib.pyplot as plt
import optuna
from torchmin import minimize_constr
from shared_features import soft_norm_block, norm_block, project_on_ball, rwd_block, gen_A
import pickle
import scipy.optimize
from LinUCB import LinUCB
from numpy.random.mtrand import dirichlet


# dimensionality
d = 3
# size of the sliding window
m = 2
# size L
L = 20
# horizon
T = 1218 # (m + L) * 50
# theta^*
#theta = torch.zeros(d)
#theta[0] = 1.
#theta = torch.rand(d)
#theta /= np.linalg.norm(theta) # n**(1/d)
#print("Theta *: ", theta)
#print(torch.norm(theta))
#theta[0] = - 0.65
#theta[1] = - 0.65
#print("Theta *: ", theta)
#print(torch.norm(theta))
alpha = -3
flag_rising_block = [0 if alpha < 0 else 1]
noise_level = 0.1
delta_A = 1

n_runs = 5
runs_rwds = np.array([])
runs_rwds_greedy = np.array([])
runs_rwds_b = np.array([])
runs_block_rwds = np.array([])
runs_rwds_lin = np.array([])

np.random.seed(0)
torch.manual_seed(0)

theta = torch.rand(d)
theta /= np.linalg.norm(theta)

def constr_norm(block):
    return - np.linalg.norm(block) + 1

for run in range(n_runs):
    #np.random.seed(run)
    #torch.manual_seed(run)

    #theta = torch.rand(d)
    #theta /= np.linalg.norm(theta) # n**(1/d)

    #theta = torch.tensor(dirichlet([1] * d))
    #print(theta)
    #print(np.linalg.norm(theta))
    '''
    if run == 0:
        theta = torch.zeros(d)
        theta[0] = 1.
    elif run == 1:
        theta = torch.zeros(d)
        theta[1] = 1.
    elif run == 2:
        theta = torch.zeros(d)
        theta[2] = 1.
    else:
        theta = torch.rand(d)
        theta /= np.linalg.norm(theta) # n**(1/d)
    '''

    #-------------------------------------------------- MAC bandits over-opt--------------------------------------------
    print("Over-opt MAC bandits algorithm ------------------------------------ ")
    start_1 = time.time()
    torch.autograd.set_detect_anomaly(True)
    if alpha < 0:
        f = 1
    else:
        f = 0
    mab = MAC_Bandit(d, m, L, theta, alpha, delta_A, noise_level, flag_rising=f)
    rwds = np.array([])
    rwd = 0.0
    pre_actions = torch.zeros((m, d))
    obj_plot = np.array([])

    #block = torch.ones((m + L, d)) / torch.sqrt(torch.tensor(d))
    block = torch.rand((m + L, d))
    for index in range(m + L):
        block[index, :] /= np.linalg.norm(block[index, :])
    block.requires_grad = True
    # for each time step
    #print("Tot rounds: ", T // (m + L))
    for tau in range(T // (m + L)):
        #print("round: ", tau)

        # GD
        lr = 0.8  # 0.8  # 1 / (tau + 1)
        block = mab.oracle(block, lr)  #, obj_plot)
        actions = block.detach().clone()
        #print("Actions: ", actions)

        # PLAY BLOCK AND GET REWARD of m and L actions
        # for m actions, get last m actions in previous block
        rwd_temp = 0.0
        rwd_seq = 0.00
        preactions_block = torch.vstack((pre_actions, actions))
        for i in range(m+L):
            A_t = mab.gen_A(preactions_block[i:i + m, :])
            #print(A_t)
            rwd = mab.compute_rwd(actions, i, A_t)
            rwds = np.append(rwds, rwd.item())
            rwd_seq += rwd
            if i >= m:
                mab.update_coeff(rwd, actions, i, A_t)
                rwd_temp += rwd

        #print("rwd temp star: ", rwd_temp)
        #print("rwd block: ", rwd_seq)
        pre_actions = actions[- m:, :]

        mab.update_beta(tau)
        #print("V: ", mab.V_tau)
        #print("U: ", mab.U_tau)
        mab.est_theta_hat()

    #print(rwd_block(theta, actions, m, L, alpha, delta_A))

    print("--- %s seconds ---" % (time.time() - start_1))
    print("Real theta *: ", theta)
    print("Theta hat estimate: ", mab.theta_hat)
    print("Theta hat estimate rescaled: ", mab.theta_hat / torch.norm(mab.theta_hat))
    print("Total reward: ", rwds.sum())


    #-------------------------------------------------- Greedy --------------------------------------------------
    print("Greedy algorithm ---------------------------------------------------")
    start_2 = time.time()
    pre_actions = torch.zeros(m, d)
    rwds_greedy = np.array([])
    rwd_greedy = 0.0
    for tau in range(T):
        # compute A matrix
        A_t = gen_A(pre_actions, alpha, delta_A, d)

        # play actions that maximizes rwd in current time step
        y = torch.matmul(A_t, theta)
        x = theta.clone()
        rwd_greedy = torch.dot(x, y) + noise_level * np.random.randn(1)
        rwds_greedy = np.append(rwds_greedy, rwd_greedy)
        if m <= 1:
            pre_actions = x
        else:
            pre_actions = torch.vstack((pre_actions[1:, :], x.reshape(1, -1)))


    print("--- %s seconds ---" % (time.time() - start_2))
    print("Total reward of Greedy: ", rwds_greedy.sum())

    #-------------------------------------------------- LinUCB --------------------------------------------------
    print("LinUCB algorithm ---------------------------------------------------")
    start_2 = time.time()
    mab_lin = LinUCB(d, theta, delta_A, noise_level)
    pre_actions = torch.zeros((m, d))
    rwds_lin = np.array([])
    rwd_lin = 0.0
    for tau in range(T):
        # compute A matrix
        #A_t = gen_A(pre_actions, alpha, delta_A, d)
        lr = 1
        arm = torch.rand((d))
        arm.requires_grad = True
        arm = mab_lin.oracle(arm, lr)
        action = arm.detach().clone()

        A_t = gen_A(pre_actions, alpha, delta_A, d)
        a_tilde = torch.matmul(A_t, action)
        y = torch.dot(a_tilde, theta)
        rwd_lin = np.random.normal(loc=y, scale=noise_level, size=1)
        rwds_lin = np.append(rwds_lin, rwd_lin.item())

        if m <= 1:
            pre_actions = action
        else:
            pre_actions = torch.vstack((pre_actions[1:, :], action.reshape(1, -1)))

        mab_lin.update_beta(tau)
        mab_lin.est_theta_hat()


    print("--- %s seconds ---" % (time.time() - start_2))
    print("Total reward of LinUCB: ", rwds_lin.sum())

    #---------------------------------------------- BLOCK Mac bandits ----------------------------------------------
    print("Block MAC Bandits algorithm -----------------------------------------")
    start_4 = time.time()
    torch.autograd.set_detect_anomaly(True)
    block_mab= BlockMAC_Bandit(d, m, L, theta, alpha, delta_A, noise_level, flag_rising_block)
    block_rwds = np.array([])
    block_rwd = 0.0
    block_rwd_array = torch.zeros(m+L)
    pre_actions = torch.zeros(m, d)
    preactions_block = torch.zeros(m+L, d)
    A_t = torch.eye(d)
    # for each time step

    #block = torch.ones((m + L, d))
    #for index in range(m + L):
    #    block[index, :] /= np.linalg.norm(block[index, :])
    #block.requires_grad = True
    block = torch.rand((m + L, d))
    for index in range(m + L):
        block[index, :] /= np.linalg.norm(block[index, :])
    block.requires_grad = True

    for tau in range(T // (m + L)):
        # INITIALIZE TENSOR WHICH IS GOING TO STORE THE BLOCK
        # last best block, first time: ones
        #block = torch.ones((m + L, d)) / torch.sqrt(torch.tensor(d))


        # COMPUTE THE BLOCK TO PLAY
        #res = minimize_constr(block_mab.UCB_block, block, constr=dict(fun=soft_norm_block, lb=0., ub=1.), max_iter=300, tol=1e-3, disp=2)
        #actions_outside_ball = res.x
        #print("Actions outside: ", actions_outside_ball)
        #actions = project_on_ball(actions_outside_ball)
        lr = 0.8  #1 / ((tau + 1) )
        block = block_mab.oracle(block,lr)
        actions = block.detach().clone()
        #print("Actions: ", actions)

        # PLAY BLOCK AND GET REWARD of m and L actions
        preactions_block = torch.vstack((pre_actions, actions))
        for i in range(m + L):
            # for m actions, get last m actions in previous block
            #if i < m:
            #    preactions_block = torch.vstack((preactions, actions))
            #    A_t = block_mab.gen_A(preactions_block[i:i+m, :])
            #else:
            #    A_t = block_mab.gen_A(actions[i - m:i, :])
            A_t = block_mab.gen_A(preactions_block[i:i+m, :])
            block_rwd = block_mab.compute_rwd(actions, i, A_t)
            block_rwds = np.append(block_rwds, block_rwd.item())

        block_rwd_array[m:] = torch.tensor(block_rwds[-L:])
        A_block = block_mab.gen_A_block_mL(preactions_block)
        if alpha > 0.0:
            block_mab.update_coeff(block_rwd_array, actions, A_block, 1)
        else:
            block_mab.update_coeff(block_rwd_array, actions, A_block, 0)

        pre_actions = actions[- m:, :]

        block_mab.update_beta(tau)
        #print("V block: ", block_mab.V_tau)
        #print(block_mab.U_tau)
        block_mab.est_theta_hat()

        #block = actions  #.values()

    #print(rwd_block(theta, actions, m, L, alpha, delta_A))


    print("--- %s seconds ---" % (time.time() - start_4))
    print("Real theta *: ", theta)
    print("Theta hat estimate ", block_mab.theta_hat_block)
    print("Average estimate of theta hat on dimension zero: ", torch.mean(block_mab.theta_hat_block, 0))
    print("Average estimate of theta hat on dimension zero rescaled: ", torch.mean(block_mab.theta_hat_block, 0)/ torch.norm(torch.mean(block_mab.theta_hat_block, 0)))
    print("Total reward: ", block_rwds.sum())

    #-------------------------------------------------------------------------------------
    print("Total cumulative reward for all algorithms: ")
    print("Total reward OverOpt: ", rwds.sum())
    print("Total reward Greedy: ", rwds_greedy.sum())
    #print("Total reward Non-OverOpt: ", rwds_b.sum())
    print("Total reward Block: ", block_rwds.sum())
    print("Total reward LinUCB: ", rwds_lin.sum())

    runs_rwds = np.append(runs_rwds, rwds[:1200])
    runs_rwds_greedy = np.append(runs_rwds_greedy, rwds_greedy[:1200])
    #runs_rwds_b = np.append(runs_rwds_b, rwds_b)
    runs_block_rwds = np.append(runs_block_rwds, block_rwds[:1200])
    runs_rwds_lin = np.append(runs_rwds_lin, rwds_lin[:1200])

with open('data_rwds_' + str(alpha) +  '.pkl', 'wb') as f:
    pickle.dump([n_runs, T, alpha, m, runs_rwds, runs_rwds_greedy, runs_rwds_b, runs_block_rwds, runs_rwds_lin], f)

'''
T = 1200
runs_rwds = runs_rwds.reshape(n_runs, T)
runs_rwds_greedy = runs_rwds_greedy.reshape(n_runs, T)
runs_block_rwds = runs_block_rwds.reshape(n_runs, T)
runs_rwds_lin = runs_rwds_lin.reshape(n_runs, T)
for run in range(n_runs):
    for t in range(1, T):
        runs_rwds[run, t] = runs_rwds[run, t] + runs_rwds[run, t - 1]
        runs_rwds_greedy[run, t] = runs_rwds_greedy[run, t] + runs_rwds_greedy[run, t - 1]  # non accumula i reward come mi aspetterei
        #runs_rwds_b[run, t] = runs_rwds_b[run, t] + runs_rwds_b[run, t - 1]
        runs_block_rwds[run, t] = runs_block_rwds[run, t] + runs_block_rwds[run, t - 1]
        runs_rwds_lin[run, t] = runs_rwds_lin[run, t] + runs_rwds_lin[run, t - 1]


for run in range(n_runs):
    avg_rwds = np.mean(runs_rwds.reshape(n_runs, T), axis=0)
    avg_rwds_greedy = np.mean(runs_rwds_greedy.reshape(n_runs, T), axis=0)
    #avg_rwds_b = np.mean(runs_rwds_b.reshape(n_runs, T), axis=0)
    avg_block_rwds = np.mean(runs_block_rwds.reshape(n_runs, T), axis=0)
    avg_rwds_lin = np.mean(runs_rwds_lin.reshape(n_runs, T), axis=0)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
ax = plt.figure(1)
ax = plt.plot(avg_rwds, marker="o", label="OverOpt", color="red")
ax = plt.plot(avg_rwds_greedy, marker="s", label="Greedy", color="green")
#ax = plt.plot(avg_rwds_b, marker="d", label="Non Over Opt", color="blue")
ax = plt.plot(avg_block_rwds, marker="d", label="Block", color="yellow")
ax = plt.plot(avg_rwds_lin, marker="s", label="LinUCB", color="orange")
ax = plt.legend(fontsize='xx-large')
plt.grid()
plt.show()

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
ax = plt.figure(1)
ax = plt.plot(rwds, marker="o",
              label="OverOpt", color="red")
ax = plt.plot(rwds_greedy, marker="s",
              label="Greedy", color="green")
ax = plt.plot(rwds_b, marker="d",
              label="Non Over Opt", color="blue")
ax = plt.plot(block_rwds, marker="d",
              label="Bock", color="yellow")
ax = plt.legend(fontsize='xx-large')
plt.grid()
plt.show()
'''
