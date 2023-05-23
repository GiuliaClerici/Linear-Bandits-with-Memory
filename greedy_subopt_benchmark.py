import numpy as np
import torch
from MACbanditsOverOpt import MAC_Bandit
import time
import matplotlib
import matplotlib.pyplot as plt
import pickle
from LinUCB import LinUCB
from numpy.random.mtrand import dirichlet
from scipy.optimize import minimize


# dimensionality
d = 2
# size of the sliding window
m = 2
# size L
L = 10
# horizon
T = 1200 # (m + L) * 50
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
alpha = 1
noise_level = 0.3
delta_A = 1

n_runs = 10
runs_rwds = np.array([])
runs_rwds_greedy = np.array([])
runs_rwds_e2 = np.array([])
runs_rwds_lin = np.array([])

def constr_norm(block):
    return - np.linalg.norm(block) + 1

def gen_A_Greedysubopt(pre_actions, alpha, delta_A, d):
    m, _ = pre_actions.shape
    init_mat = torch.zeros((d, d))
    init_mat[0, 0] = 1
    e_1 = torch.zeros((d))
    e_1[0] = 1
    e_2 = torch.zeros((d))
    e_2[1] = 1
    mat_1 = torch.zeros((d, d))
    mat_2 = torch.zeros((d, d))
    for i in range(m):
        mat_1 += torch.matmul(pre_actions[i, :], e_1) * torch.outer(e_1, e_1)
        mat_2 += torch.matmul(pre_actions[i, :], e_2) * torch.outer(e_2, e_2)
    res = init_mat + mat_1 + mat_2
    return res

def gen_A_Greedysubopt2(pre_actions, alpha, delta_A, d):
    m, _ = pre_actions.shape
    init_mat = torch.zeros((d, d))
    init_mat[0, 0] = 1
    mat = init_mat + torch.matmul(pre_actions.T, pre_actions)
    return mat

for run in range(n_runs):
    np.random.seed(run)
    torch.manual_seed(run)

    theta = torch.zeros((d))
    epsilon = 0.01
    theta[0] = np.sqrt(epsilon)
    theta[1] = np.sqrt(1 - epsilon)
    print("Theta*: ", theta)

    #-------------------------------------------------- Greedy --------------------------------------------------
    print("Greedy algorithm ---------------------------------------------------")
    start_2 = time.time()
    pre_actions = torch.zeros(m, d)
    rwds_greedy = np.array([])
    rwd_greedy = 0.0
    A_t = torch.zeros((d, d))
    x = torch.from_numpy(np.ones(d)).float()
    for tau in range(T):
        # compute A matrix
        A_t = gen_A_Greedysubopt2(pre_actions, alpha, delta_A, d)
        print("A_t: ", A_t)
        y = torch.matmul(A_t, theta)

        # maximize rwd for Greedy
        def const_real(t):
            return np.linalg.norm(t) - 1

        constarnt = [{'type': 'eq', 'fun': const_real}]

        def function(x):
            return - x @ np.array(y)

        res = minimize(function, x, constraints=constarnt)
        x = torch.from_numpy(res.x).float()

        #x = torch.zeros((d))
        #x[0] = 1  # e_1
        print("Action played by Greedy", x)
        rwd_greedy = torch.dot(x, y) + noise_level * np.random.randn(1)
        print(rwd_greedy)
        if tau == 1:
            e_1 = torch.zeros((d))
            e_1[0] = 1
            print("rwd e_1 in t=1: ", torch.dot(e_1, y) + noise_level * np.random.randn(1))
        rwds_greedy = np.append(rwds_greedy, rwd_greedy)
        if m <= 1:
            pre_actions = x
        else:
            pre_actions = torch.vstack((pre_actions[1:, :], x.reshape(1, -1)))


    print("--- %s seconds ---" % (time.time() - start_2))
    print("Total reward of Greedy: ", rwds_greedy.sum())

    #    #-------------------------------------------------- MAC bandits over-opt--------------------------------------------
    print("Over-opt MAC bandits algorithm ------------------------------------ ")
    start_1 = time.time()
    torch.autograd.set_detect_anomaly(True)
    mab = MAC_Bandit(d, m, L, theta, alpha, delta_A, noise_level, flag_rising=2)
    rwds = np.array([])
    rwd = 0.0
    pre_actions = torch.zeros((m, d))

    #block = torch.rand((m + L, d))
    #for index in range(m + L):
    #    block[index, :] /= np.linalg.norm(block[index, :])
    #block.requires_grad = True

    if run == 0:
        block = torch.rand((m + L, d))
        for index in range(m + L):
            block[index, :] /= np.linalg.norm(block[index, :])
        block.requires_grad = True
        init_block = block.detach().clone()
    else:
        block = init_block
        block.requires_grad = True

    print("Init: ", block)
    # for each time step
    print("Tot rounds: ", T // (m + L))
    for tau in range(T // (m + L)):
        print("round: ", tau)

        # GD
        lr = 0.8
        block = mab.oracle_Greedysubopt(block, lr)
        actions = block.detach().clone()
        print("Actions: ", actions)

        # PLAY BLOCK AND GET REWARD of m and L actions
        # for m actions, get last m actions in previous block
        rwd_temp = 0.0
        rwd_seq = 0.0
        preactions_block = torch.vstack((pre_actions, actions))
        for i in range(m+L):
            A_t = mab.gen_A_Greedysubopt2(preactions_block[i:i + m, :])
            print(A_t)
            rwd = mab.compute_rwd(actions, i, A_t)
            rwds = np.append(rwds, rwd.item())
            if i >= m:
                mab.update_coeff(rwd, actions, i, A_t)
                rwd_seq += rwd
        print("Sum of rwds of block: ", rwd_seq)
        pre_actions = actions[- m:, :]

        mab.update_beta(tau)
        print("V: ", mab.V_tau)
        print("U: ", mab.U_tau)
        mab.est_theta_hat()

    #print(rwd_block(theta, actions, m, L, alpha, delta_A))

    print("--- %s seconds ---" % (time.time() - start_1))
    print("Real theta *: ", theta)
    print("Theta hat estimate: ", mab.theta_hat)
    print("Theta hat estimate rescaled: ", mab.theta_hat / torch.norm(mab.theta_hat))
    print("Total reward: ", rwds.sum())

    # -------------------------------------------------- --------------------------------------------
    print("e_2 algorithm ------------------------------------ ")
    start_1 = time.time()
    rwds_e2 = np.array([])
    rwd_e2 = 0.0
    pre_actions = torch.zeros(m, d)
    A_t = torch.zeros((d, d))
    for tau in range(T):
        # compute A matrix
        A_t = gen_A_Greedysubopt(pre_actions, alpha, delta_A, d)

        # play actions that maximizes rwd in current time step
        y = torch.matmul(A_t, theta)
        print(y)
        x = torch.zeros(d)
        x[1] = 1  # e_2
        print(x)
        rwd_e2 = torch.dot(x, y) + noise_level * np.random.randn(1)
        print(rwd_e2)
        rwds_e2 = np.append(rwds_e2, rwd_e2)
        if m <= 1:
            pre_actions = x
        else:
            pre_actions = torch.vstack((pre_actions[1:, :], x.reshape(1, -1)))

    print("--- %s seconds ---" % (time.time() - start_1))
    print("Real theta *: ", theta)
    print("Total reward e_2: ", rwds_e2.sum())


    #-------------------------------------------------- LinUCB --------------------------------------------------
    print("LinUCB algorithm ---------------------------------------------------")
    start_2 = time.time()
    mab_lin = LinUCB(d, theta, delta_A, noise_level)
    pre_actions = torch.zeros((m, d))
    rwds_lin = np.array([])
    rwd_lin = 0.0
    arm = torch.rand((d))
    arm.requires_grad = True
    for tau in range(T):
        # compute A matrix
        #A_t = gen_A(pre_actions, alpha, delta_A, d)
        lr = 1
        arm = mab_lin.oracle(arm, lr)
        action = arm.detach().clone()

        A_t = gen_A_Greedysubopt2(pre_actions, alpha, delta_A, d)
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


    print("---------------------------------------------")
    print("Total cumulative reward for all algorithms: ")
    print("Total reward Greedy: ", rwds_greedy.sum())
    print("Total reward OverOpt: ", rwds.sum())
    print("Total reward of e_2: ", rwds_e2.sum())
    print("Total reward of LinUCB: ", rwds_lin.sum())

    runs_rwds = np.append(runs_rwds, rwds)
    runs_rwds_greedy = np.append(runs_rwds_greedy, rwds_greedy)
    runs_rwds_e2 = np.append(runs_rwds_e2, rwds_e2)
    runs_rwds_lin = np.append(runs_rwds_lin, rwds_lin)

with open('greedy_subopt_benchmark.pkl', 'wb') as f:
    pickle.dump([n_runs, T, alpha, m, runs_rwds, runs_rwds_greedy, runs_rwds_e2, runs_rwds_lin], f)
