import numpy as np
import torch
from BlockMACbandits import BlockMAC_Bandit
import time
from shared_features import soft_norm_block, norm_block, project_on_ball, rwd_block, gen_A
import pickle
from numpy.random.mtrand import dirichlet

# dimensionality
d = 3
# size of the sliding window
m = 1
# size L
L_range = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
# horizon
#T = 1250 # (m + L) * 50
# alpha
alpha = 2
flag_rising = [0 if alpha < 0 else 1]
noise_level = 0.1
delta_A = 1
totrwds_overopt = np.array([])
totrwds_block = np.array([])
totoveropt_regret = np.array([])
totblock_regret = np.array([])

n_runs = L_range.flatten().shape[0]
n_repetitions = 5
runs_rwds = np.array([])
runs_block_rwds = np.array([])

theta = torch.zeros(d)
theta[0] = 1.

def constr_norm(block):
    return - np.linalg.norm(block) + 1

for n_rep in range(n_repetitions):
    for run in range(n_runs):
        if run == 17 or run == 18:
            np.random.seed(16)
            torch.manual_seed(16)
        elif n_rep == 0:
            np.random.seed(run)
            torch.manual_seed(run)
        else:
            np.random.seed(n_rep)
            torch.manual_seed(n_rep)

        #if run == 0:
        #    theta = torch.zeros(d)
        #    theta[0] = 1.
        #elif run == 1:
        #    theta = torch.zeros(d)
        #    theta[1] = 1.
        #elif run == 2:
        #    theta = torch.zeros(d)
        #    theta[2] = 1.
        #else:
        #    theta = torch.rand(d)
        #    theta /= np.linalg.norm(theta) # n**(1/d)
        #theta = torch.rand(d)
        #theta /= np.linalg.norm(theta) # n**(1/d)

        theta = torch.rand(d)
        theta /= np.linalg.norm(theta)
        print(theta)

        L = L_range[run]
        #------------------------------------ OverOpt ---------------------------------------------------------------
        T_range_overopt = np.array([16, 81, 256, 625, 1296, 2401, 4096, 6561, 10000, 14641, 20736, 28561, 38416, 50625, 65536, 83521, 104976, 130321, 160000])
        T_overopt = T_range_overopt[run]
        #-------------------------------------------------- Greedy --------------------------------------------------
        print("Greedy algorithm ---------------------------------------------------")
        start_g = time.time()
        pre_actions = torch.zeros(m, d)
        tot_greedy = 0.0
        rwds_greedy = np.array([])
        rwd_greedy = 0.0
        for tau in range(T_overopt):
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

        print("--- %s seconds ---" % (time.time() - start_g))
        print("Total reward of Greedy: ", rwds_greedy.sum())
        tot_greedy = rwds_greedy.sum()

        #-------------------------------------------------- MAC bandits over-opt--------------------------------------------
        print("Over-opt MAC bandits algorithm ------------------------------------ ")
        start_1 = time.time()
        torch.autograd.set_detect_anomaly(True)
        mab = MAC_Bandit(d, m, L, theta, alpha, delta_A, noise_level, flag_rising)
        rwds = np.array([])
        rwd = 0.0
        pre_actions = torch.zeros((m, d))

        block = torch.rand((m + L, d))
        for index in range(m + L):
            block[index, :] /= np.linalg.norm(block[index, :])
        block.requires_grad = True
        # for each time step

        for tau in range(T_overopt // (m + L)):
            block = torch.rand((m + L, d))
            for index in range(m + L):
                block[index, :] /= np.linalg.norm(block[index, :])
            block.requires_grad = True

            # GD
            lr = 0.8  # 0.8  # 1 / (tau + 1)
            block = mab.oracle(block, lr)  #, obj_plot)
            actions = block.detach().clone()

            # PLAY BLOCK AND GET REWARD of m and L actions
            # for m actions, get last m actions in previous block
            rwd_temp = 0.0
            rwd_seq = 0.0
            preactions_block = torch.vstack((pre_actions, actions))
            for i in range(m+L):
                A_t = mab.gen_A(preactions_block[i:i + m, :])
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
            mab.est_theta_hat()
            #print(mab.U_tau)

        #print(rwd_block(theta, actions, m, L, alpha, delta_A))

        print("--- %s seconds ---" % (time.time() - start_1))
        print("Real theta *: ", theta)
        print("Theta hat estimate: ", mab.theta_hat)
        print("Theta hat estimate rescaled: ", mab.theta_hat / torch.norm(mab.theta_hat))
        print("Total reward: ", rwds.sum())
        totrwds_overopt = rwds.sum()
        totoveropt_regret = np.append(totoveropt_regret, tot_greedy - totrwds_overopt)

        #--------------------------------------------------- Block ---------------------------------------------------
        T_range_block = np.array([32, 243, 1024, 3125, 7776, 16807, 32768, 59049, 100000, 161051, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # I'm using the last two just to have an equal number of run and not have to touch the save to file part
        T_block = T_range_block[run]
        # -------------------------------------------------- Greedy --------------------------------------------------
        print("Greedy algorithm ---------------------------------------------------")
        start_g = time.time()
        pre_actions = torch.zeros(m, d)
        tot_greedy = 0.0
        rwds_greedy = np.array([])
        rwd_greedy = 0.0
        for tau in range(T_block):
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

        print("--- %s seconds ---" % (time.time() - start_g))
        print("Total reward of Greedy: ", rwds_greedy.sum())
        tot_greedy = rwds_greedy.sum()
        if T_block == 0:
            tot_greedy = 0.0

        #---------------------------------------------- BLOCK Mac bandits ----------------------------------------------
        print("Block MAC Bandits algorithm -----------------------------------------")
        start_4 = time.time()
        torch.autograd.set_detect_anomaly(True)
        block_mab= BlockMAC_Bandit(d, m, L, theta, alpha, delta_A, noise_level, flag_rising=1)
        block_rwds = np.array([])
        block_rwd = 0.0
        block_rwd_array = torch.zeros(m+L)
        pre_actions = torch.zeros(m, d)
        preactions_block = torch.zeros(m+L, d)
        A_t = torch.eye(d)
        # for each time step

        block = torch.rand((m + L, d))
        for index in range(m + L):
            block[index, :] /= np.linalg.norm(block[index, :])
        block.requires_grad = True

        for tau in range(T_block // (m + L)):


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
            block_mab.est_theta_hat()

        print("--- %s seconds ---" % (time.time() - start_4))
        print("Real theta *: ", theta)
        print("Theta hat estimate ", block_mab.theta_hat_block)
        print("Average estimate of theta hat on dimension zero: ", torch.mean(block_mab.theta_hat_block, 0))
        print("Average estimate of theta hat on dimension zero rescaled: ", torch.mean(block_mab.theta_hat_block, 0)/ torch.norm(torch.mean(block_mab.theta_hat_block, 0)))
        print("Total reward: ", block_rwds.sum())
        totrwds_block = block_rwds.sum()
        totblock_regret = np.append(totblock_regret, tot_greedy - totrwds_block)

        #-------------------------------------------------------------------------------------
        with open('regret_rising_'+ str(run) +'_rep'+ str(n_rep) +'.pkl', 'wb') as f:
            pickle.dump([n_runs, T_overopt, T_block, alpha, m, totoveropt_regret, totblock_regret], f)
