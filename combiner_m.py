import time
import numpy as np
import torch
from sklearn.preprocessing import normalize

class BanditCombiner:

    def __init__(self, d, T, m_range, m_star, alpha_range, L_range, n_models, noise_level, delta_regret, lambda_A, alpha_regret):
        # if array of size n_model then we interpret it as if it's the raveled arry of size (m rows, alpha columns)
        self.d, self.n_models, self.T, self.m_range, self.alpha_range, self.L_range = d, n_models, T, m_range, alpha_range, L_range
        self.m_star = m_star
        self.noise_level = noise_level
        self.t_plays = np.zeros((n_models))
        self.u_inds = np.ones((n_models)) * 10  # should be enough if set to 2
        self.mean_rwds = np.zeros((n_models))
        self.i_set = np.ones((n_models))
        self.sum_dev = np.zeros((n_models))
        self.delta_regret = delta_regret
        self.put_regs = np.zeros((n_models))
        self.tar_regs = np.zeros((n_models))
        self.lambda_regret = 1.0  # regret bound
        self.alpha_regret = alpha_regret
        self.c_coeff_tot = np.ones((n_models))
        self.c_coeff = np.zeros((n_models))
        self.eta = np.ones((n_models)) * (np.sqrt(n_models) / (self.T**(self.alpha_regret)))
        self.lambda_A = lambda_A
        self.beta = np.repeat(np.sqrt(self.lambda_A) + np.sqrt(
            2 * np.log(1 / self.delta_regret) + d * np.log(1 + 1 / (d * self.lambda_A))), n_models )

    def update_beta(self, idx):
        idx_m, idx_alpha = np.unravel_index([idx], [len(self.m_range), int(self.n_models / len(self.m_range))])
        self.beta[idx] = np.sqrt(self.lambda_A) + np.sqrt(
            2 * np.log(1 / self.delta_regret) + self.d * np.log(1 + (((self.t_plays[idx] + 1) + 1) * self.L_range[idx_m]) / (self.d * self.lambda_A)))  # I need self.t_plays[idx] += 1 but I'm updating it in update_stats_and_discard


    def putative_regret(self):
        # compute putative regrets with changing T(i, t)
        for i in range(self.n_models):
            idx_m, _ = np.unravel_index([i], [len(self.m_range), int(self.n_models / len(self.m_range))])
            op_norm = 1 #np.array([1 if -self.alpha_range[idx_alpha] > 0.0 else float((1 + self.m_range[idx_m]))**(-self.alpha_range[idx_alpha])])
            self.c_coeff[i] = 4 * self.L_range[idx_m] * op_norm * \
                              np.sqrt((self.d**(1/3)) * (self.m_range[idx_m]**(2/3)) * np.log(1 + (self.t_plays[i]**(4/3) * (self.m_range[idx_m]**(2/3)) *  op_norm**2 )/( 2 * self.d**(5/3) * self.lambda_A))) \
                              * (np.sqrt(self.lambda_A) + np.sqrt(np.log(1/self.delta_regret) + self.d * np.log(1 + (self.t_plays[i] * op_norm**2 )/(self.d * self.lambda_A))))  # * (self.m_range[idx_m] + self.L_range[idx_m])
            # I added the **2 to go from cumulative w.r.t. T and block to cumulative w.r.t. T and averaged on block
            self.put_regs[i] = self.c_coeff[i] * (self.t_plays[i] ** self.alpha_regret)

    def target_regret(self):
        for i in range(self.n_models):
            # to compute
            idx_m, _ = np.unravel_index([i], [len(self.m_range), int(self.n_models / len(self.m_range))])
            #self.eta[i] = 1 / np.sqrt(self.T)
            sum_etas = np.sum(self.eta) - self.eta[i]
            op_norm = 1  # np.array([1 if -self.alpha_range[idx_alpha] > 0.0 else float((1 + self.m_range[idx_m]))**(-self.alpha_range[idx_alpha])])
            print(self.m_range[idx_m])
            self.c_coeff_tot[i] = 4 * self.L_range[idx_m] * op_norm * \
                              np.sqrt((self.d**(1/3)) * (self.m_range[idx_m]**(2/3)) * np.log(1 + (self.T**(4/3) * (self.m_range[idx_m]**(2/3)) *  op_norm**2 )/( 2 * self.d**(5/3) * self.lambda_A))) \
                              * (np.sqrt(self.lambda_A) + np.sqrt(np.log(1/self.delta_regret) + self.d * np.log(1 + (self.T * op_norm**2 )/(self.d * self.lambda_A))))# removed (self.m_range[idx_m] + self.L_range[idx_m])
            print("Coeff: ", self.c_coeff_tot[i])
            second_term = ((((1 - self.alpha_regret) ** ((1 - self.alpha_regret)/ self.alpha_regret))
                           * ((1 + self.alpha_regret) ** (1 / self.alpha_regret)) )/( self.alpha_regret ** ((1 - self.alpha_regret)/self.alpha_regret))) \
                           * (self.c_coeff_tot[i] ** (1/self.alpha_regret)) * self.T * (self.eta[i] ** ((1 - self.alpha_regret)/self.alpha_regret))
            #print(second_term)
            #print("second term: ", ((((1 - self.alpha_regret) ** ((1 - self.alpha_regret)/ self.alpha_regret))
            #               * ((1 + self.alpha_regret) ** (1 / self.alpha_regret)) )/( self.alpha_regret ** ((1 - self.alpha_regret)/self.alpha_regret))))
            self.tar_regs[i] = (self.c_coeff_tot[i] * (self.T ** self.alpha_regret)) + second_term + 1152 * op_norm**2 * np.log((self.T**3) * self.n_models/self.delta_regret) * self.T * self.eta[i] + sum_etas
            #print("Third: ", 288 * np.log((self.T**3) * self.n_models/self.delta_regret) * self.T * self.eta[i] + sum_etas)
            #print(288 * np.log((self.T**3) * self.n_models/self.delta_regret) * self.T * self.eta[i] )
            #print(sum_etas)
            print("target reg: ", self.tar_regs[i])


    def compute_UCBs(self):
        for i in range(self.n_models):
            self.u_inds[i] = self.mean_rwds[i] + min(1, ((self.put_regs[i] + np.sqrt(8 * np.log(((self.T ** 3) * self.n_models) / self.delta_regret) * self.t_plays[i]) )/ self.t_plays[i])) #- self.tar_regs[i]/ self.T


    def UCB_selection(self):
        #best_idx = np.argmax(self.u_inds)
        choices = np.array([0, 1])
        x = np.random.choice(choices, 1, p=[0.9, 0.1])
        if x == 0:
            best_idx = np.argmax(self.u_inds)
        else:
            best_idx = np.random.randint(3)
        return best_idx


    def update_stats_and_discard(self, best_idx, rwd):
        # update number of plays
        self.t_plays[best_idx] += 1
        # update mean reward
        self.mean_rwds[best_idx] = ((self.mean_rwds[best_idx] * (self.t_plays[best_idx] - 1)) + rwd)/ self.t_plays[best_idx]
        self.putative_regret()
        # update UCB index
        op_norm = 1 # rotting case
        self.u_inds[best_idx] = self.mean_rwds[best_idx] + min( op_norm**2 , ((self.put_regs[best_idx] + 4 * op_norm**2 * np.sqrt(
                                 2 * np.log(((self.T ** 3) * self.n_models) / self.delta_regret) * self.t_plays[best_idx])) / self.t_plays[best_idx])) - \
                                self.tar_regs[best_idx] / self.T
        #print("min term: ", ((self.put_regs[best_idx] + np.sqrt(8 * np.log(((self.T ** 3) * self.n_models) / self.delta_regret) * self.t_plays[best_idx])) / self.t_plays[best_idx]))
        #print("Term in UCB PUT REG: ", ((self.put_regs[best_idx] + np.sqrt(
        #                        32 * np.log(((self.T ** 3) * self.n_models) / self.delta_regret) * self.t_plays[best_idx])) / self.t_plays[best_idx]))
        # update sum dev to be used later on to discard
        self.sum_dev[best_idx] += self.mean_rwds[best_idx] - rwd
        # compute threshold to evaluate if discarded or not
        thresh = self.put_regs[best_idx] + (12 * op_norm**2 * np.sqrt(np.log(((self.T**3) * self.n_models) / self.delta_regret)
                                                        * self.t_plays[best_idx]))
        # evaluate if discarded or not
        if self.sum_dev[best_idx] >= thresh:
            self.i_set[best_idx] = 0
