import numpy as np
import torch

class BlockMAC_Bandit:

    def __init__(self, d, m, L, theta, alpha, delta_mat, noise_level, flag_rising):
        self.d, self.m, self.L = d, m, L
        self.alpha = alpha
        self.noise_level = noise_level
        # theta - simple case: have one dimension equal to 1
        self.theta = theta
        self.theta_block = torch.zeros((self.m + self.L, self.d))
        self.theta_block += self.theta
        # initial matrix A
        self.delta = delta_mat
        self.A = self.delta * torch.eye(d)
        self.A_block = self.delta * torch.eye(d*(m+L))
        self.theta_hat = torch.ones(d)
        # theta hat
        # theta_hat = torch.zeros(d, requires_grad=True)
        if flag_rising == 2:
            theta_hat = torch.tensor([1./5., 4.9/5.])
            self.theta_hat_block = theta_hat.tile((self.m + self.L))
            self.theta_hat_block = self.theta_hat_block.reshape((self.m + self.L), self.d)
        else:
            self.theta_hat_block = torch.zeros((self.m + self.L), self.d)
        self.theta_hat_block += self.theta #torch.ones((m+L, d))  # torch.matmul(torch.inverse(torch.eye(d)), )  # MODIFY NORM
        # beta of confidence bonus
        self.lambda_ = 1.
        self.delta_beta = 0.01
        self.beta = np.sqrt((self.m + self.L)) * np.sqrt(self.lambda_) + np.sqrt(2 * np.log(1 / self.delta_beta) + (self.d * (self.m + self.L)) * np.log(1 + 1 / (self.d * self.lambda_)))
        # block
        self.block = torch.zeros((m + L, d))
        # V_tau matrix
        self.V_tau = torch.eye(d*(m+L)) * 1
        # U_tau matrix
        self.U_tau = torch.zeros(d*(m+L)) * 1

    def gen_A(self, past_actions):
        mat = self.delta * torch.eye(self.d) + torch.matmul(past_actions.T, past_actions)
        #mat = torch.linalg.matrix_power(torch.inverse(mat), self.alpha)
        mat_tmp = mat.detach().numpy()
        eigenv_tmp, U_tmp = np.linalg.eigh(mat_tmp)
        U = torch.tensor(U_tmp)
        eigenv = torch.tensor(eigenv_tmp)
        res = torch.matmul(torch.matmul(U, torch.diag(torch.pow(eigenv, self.alpha))), U.T)
        return res

    def gen_A_Greedysubopt(self, past_actions):
        m, d = past_actions.shape
        init_mat = torch.zeros((d, d))
        init_mat[0, 0] = 1
        e_1 = torch.zeros((d))
        e_1[0] = 1
        e_2 = torch.zeros((d))
        e_2[1] = 1
        mat_1 = torch.zeros((d, d))
        mat_2 = torch.zeros((d, d))
        for i in range(m):
            mat_1 += torch.matmul(past_actions[i, :], e_1) * torch.outer(e_1, e_1)
            mat_2 += torch.matmul(past_actions[i, :], e_2) * torch.outer(e_2, e_2)
        res = init_mat + mat_1 + mat_2
        return res

    def gen_A_Greedysubopt2(self, pre_actions):
        m, _ = pre_actions.shape
        init_mat = torch.zeros((self.d, self.d))
        init_mat[0, 0] = 1
        mat = init_mat + torch.matmul(pre_actions.T, pre_actions)
        return mat

    def gen_A_block_L(self, block):
        mat_block = torch.zeros((self.d * self.L, self.d * self.L))
        for i in range(self.L):
            mat = self.gen_A(block[i:self.m + i, :])
            mat_block[(self.d * i): (self.d * i) + self.d, (self.d * i): (self.d * i) + self.d] = mat
        return mat_block

    def gen_A_block_mL(self, block):
        mat_block = torch.zeros((self.d * (self.m + self.L), self.d * (self.m + self.L)))
        for i in range(self.m):
            mat_block[(self.d * i): (self.d * i) + self.d,
                      (self.d * i): (self.d * i) + self.d] = self.delta * torch.eye(self.d)
        for i in range(self.L):
            mat = self.gen_A(block[i:self.m + i, :])
            mat_block[(self.m * self.d) + (self.d * i): (self.m * self.d) + (self.d * i) + self.d,
                      (self.m * self.d) + (self.d * i): (self.m * self.d) + (self.d * i) + self.d] = mat
        return mat_block

    def gen_A_block_mL_Greedysubopt(self, block):
        mat_block = torch.zeros((self.d * (self.m + self.L), self.d * (self.m + self.L)))
        for i in range(self.m):
            mat_block[(self.d * i): (self.d * i) + self.d,
                      (self.d * i): (self.d * i) + self.d] = self.delta * torch.eye(self.d)
        for i in range(self.L):
            mat = self.gen_A_Greedysubopt2(block[i:self.m + i, :])
            mat_block[(self.m * self.d) + (self.d * i): (self.m * self.d) + (self.d * i) + self.d,
                      (self.m * self.d) + (self.d * i): (self.m * self.d) + (self.d * i) + self.d] = mat
        return mat_block

    # OBJECTIVE FUNCTION FOR GRADIENT DESCENT WITH GOAL OF FINDING BEST BLOCK USING UCBs
    def UCB_block(self, block):
        ucb = 0.0
        A_block = self.gen_A_block_L(block)
        a_tilde = torch.matmul(A_block, block[self.m:, :].ravel())
        y = torch.dot(a_tilde, self.theta_block[self.m:, :].ravel()) # self.theta_hat_block[self.m:, :].ravel())
        z = torch.sqrt(torch.matmul(torch.t(a_tilde), torch.matmul(torch.inverse(self.V_tau[(self.m * self.d):, (self.m * self.d):]),
                                                                    a_tilde)))
        ucb = (y + self.beta * z) #(y ) # + self.beta * z)
        return ucb


    def oracle(self, block, lr):
        n_iter = 100
        obj_tmp = 0.0
        block_tmp = torch.zeros((self.m + self.L, self.d))
        for lr in [0.1, 0.8, 1.5]:
            for epoch in range(n_iter):
                obj = self.UCB_block(block)
                if obj > obj_tmp:
                    obj_tmp = obj.detach().clone()
                    #print(obj)
                    obj.backward()
                    with torch.no_grad():
                        block += lr * block.grad  # difficult to tune this rate
                        # we project
                        norm = torch.clamp(torch.norm(block.detach(), dim=1), min=1)
                        block /= torch.outer(norm, torch.ones(self.d))
                        block.grad.zero_()
        return block

    def gen_A_block_L_Greedysubopt(self, block):
        mat_block = torch.zeros((self.d * self.L, self.d * self.L))
        for i in range(self.L):
            mat = self.gen_A_Greedysubopt2(block[i:self.m + i, :])
            mat_block[(self.d * i): (self.d * i) + self.d, (self.d * i): (self.d * i) + self.d] = mat
        return mat_block


    # OBJECTIVE FUNCTION FOR GRADIENT DESCENT WITH GOAL OF FINDING BEST BLOCK USING UCBs
    def UCB_Greedysubopt(self, block):
        ucb = 0.0
        A_block = self.gen_A_block_L_Greedysubopt(block)
        a_tilde = torch.matmul(A_block, block[self.m:, :].ravel())
        y = torch.dot(a_tilde, self.theta_block[self.m:, :].ravel())  # self.theta_hat_block[self.m:, :].ravel())
        z = torch.sqrt(torch.matmul(torch.t(a_tilde),
                                    torch.matmul(torch.inverse(self.V_tau[(self.m * self.d):, (self.m * self.d):]),
                                                     a_tilde)))
        ucb = (y + self.beta * z)
        return ucb


    def oracle_Greedysubopt(self, block, lr):  # , obj_plot):
        n_iter = 100
        obj_tmp = 0.0
        block_tmp = torch.zeros((self.m + self.L, self.d))
        for lr in [0.1, 0.8, 1.5]:
            for epoch in range(n_iter):
                obj = self.UCB_Greedysubopt(block)
                if obj > obj_tmp:
                    obj_tmp = obj.detach().clone()
                    # print(obj)
                    obj.backward()
                    with torch.no_grad():
                        block += lr * block.grad  # difficult to tune this rate
                        # we project
                        norm = torch.clamp(torch.norm(block.detach(), dim=1), min=1)
                        block /= torch.outer(norm, torch.ones(self.d))
                        block.grad.zero_()
        return block

    def compute_rwd(self, actions, i, A_curr):
        a_tilde_t = torch.matmul(A_curr, actions[i, :])
        y = torch.dot(a_tilde_t, self.theta)
        rwd = np.random.normal(loc=y, scale=self.noise_level, size=1)
        return rwd

    def update_coeff(self, rwd, block, A_block_curr, flag):
        if flag == 0:
            # UPDATE U MATRIX USED TO COMPUTE THETA_HAT
            actions_tilde = torch.matmul(A_block_curr, block.ravel())
            actions_tilde[:self.d * self.m] = torch.zeros(self.d * self.m)  # the first d*m elements put to zero snce we do not use them for the estimate of theta_hat_block
            actions_tilde_x = torch.reshape(actions_tilde, (self.m + self.L, self.d))
            for i in range(self.m + self.L):
                actions_tilde_x[i, :] = rwd[i] * actions_tilde_x[i, :]
            actions_tilde_x = actions_tilde_x.ravel()
            self.U_tau += actions_tilde_x
            # UPDATE V_tau
            self.V_tau += torch.matmul(torch.t(actions_tilde), actions_tilde)
        else:
            actions_tilde = torch.matmul(A_block_curr, block.ravel())
            actions_tilde[:self.d * self.m] = torch.zeros(self.d * self.m)  # the first d*m elements put to zero snce we do not use them for the estimate of theta_hat_block
            #actions_tilde /= torch.outer(self.m ** (-self.alpha), torch.ones(self.d))
            #print(actions_tilde.shape)
            #for index in range(self.m+self.L):
            #    actions_tilde[index, :] /= torch.ones() * (self.m **-self.alpha)
            actions_tilde /= (((1 + self.m) ** self.alpha) * (self.m + self.L))
            actions_tilde_x = torch.reshape(actions_tilde, (self.m + self.L, self.d))
            for i in range(self.m + self.L):
                actions_tilde_x[i, :] = rwd[i] * actions_tilde_x[i, :]
            actions_tilde_x = actions_tilde_x.ravel()
            self.U_tau += actions_tilde_x
            # UPDATE V_tau
            self.V_tau += torch.matmul(torch.t(actions_tilde), actions_tilde)


    def update_beta(self, s):
        # UPDATE BOUND ON CONFIDENCE SET
        self.beta = np.sqrt(self.m + self.L) * np.sqrt(self.lambda_) + np.sqrt(
            2 * np.log(1 / self.delta_beta) + (self.d * (self.m + self.L)) * np.log(1 + ((s + 1) / ( self.d * self.lambda_))))

    def est_theta_hat(self):
        # UPDATE THE ESTIMATE OF THETA HAT
        self.theta_hat_block = torch.reshape(torch.matmul((torch.inverse(self.V_tau)), self.U_tau), (self.m + self.L, self.d))

