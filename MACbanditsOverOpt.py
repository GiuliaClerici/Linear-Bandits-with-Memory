import numpy as np
import torch

class MAC_Bandit:

    def __init__(self, d, m, L, theta, alpha, delta_mat, noise_level, flag_rising):
        self.d, self.m, self.L = d, m, L
        self.alpha = alpha
        self.noise_level = noise_level
        # theta - simple case: have one dimension equal to 1
        self.theta = theta
        # theta block of size d(m+l)
        theta_block = torch.zeros((m + L, d))
        # vector storing the m pre_actions
        self.pre_actions = torch.zeros(m, d)
        # initial matrix A
        self.delta = delta_mat
        self.A = self.delta * torch.eye(d)
        self.A_t = self.delta * torch.eye(d)
        # theta hat
        # theta_hat = torch.zeros(d, requires_grad=True)
        if flag_rising == 2:
            self.theta_hat = torch.tensor([1./5., 4.9/5.]) #torch.ones(d) # for model sel alpha torch.tensor([1./5., 4.9/5.])  # torch.matmul(torch.inverse(torch.eye(d)), )
        else:
            self.theta_hat = torch.ones(d)
        # beta of confidence bonus
        self.lambda_ = 1
        self.delta_beta = 0.01
        self.beta = np.sqrt(self.lambda_) + np.sqrt(2 * np.log(1 / self.delta_beta) + d * np.log(1 + 1 / (d * self.lambda_)))
        # block
        self.block = torch.zeros((m + L, d))
        # V_tau matrix
        self.flag = flag_rising
        if self.flag == 1: # or 2??
            self.V_tau = torch.eye(d) * 1.1 #.1 # 1 if satiation, 1.1 if rising
        else:
            self.V_tau = torch.eye(d) * 1
        # U_tau matrix
        self.U_tau = torch.zeros(d)


    def gen_A(self, past_actions):
        mat = torch.eye(self.d) + torch.matmul(past_actions.T, past_actions)
        #print(mat)
        #res = torch.linalg.matrix_power(torch.inverse(mat), self.alpha)
        #eigenv, U = torch.linalg.eigh(mat)
        mat_tmp = mat.detach().numpy()
        eigenv_tmp, U_tmp = np.linalg.eigh(mat_tmp)
        U = torch.tensor(U_tmp)
        eigenv = torch.tensor(eigenv_tmp)
        res = torch.matmul(torch.matmul(U, torch.diag(torch.pow(eigenv, self.alpha))), U.T)
        return res

    # OBJECTIVE FUNCTION FOR GRADIENT DESCENT WITH GOAL OF FINDING BEST BLOCK USING UCBs
    def UCB(self, block):
        ucb = 0.0
        for ind in range(self.L):
            self.A_t = self.gen_A(block[ind:self.m + ind, :])
            a_tilde = torch.matmul(self.A_t, block[self.m + ind, :])
            y = torch.dot(a_tilde, self.theta_hat)
            z = torch.sqrt(torch.matmul(torch.t(a_tilde), torch.matmul(torch.inverse(self.V_tau),
                                                                       a_tilde)))  # look for norm func if there's one
            ucb += (y + self.beta * z)

        return ucb

    def UCB_star(self, block):
        ucb = 0.0
        for ind in range(self.L):
            self.A_t = self.gen_A(block[ind:self.m + ind, :])
            a_tilde = torch.matmul(self.A_t, block[self.m + ind, :])
            y = torch.dot(a_tilde, self.theta)
            ucb += (y)
        return ucb

    def oracle(self, block, lr):  # , obj_plot):
        n_iter = 100
        obj_tmp = 0.0
        block_tmp = torch.zeros((self.m + self.L, self.d))
        for lr in [0.1, 0.8, 1.5]:
            for epoch in range(n_iter):
                obj = self.UCB(block)
                if obj > obj_tmp:
                    obj_tmp = obj.detach().clone()
                    #print("check: ", obj.backward())
                    obj.backward() # check if argument is causing errors retain_graph=True
                    with torch.no_grad():
                        #print("Gradient: ", block.grad)
                        block += lr * block.grad  # difficult to tune this rate
                        # we project
                        norm = torch.clamp(torch.norm(block.detach(), dim=1), min=1)
                        block /= torch.outer(norm, torch.ones(self.d))
                        block.grad.zero_()
                        #if obj > obj_tmp:
                        #    block_tmp = block.detach().clone()

        return block #_tmp


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

    # OBJECTIVE FUNCTION FOR GRADIENT DESCENT WITH GOAL OF FINDING BEST BLOCK USING UCBs
    def UCB_Greedysubopt(self, block):
        ucb = 0.0
        for ind in range(self.L):
            self.A_t = self.gen_A_Greedysubopt2(block[ind:self.m + ind, :])
            a_tilde = torch.matmul(self.A_t, block[self.m + ind, :])
            y = torch.dot(a_tilde, self.theta_hat)
            z = torch.sqrt(torch.matmul(torch.t(a_tilde), torch.matmul(torch.inverse(self.V_tau),
                                                                       a_tilde)))  # look for norm func if there's one
            ucb += (y + self.beta * z)

        return ucb


    def oracle_Greedysubopt(self, block, lr):  # , obj_plot):
        n_iter = 100
        for epoch in range(n_iter):
            obj = self.UCB_Greedysubopt(block)
            #print("check: ", obj.backward())
            obj.backward() # check if argument is causing errors retain_graph=True
            with torch.no_grad():
                #print("Gradient: ", block.grad)
                block += lr * block.grad  # difficult to tune this rate
                # we project
                norm = torch.clamp(torch.norm(block.detach(), dim=1), min=1)
                block /= torch.outer(norm, torch.ones(self.d))
                block.grad.zero_()
        return block

    def compute_rwd(self, actions, i, A_curr):
        a_tilde_t = torch.matmul(A_curr.T, actions[i, :])
        y = torch.dot(a_tilde_t, self.theta)
        rwd = np.random.normal(loc=y, scale=self.noise_level, size=1) #+ self.noise_level * np.random.randn(1)
        return rwd

    def update_coeff(self, rwd, actions, i, A_curr):
        rwd = torch.tensor(rwd)
        if self.flag == 0:
            # UPDATE U MATRIX USED TO COMPUTE THETA_HAT
            #self.U_tau += (rwd * torch.matmul(A_curr, actions[self.m + i, :]))
            actions_tilde = torch.matmul(A_curr.T, actions[i, :])
            self.U_tau += (rwd * actions_tilde )
            # UPDATE V_tau
            #actions_tilde = torch.matmul(A_curr, actions[self.m + i, :])
            self.V_tau += torch.matmul(torch.t(actions_tilde), actions_tilde)
        else:
            # UPDATE U MATRIX USED TO COMPUTE THETA_HAT
            # self.U_tau += (rwd * torch.matmul(A_curr, actions[self.m + i, :]))
            actions_tilde = torch.matmul(A_curr, actions[i, :])
            actions_tilde /= self.m **(self.alpha)
            self.U_tau += (rwd * actions_tilde)
            # UPDATE V_tau
            self.V_tau += torch.matmul(torch.t(actions_tilde), actions_tilde)

    def update_beta(self, s):
        # UPDATE BOUND ON CONFIDENCE SET
        self.beta = np.sqrt(self.lambda_) + np.sqrt(
            2 * np.log(1 / self.delta_beta) + self.d * np.log(1 + ((s + 1) * self.L) / (self.d * self.lambda_)))

    def est_theta_hat(self):
        # UPDATE THE ESTIMATE OF THETA HAT
        self.theta_hat = torch.matmul((torch.inverse(self.V_tau)), self.U_tau)