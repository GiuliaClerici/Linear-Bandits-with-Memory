import numpy as np
import torch


class LinUCB:

    def __init__(self, d, theta, delta_mat, noise_level):
        self.d = d
        self.noise_level = noise_level
        # theta - simple case: have one dimension equal to 1
        self.theta = theta
        # initial matrix A
        self.delta = delta_mat
        self.A = self.delta * torch.eye(d)
        self.A_t = self.delta * torch.eye(d)
        # theta hat
        # theta_hat = torch.zeros(d, requires_grad=True)
        self.theta_hat = torch.ones(d)  # torch.matmul(torch.inverse(torch.eye(d)), )
        # beta of confidence bonus
        self.lambda_ = 1
        self.delta_beta = 0.01
        self.beta = np.sqrt(self.lambda_) + np.sqrt(2 * np.log(1 / self.delta_beta) + d * np.log(1 + 1 / (d * self.lambda_)))
        # V_tau matrix
        self.V_tau = torch.eye(d) * 1
        # U_tau matrix
        self.U_tau = torch.zeros(d)

    # OBJECTIVE FUNCTION FOR GRADIENT DESCENT WITH GOAL OF FINDING BEST BLOCK USING UCBs
    def UCB(self, action):
        ucb = 0.0
        y = torch.dot(action, self.theta_hat)
        z = torch.sqrt(torch.matmul(torch.t(action), torch.matmul(torch.inverse(self.V_tau), action)))
        ucb = (y + self.beta * z)

        return ucb

    def oracle(self, action, lr):  # , obj_plot):
        n_iter = 100
        for epoch in range(n_iter):
            obj = self.UCB(action)
            #print(obj)
            obj.backward() # check if argument is causing errors retain_graph=True
            with torch.no_grad():
                action += lr * action.grad  # difficult to tune this rate
                # we project - sure about norm in 1D?
                norm = torch.norm(action).item()
                action /= norm
                action.grad.zero_()
        return action

    def gen_A(self, past_actions, alpha):
        mat = torch.eye(self.d) + torch.matmul(past_actions.T, past_actions)
        #mat = torch.linalg.matrix_power(torch.inverse(mat), alpha)
        mat_tmp = mat.detach().numpy()
        eigenv_tmp, U_tmp = np.linalg.eigh(mat_tmp)
        U = torch.tensor(U_tmp)
        eigenv = torch.tensor(eigenv_tmp)
        res = torch.matmul(torch.matmul(U, torch.diag(torch.pow(eigenv, alpha))), U.T)
        return mat


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

    def compute_rwd(self, action, A_t):
        a_tilde_t = torch.matmul(A_t, action)
        y = torch.dot(a_tilde_t, self.theta)
        rwd = np.random.normal(loc=y, scale=self.noise_level, size=1) #+ self.noise_level * np.random.randn(1)
        return rwd

    def update_coeff(self, rwd, action):
        rwd = torch.tensor(rwd)
        # UPDATE U MATRIX USED TO COMPUTE THETA_HAT
        self.U_tau += (rwd * action)
        # UPDATE V_tau
        self.V_tau += torch.matmul(torch.t(action), action)

    def update_beta(self, s):
        # UPDATE BOUND ON CONFIDENCE SET
        self.beta = np.sqrt(self.lambda_) + np.sqrt(
            2 * np.log(1 / self.delta_beta) + self.d * np.log(1 + (s + 1) / (self.d * self.lambda_)))

    def est_theta_hat(self):
        # UPDATE THE ESTIMATE OF THETA HAT
        self.theta_hat = torch.matmul((torch.inverse(self.V_tau)), self.U_tau)

