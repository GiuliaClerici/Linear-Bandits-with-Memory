import numpy as np
import torch


def soft_max(x):
    y = torch.exp(5 * x)
    z = y / torch.sum(y)
    res = torch.sum(z * x)
    return res

def soft_norm_block(block):
    n = torch.norm(block, dim=1)
    res = soft_max(n)
    #res = torch.max(n)
    return res

def soft_max_np(x):
    y = np.exp(5 * x)
    z = y / np.sum(y)
    res = np.sum(z * x)
    return res

def soft_norm_block_np(block):
    n = np.linalg.norm(block, axis=1)
    res = soft_max_np(n)
    #res = torch.max(n)
    return res

def norm_block(block):
    n = torch.norm(block, dim=1)
    res = torch.max(n)
    return res

def project_on_ball(block):
    x, d = block.size()
    proj_block = torch.zeros(x, d)
    for i in range(x):
        y = block[i, :]
        n = torch.norm(y).item()
        z = y / n
        proj_block[i, :] = z
    return proj_block

def gen_A(past_actions, alpha, delta, dim):
    mat = delta * torch.eye(dim) + torch.matmul(past_actions.T, past_actions)
    #res = torch.linalg.matrix_power(torch.inverse(mat), alpha) #torch.inverse(mat) ** alpha
    mat_tmp = mat.detach().numpy()
    eigenv_tmp, U_tmp = np.linalg.eigh(mat_tmp)
    U = torch.tensor(U_tmp)
    eigenv = torch.tensor(eigenv_tmp)
    res = torch.matmul(torch.matmul(U, torch.diag(torch.pow(eigenv, alpha))), U.T)
    return res

def rwd_block(theta, block, m, L, alpha, delta):
    ucb = 0.0
    for ind in range(L):
        A_t = gen_A(block[ind:m + ind, :], alpha, delta)
        a_tilde = torch.matmul(A_t, block[m + ind, :])
        y = torch.dot(a_tilde, theta)
        ucb += y
    return ucb

def compute_rwd(actions, i, A_curr, theta_star, noise):
    a_tilde_t = torch.matmul(A_curr, actions[i, :])
    y = torch.dot(a_tilde_t, theta_star)
    rwd = np.random.normal(loc=y, scale=noise, size=1) #+ self.noise_level * np.random.randn(1)
    return rwd
