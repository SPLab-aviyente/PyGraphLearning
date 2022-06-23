# PyGraphLearning - a python package for graph signal processing based graph learning
# Copyright (C) 2021 Abdullah Karaaslanli <evdilak@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import multiprocessing
from numba.core.decorators import njit

import numpy as np
import scipy as sp
from joblib import Parallel, delayed, parallel
from scipy import sparse

from ..utils import rowsum_matrix
from .singleview import unsigned_ladmm

# def _project_hyperplane(v, n):
#     """ Project v onto the hyperplane defined by np.sum(v) = -n
#     """
#     return v - (n + np.sum(v))/(len(v))

# def _qp_admm(b, inv_mat_part1, inv_mat_part2, rho=1, max_iter=1000, size_constrained=False, 
#              w_init=None, lambda_init=None):
#     m = len(b) # number of node pairs
#     n = (1 + np.sqrt(8*m + 1))//2 # number of nodes

#     # Initialization
#     if w_init is None:
#         w = np.zeros((m, 1)) # slack variable
#     else:
#         w = w_init
    
#     if lambda_init is None:
#         lambda_ = np.zeros((m, 1)) # Lagrange multiplier
#     else:
#         lambda_ = lambda_init

#     for iter in range(max_iter):
#         # Update l
#         l_temp = b + rho*w + lambda_
#         l = np.asarray(inv_mat_part1@l_temp + inv_mat_part2(l_temp))

#         if size_constrained:
#             l = _project_hyperplane(l, n)

#         # Update slack variable
#         w = l - lambda_/rho
#         w[w>0] = 0
        
#         # Update Lagrange multiplier
#         lambda_ += rho*(w - l)

#         residual = np.linalg.norm(w-l)
#         if residual < 1e-4:
#             break

#     w[w>-1e-4] = 0

#     return w, lambda_

# def unsigned(k, d, alpha, beta):
#     if not isinstance(k, list):
#         k = [k]

#     if not isinstance(d, list):
#         d = [d]

#     n = len(d[0]) # number of nodes
#     m = len(k[0]) # number of node pairs

#     S = rowsum_matrix(n)

#     # ADMM Parameter for li subproblems
#     rho_li = 1 

#     # Inverse matrix for li subproblems
#     a = 4*alpha + rho_li
#     b = 0 # 2*alpha
#     c1 = 1/a
#     c2 = b/(a*(a+n*b-2*b))
#     c3 = 4*b**2/(a*(a+2*n*b-2*b)*(a+n*b-2*b))
#     li_inv_mat_part1 = c1*sp.sparse.eye(m) - c2*S.T@S
#     li_inv_mat_part2 = lambda x : c3*np.sum(x)*np.ones((m, 1))

#     # ADMM Parameter for l subproblem
#     N = len(k) # number of kernels
#     rho_l = 1

#     # Inverse matrix for l subproblem
#     a = 4*alpha*N + 4*beta + rho_li
#     b = 2*beta # 2*alpha*N + 2*beta
#     c1 = 1/a
#     c2 = b/(a*(a+n*b-2*b))
#     c4 = 4*b**2/(a*(a+2*n*b-2*b)*(a+n*b-2*b))
#     l_inv_mat_part1 = c1*sp.sparse.eye(m) - c2*S.T@S
#     l_inv_mat_part2 = lambda x : c4*np.sum(x)*np.ones((m, 1))

#     bs = []
#     for i in range(len(k)):
#         bs.append(2*k[i] - S.T@d[i])
#         if np.ndim(bs[i]) == 1:
#             bs[i] = bs[i][..., None]

#     l = np.zeros((m, 1))
#     lis = np.zeros((m, N))
#     lambdas = np.zeros((m, N))
#     lambda_l = np.zeros((m, 1))

#     # print("_"*20)

#     for iter in range(1000):

#         lis_old = lis.copy()

#         # Update lis
#         b = 4*alpha*l # + 2*alpha*S.T@S@l
#         for i in range(len(k)):
#             b_i = b - bs[i]

#             w = lis[:, i][..., None]
#             lambda_ = lambdas[:, i][..., None]
            
#             w, lambda_ = _qp_admm(b_i, li_inv_mat_part1, li_inv_mat_part2, rho_li, 
#                                   size_constrained=True, w_init=None, lambda_init=None)

#             lis[:, i] = np.squeeze(w)
#             lambdas[:, i] = np.squeeze(lambda_)

#         change1 = np.linalg.norm(lis_old - lis)/np.size(lis)

#         # print("Iter {}: {:.3f}".format(iter, change1), end=" ")

#         # Update l
#         l_old = l

#         li_sum = np.sum(lis, axis=1)[..., None]
#         b = 4*alpha*li_sum # + 2*alpha*S.T@S@li_sum
#         l, lambda_l = _qp_admm(b, l_inv_mat_part1, l_inv_mat_part2, rho_l, size_constrained=False, 
#                      w_init = None, lambda_init = None)

#         change2 = np.linalg.norm(l_old - l)/np.size(l)

#         if change1 < 1e-4 and change2 < 1e-4:
#             # print(iter)
#             break

#         # print("{:.3f}".format(change2))

#         # print("X", end="")

#     # print("")

#     return lis, l

# def model1(k, d, alpha1, alpha2, alpha3, degree_reg="l2"):
#     if not isinstance(k, list):
#         k = [k]

#     if not isinstance(d, list):
#         d = [d]

#     n = len(d[0]) # number of nodes
#     m = len(k[0]) # number of node pairs
#     N = len(k) # number of views

#     l = np.zeros((m, 1))

#     num_cores = multiprocessing.cpu_count()

#     for iter in range(100):
#         lis = Parallel(n_jobs=min(num_cores, N))(
#             delayed(unsigned_ladmm)(k[i] - alpha2*np.squeeze(l), d[i], alpha1, alpha2, degree_reg) 
#             for i in range(N))
#         # for i in range(N):
#         #     ki = k[i] - alpha2*np.squeeze(l)
#         #     lis[i] = unsigned_ladmm(ki, d[i], alpha1, alpha2, degree_reg=degree_reg)

#         l_old = l
#         l = np.zeros((m, 1))
#         for i in range(N):
#             l -= lis[i]
#         l /= N + alpha3/alpha2

#         if np.linalg.norm(l-l_old) < 1e-4:
#             break

#     return np.abs(l), lis

# def model2(k, d, alpha1, alpha2, alpha3, alpha4, degree_reg="l2"):
#     if not isinstance(k, list):
#         k = [k]

#     if not isinstance(d, list):
#         d = [d]

#     n = len(d[0]) # number of nodes
#     m = len(k[0]) # number of node pairs
#     N = len(k) # number of views

#     l = np.zeros((m, 1))
    
#     num_cores = multiprocessing.cpu_count()

#     for iter in range(100):
#         lis = Parallel(n_jobs=min(num_cores, N))(
#             delayed(unsigned_ladmm)(k[i] - alpha2*np.squeeze(l), d[i], alpha1, alpha2, degree_reg) 
#             for i in range(N))
#         for i in range(N):
#             ki = k[i] - alpha2*np.squeeze(l)
#             lis[i] = unsigned_ladmm(ki, d[i], alpha1, alpha4, degree_reg=degree_reg)

#         l_old = l
#         l = np.zeros((m, 1))
#         for i in range(N):
#             l -= lis[i]/N

#         l = - np.maximum(- l - alpha3/(2*N*alpha2), 0)

#         if np.linalg.norm(l-l_old) < 1e-4:
#             break

#     return np.abs(l), lis

def _random_laplacian(m, n):
    rng = np.random.default_rng()
    l = rng.normal(size=(m, 1))
    l[l>0] = 0
    return l/np.abs(np.sum(l))*n

@njit
def _project_to_neg_simplex_numba(v_sorted, a):
    # Find mu
    for j in range(1, len(v_sorted)+1):
        mu = (np.sum(v_sorted[:j]) + a)/j
        if v_sorted[j-1] - mu > 0:
            mu = (np.sum(v_sorted[:j-1]) + a)/(j-1)
            break

    return mu

def unsigned_proxlinear(k, d, alpha1, alpha2, alpha3, alpha4, sparsity="l2", max_iter=200, 
                        return_obj=False):
    # TODO: Docstring
    if not isinstance(k, list):
        k = [k]

    if not isinstance(d, list):
        d = [d]

    n = len(d[0]) # number of nodes
    m = len(k[0]) # number of node pairs
    N = len(k) # number of views

    S = rowsum_matrix(n)

    # initialization
    lis = []
    lis_prev = []
    for i in range(len(k)):
        lis.append(_random_laplacian(m, n))
        lis_prev.append(np.zeros((m, 1)))

    l = _random_laplacian(m, n)

    # data vectors: 2*k[i] - S.T@d[i]
    bis = []
    for i in range(len(k)):
        bis.append(2*k[i] - S.T@d[i])
        if np.ndim(bis[i]) == 1:
            bis[i] = bis[i][..., None]

    # Prox-linear BCD parameters
    step_size_li = (4*alpha1*(n-1) + 2*alpha2) # L_i^{k-1}
    w = 1 # Extrapolation parameter

    A = S.T@S # Pre-compute for computational efficieny
    
    # Should I calculate the value of objective function at each iteration?
    if return_obj:
        degree_term = lambda li: alpha1*np.linalg.norm(S@li)**2
        consensus_term = lambda  lis, l: alpha2*np.linalg.norm(alpha4*lis[i]-l)**2
        if sparsity == "l2":
            sparsity_term = lambda l: alpha3*np.linalg.norm(l)**2
        else:
            sparsity_term = lambda l: alpha3*np.linalg.norm(l, 1)**2

        objective = lambda lis, l: np.sum(
            [np.asarray(bis[i].T@lis[i] + degree_term(lis[i]) + consensus_term(lis[i], l)).item()
             for i in range(N)]) + sparsity_term(l)

        objective_vals = np.zeros(max_iter)


    for iter in range(max_iter):
        # Update L_i's
        # TODO: This can be parallelized, but it doesn't speed the method up
        for i in range(N): 
            l_i_hat = (1+w)*lis[i] - w*lis_prev[i]
            g_i = bis[i] + 2*alpha1*A@l_i_hat - 2*alpha2*(l - alpha4*l_i_hat)
            v_i_hat = l_i_hat - g_i/step_size_li

            lis_prev[i] = lis[i]
            
            # Projection
            # TODO: Sorting make this slow, alternative is binary search
            mu = _project_to_neg_simplex_numba(np.sort(v_i_hat, axis=0), n)
            lis[i] = v_i_hat - mu
            lis[i][lis[i] > 0] = 0

        # Update L
        l_old = l
        l = np.zeros((m, 1))
        for i in range(N):
            l += lis[i]

        if sparsity == "l2":
            l /= N + alpha3
        else:
            l = - np.maximum(- l/N - alpha3, 0)

        if return_obj:
            objective_vals[iter] = objective(lis, l)

        if np.linalg.norm(l_old - l) < 1e-4:
            break

    l[l>-1e-4] = 0
    for v in range(N):
        lis[v][lis[v] > -1e-4] = 0
        lis[v] = np.abs(lis[v])
    
    return np.abs(l), lis # returns adjacency matrices