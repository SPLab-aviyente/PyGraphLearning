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

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from ..singleview.synthetic import gen_smooth_gs
from ...utils import edge_swap

def gen_smooth_gs_er(n_nodes, p, n_views, pert_percent, n_signals, filter="Gaussian", alpha=10, 
                     noise_amount=0.5):
    """Generate multiple sets of smooth signals from multiple graphs created by perturbing an 
    Erdos-Renyi graph. The method generate a random Erdos-Renyi graph G, then perturbs G to create 
    multiple graphs, i.e. G_i's. Perturbation is by done swapping the edges of G while preserving 
    the degree distribution. From each G_i, smooth signals are generated. See 
    `data.singleview.synthetic.gen_smooth_gs` for how signal generation is performed. 

    Parameters
    ----------
    n_nodes : int
        Number of nodes
    p : float
        Edge probability of the Erdos-Renyi random graph
    n_views : int
        Number of perturbed graphs
    pert_percent : float, or list of floats
        Amount of perturbation to generate perturbed graphs from the original graph. If float, each 
        graph is generated with the same amount perturbation. If list, its length must be n_views 
        and pert_percent[i] is  the amount of perturbation for G_i.
    n_signals : int, or list of ints
        Number of signals to generate from each G_i. If int, the same number of signals are 
        generated from each graph. If list, its length must be n_views and n_signals[i] number of 
        signals are generated from G_i.
    filter : str, optional
        The filter to use to generate smooth signals. It can be 'Gaussian', 'Tikhonov' or 'Heat', 
        by default 'Gaussian' 
    alpha : float, optional
        A positive number used as the parameter for Tikhonov filter and Heat filter, by default 10
    noise_amount : float, optional
        Amount of the noise to add the graph signals. Amount of the noise determined in L2-sense, 
        that is if x is the clean signal |e|_2/|x|_2 = noise_amount where is e the additive noise.
        By default 0.1.

    Returns
    -------
    G : networkx graph
        The original Erdos-Renyi graph
    G_views : list of networkx graphs
        Perturbed graphs
    X_views : list of ndarrays
        The sets of generated smooth graph signals.
    """

    if isinstance(pert_percent, float) or isinstance(pert_percent, int): 
        # All views have the same amount of perturbation
        pert_percent = [pert_percent]*n_views

    if len(pert_percent) != n_views:
        raise ValueError("The length of the argument 'pert_percent' must be 'n_views'")

    if isinstance(n_signals, int): 
        # All views have the same amount of signals
        n_signals = [n_signals]*n_views

    if len(n_signals) != n_views:
        raise ValueError("The length of the argument 'n_signals' must be 'n_views'")

    # Generate underlying graph
    G = nx.erdos_renyi_graph(n_nodes, p)

    n_edges = G.number_of_edges()

    G_views = []
    X_views = []
    for i in range(n_views):
        G_views.append(G.copy())
        n_swaps = int(n_edges*pert_percent[i]/2)
        edge_swap.topological_undirected(G_views[i], n_swaps)
        X_views.append(gen_smooth_gs(G_views[i], n_signals[i], filter, alpha, noise_amount))

    return G, G_views, X_views

def gen_smooth_gs_ba(n_nodes, m, n_views, pert_percent, n_signals, filter="Gaussian", alpha=10, 
                     noise_amount=0.5):
    """Generate multiple sets of smooth signals from multiple graphs created by perturbing a 
    Barabasi-Albert graph. The method generate a random Barabasi-Albert graph G, then perturbs G to 
    create multiple graphs, i.e. G_i's. Perturbation is by done swapping the edges of G while 
    preserving the degree distribution. From each G_i, smooth signals are generated. See 
    `data.singleview.synthetic.gen_smooth_gs` for how signal generation is performed. 

    Parameters
    ----------
    n_nodes : int
        Number of nodes
    m : int
        Number of edges to add at each step of Barabasi-Albert random network generation
    n_views : int
        Number of perturbed graphs
    pert_percent : float, or list of floats
        Amount of perturbation to generate perturbed graphs from the original graph. If float, each 
        graph is generated with the same amount perturbation. If list, its length must be n_views 
        and pert_percent[i] is  the amount of perturbation for G_i.
    n_signals : int, or list of ints
        Number of signals to generate from each G_i. If int, the same number of signals are 
        generated from each graph. If list, its length must be n_views and n_signals[i] number of 
        signals are generated from G_i.
    filter : str, optional
        The filter to use to generate smooth signals. It can be 'Gaussian', 'Tikhonov' or 'Heat', 
        by default 'Gaussian' 
    alpha : float, optional
        A positive number used as the parameter for Tikhonov filter and Heat filter, by default 10
    noise_amount : float, optional
        Amount of the noise to add the graph signals. Amount of the noise determined in L2-sense, 
        that is if x is the clean signal |e|_2/|x|_2 = noise_amount where is e the additive noise.
        By default 0.1.

    Returns
    -------
    G : networkx graph
        The original Erdos-Renyi graph
    G_views : list of networkx graphs
        Perturbed graphs
    X_views : list of ndarrays
        The sets of generated smooth graph signals.
    """

    if isinstance(pert_percent, float): # All views have the same amount of perturbation
        pert_percent = [pert_percent]*n_views

    if len(pert_percent) != n_views:
        raise ValueError("The length of the argument 'pert_percent' must be 'n_views'")

    if isinstance(n_signals, int): # All views have the same amount of signals
        n_signals = [n_signals]*n_views

    if len(n_signals) != n_views:
        raise ValueError("The length of the argument 'n_signals' must be 'n_views'")

    # Generate underlying graph
    G = nx.barabasi_albert_graph(n_nodes, m)

    n_edges = G.number_of_edges()

    G_views = []
    X_views = []
    for i in range(n_views):
        G_views.append(G.copy())
        n_swaps = int(n_edges*pert_percent[i]/2)
        edge_swap.topological_undirected(G_views[i], n_swaps)
        X_views.append(gen_smooth_gs(G_views[i], n_signals[i], filter, alpha, noise_amount))

    return G, G_views, X_views

def gen_smooth_gs_rgg(n_nodes, std, threshold, n_views, pert_percent, n_signals, filter="Gaussian", alpha=10, 
                     noise_amount=0.5):
    """Generate multiple sets of smooth signals from multiple graphs created by perturbing a 
    random geometric graph. The method generate a random geometric graph G, which is a graph 
    generated from a set of random points drawn uniformly from 2D unit square by adding a weighted 
    edge between each pair of points. Weights are calculated by a RBF kernel and those smaller than 
    a value is removed to make the graph sparser. G is then to perturbed to create multiple graphs, 
    i.e. G_i's. Perturbation is by done swapping the edges of G while preserving the degree 
    distribution. From each G_i, smooth signals are generated. 
    See `data.singleview.synthetic.gen_smooth_gs` for how signal generation is performed. 

    Parameters
    ----------
    n_nodes : int
        Number of nodes
    std : float
        Standard deviation of RBF kernel
    threshold : float
        Threshold to remove edges with small weights
    n_views : int
        Number of perturbed graphs
    pert_percent : float, or list of floats
        Amount of perturbation to generate perturbed graphs from the original graph. If float, each 
        graph is generated with the same amount perturbation. If list, its length must be n_views 
        and pert_percent[i] is  the amount of perturbation for G_i.
    n_signals : int, or list of ints
        Number of signals to generate from each G_i. If int, the same number of signals are 
        generated from each graph. If list, its length must be n_views and n_signals[i] number of 
        signals are generated from G_i.
    filter : str, optional
        The filter to use to generate smooth signals. It can be 'Gaussian', 'Tikhonov' or 'Heat', 
        by default 'Gaussian' 
    alpha : float, optional
        A positive number used as the parameter for Tikhonov filter and Heat filter, by default 10
    noise_amount : float, optional
        Amount of the noise to add the graph signals. Amount of the noise determined in L2-sense, 
        that is if x is the clean signal |e|_2/|x|_2 = noise_amount where is e the additive noise.
        By default 0.1.

    Returns
    -------
    G : networkx graph
        The original Erdos-Renyi graph
    G_views : list of networkx graphs
        Perturbed graphs
    X_views : list of ndarrays
        The sets of generated smooth graph signals.
    """

    if isinstance(pert_percent, float): # All views have the same amount of perturbation
        pert_percent = [pert_percent]*n_views

    if len(pert_percent) != n_views:
        raise ValueError("The length of the argument 'pert_percent' must be 'n_views'")

    if isinstance(n_signals, int): # All views have the same amount of signals
        n_signals = [n_signals]*n_views

    if len(n_signals) != n_views:
        raise ValueError("The length of the argument 'n_signals' must be 'n_views'")

    # Generate underlying graph
    rng = np.random.default_rng()
    points = rng.uniform(-1, 1, size=(n_nodes, 2))
    
    A = rbf_kernel(points, gamma=1/(std**2))
    A[A<threshold] = 0
    A[np.diag_indices_from(A)] = 0
    G = nx.from_numpy_array(A)

    n_edges = G.number_of_edges()

    G_views = []
    X_views = []
    for i in range(n_views):
        G_views.append(G.copy())
        n_swaps = int(n_edges*pert_percent[i]/2)
        edge_swap.topological_undirected(G_views[i], n_swaps)
        X_views.append(gen_smooth_gs(G_views[i], n_signals[i], filter, alpha, noise_amount))

    return G, G_views, X_views