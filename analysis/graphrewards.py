###############################################################################
#
# Adapted from https://github.com/lrjconan/GRAN/ which in turn is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
import graph_tool.all as gt
##Navigate to the ./util/orca directory and compile orca.cpp
# g++ -O2 -std=c++11 -o orca orca.cpp
import os
import copy
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import subprocess as sp
import concurrent.futures

import pygsp as pg
import secrets
from string import ascii_uppercase, digits
from datetime import datetime
from scipy.linalg import eigvalsh
from scipy.stats import chi2
from analysis.dist_helper import compute_mmd, gaussian_emd, gaussian, emd, gaussian_tv, compute_mmdlist
from torch_geometric.utils import to_networkx
import wandb

PRINT_TIME = False
__all__ = ['degree_stats', 'clustering_stats', 'orbit_stats_all', 'spectral_stats', 'eval_acc_lobster_graph']


def degree_worker(G):
    return np.array(nx.degree_histogram(G))

def degree_statslist(graph_ref_list, graph_pred_list, is_parallel=True, compute_emd=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
        Args:
            graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(
                nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmdlist(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmdlist(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


###############################################################################

def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist

def clustering_statslist(graph_ref_list,
                     graph_pred_list,
                     bins=100,
                     is_parallel=True, compute_emd=False):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                    clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)

        # check non-zero elements in hist
        # total = 0
        # for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        # print(total)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd, sigma=1.0 / 10)
        mmd_dist = compute_mmdlist(sample_ref, sample_pred, kernel=gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)
    else:
        mmd_dist = compute_mmdlist(sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist

# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    '3path': [1, 2],
    '4cycle': [8],
}
COUNT_START_STR = 'orbit counts:'


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    # tmp_fname = f'analysis/orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = f'orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), tmp_fname)
    # print(tmp_fname, flush=True)
    f = open(tmp_fname, 'w')
    f.write(
        str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()
    output = sp.check_output(
        [str(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'orca/orca')), 'node', '4', tmp_fname, 'std'])
    output = output.decode('utf8').strip()
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR) + 2
    output = output[idx:]
    node_orbit_counts = np.array([
        list(map(int,
                 node_cnts.strip().split(' ')))
        for node_cnts in output.strip('\n').split('\n')
    ])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts



def orbit_stats_alllist(graph_ref_list, graph_pred_list, compute_emd=False):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)

    # mmd_dist = compute_mmd(
    #     total_counts_ref,
    #     total_counts_pred,
    #     kernel=gaussian,
    #     is_hist=False,
    #     sigma=30.0)

    # mmd_dist = compute_mmd(
    #         total_counts_ref,
    #         total_counts_pred,
    #         kernel=gaussian_tv,
    #         is_hist=False,
    #         sigma=30.0)  

    if compute_emd:
        # mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=emd, sigma=30.0)
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmdlist(total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False, sigma=30.0)
    else:
        mmd_dist = compute_mmdlist(total_counts_ref, total_counts_pred, kernel=gaussian_tv, is_hist=False, sigma=30.0)
    return mmd_dist

def eval_acc_lobster_graph(G_list):
    G_list = [copy.deepcopy(gg) for gg in G_list]
    count = 0
    for gg in G_list:
        if is_lobster_graph(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_tree_graph(G_list):
    count = 0
    for gg in G_list:
        if nx.is_tree(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_sbm_graph(G_list, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000, is_parallel=True):
    count = 0.0
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for prob in executor.map(is_sbm_graph,
                                     [gg for gg in G_list], [p_intra for i in range(len(G_list))],
                                     [p_inter for i in range(len(G_list))],
                                     [strict for i in range(len(G_list))],
                                     [refinement_steps for i in range(len(G_list))]):
                count += prob
    else:
        for gg in G_list:
            count += is_sbm_graph(gg, p_intra=p_intra, p_inter=p_inter, strict=strict,
                                  refinement_steps=refinement_steps)
    return count / float(len(G_list))

def eval_acc_sbm_graphlist(G_list, p_intra=0.3, p_inter=0.005, strict=False, refinement_steps=1000, is_parallel=True):
    valid_list = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for prob in executor.map(is_sbm_graph,
                                     [gg for gg in G_list], [p_intra for i in range(len(G_list))],
                                     [p_inter for i in range(len(G_list))],
                                     [strict for i in range(len(G_list))],
                                     [refinement_steps for i in range(len(G_list))]):
                if strict:
                    if prob:
                        valid_list.append(1.)
                    else:
                        valid_list.append(0.)
                else:
                    valid_list.append(prob)
    else:
        for gg in G_list:
            prob= is_sbm_graph(gg, p_intra=p_intra, p_inter=p_inter, strict=strict,
                                  refinement_steps=refinement_steps)
            if strict:
                if prob:
                    valid_list.append(1.)
                else:
                    valid_list.append(0.)
            else:
                valid_list.append(prob)
    return valid_list


def eval_acc_planar_graph(G_list):
    count = 0
    for gg in G_list:
        if is_planar_graph(gg):
            count += 1
    return count / float(len(G_list))

def eval_acc_planar_graphlist(G_list):
    valid_list = []
    for gg in G_list:
        if is_planar_graph(gg):
            valid_list.append(1)
        else:
            valid_list.append(0)
    return valid_list


def is_planar_graph(G):
    return nx.is_connected(G) and nx.check_planarity(G)[0]


def is_lobster_graph(G):
    """
        Check a given graph is a lobster graph or not

        Removing leaf nodes twice:

        lobster -> caterpillar -> path

    """
    ### Check if G is a tree
    if nx.is_tree(G):
        G = G.copy()
        ### Check if G is a path after removing leaves twice
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        num_nodes = len(G.nodes())
        num_degree_one = [d for n, d in G.degree() if d == 1]
        num_degree_two = [d for n, d in G.degree() if d == 2]

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        else:
            return False
    else:
        return False


def is_sbm_graph(G, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000):
    """
    Check if how closely given graph matches a SBM with given probabilites by computing mean probability of Wald test statistic for each recovered parameter
    """

    adj = nx.adjacency_matrix(G).toarray()
    idx = adj.nonzero()
    g = gt.Graph()
    g.add_edge_list(np.transpose(idx))
    try:
        state = gt.minimize_blockmodel_dl(g)
    except ValueError:
        if strict:
            return False
        else:
            return 0.0

    # Refine using merge-split MCMC
    for i in range(refinement_steps):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    b = state.get_blocks()
    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = e.todense()[:n_blocks, :n_blocks]
    if strict:
        if (node_counts > 40).sum() > 0 or (node_counts < 20).sum() > 0 or n_blocks > 5 or n_blocks < 2:
            return False

    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = edge_counts / (max_inter_edges + 1e-6)

    W_p_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)

    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p = 1 - chi2.cdf(abs(W), 1)
    p = p.mean()
    if strict:
        return p > 0.9  # p value < 10 %
    else:
        return p

def distance2score(dlist,sigma=1):
    score_list = [np.e**(-x**1/(sigma**1)) for x in dlist]
    return score_list

import random

def loader_to_nx(loader):
        networkx_graphs = []
        for i, batch in enumerate(loader):
            data_list = batch.to_data_list()
            for j, data in enumerate(data_list):
                networkx_graphs.append(to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=True,
                                                   remove_self_loops=True))
        return networkx_graphs


def gen_reward_list(generated_graphs,train_graphs,dataname):
    total_train = len(train_graphs)
    reference_graphs = random.sample(train_graphs,total_train)
    networkx_graphs = []
    adjacency_matrices = []
    for graph in generated_graphs:
        node_types, edge_types = graph
        A = edge_types.bool().cpu().numpy()
        adjacency_matrices.append(A)

        nx_graph = nx.from_numpy_array(A)
        networkx_graphs.append(nx_graph)
    score_list = 0
    planar_base = {
         "degree":0.01, 
         "clustering": 0.01, 
         "orbit": 0.01,
    }
    sbm_base = {
         "degree":0.01, 
         "clustering": 0.01, 
         "orbit": 0.01,
    }
    if dataname=="planar":
        base = planar_base
    else:
        base = sbm_base
    degrees = degree_statslist(reference_graphs, networkx_graphs, is_parallel=True,
                            compute_emd=False)

    score_list += np.array(distance2score(degrees,base["degree"]))

    clustering = clustering_statslist(reference_graphs, networkx_graphs, bins=100, is_parallel=True,
                                    compute_emd=False)
    score_list+=np.array(distance2score(clustering,base["clustering"]))

    orbit = orbit_stats_alllist(reference_graphs, networkx_graphs, compute_emd=False)
    score_list += np.array(distance2score(orbit,base["orbit"]))

    score_list = score_list/3

    if dataname=="sbm":
        acc = eval_acc_sbm_graphlist(networkx_graphs, refinement_steps=100, strict=True)
    elif dataname=="planar":
        acc = eval_acc_planar_graphlist(networkx_graphs)
    score_list = 0.2*score_list+0.8*np.array(acc)
    return score_list

def gen_reward_label(generated_graphs,dataname):
    # total_train = len(train_graphs)
    # reference_graphs = random.sample(train_graphs,total_train)
    networkx_graphs = []
    adjacency_matrices = []
    for graph in generated_graphs:
        node_types, edge_types = graph
        A = edge_types.bool().cpu().numpy()
        adjacency_matrices.append(A)

        nx_graph = nx.from_numpy_array(A)
        networkx_graphs.append(nx_graph)

    if dataname=="sbm":
        acc = eval_acc_sbm_graphlist(networkx_graphs, refinement_steps=100, strict=True)
    elif dataname=="planar":
        acc = eval_acc_planar_graphlist(networkx_graphs)
    score_list = np.array(acc).tolist()
    return score_list

def gen_toy_reward_list(generated_graphs):
    networkx_graphs = []
    adjacency_matrices = []
    for graph in generated_graphs:
        node_types, edge_types = graph
        A = edge_types.bool().cpu().numpy()
        adjacency_matrices.append(A)

        nx_graph = nx.from_numpy_array(A)
        networkx_graphs.append(nx_graph)
    score_list = []
    for nx_graph in networkx_graphs:
        if nx.is_connected(nx_graph):
            score_list.append(1)
        else:
            score_list.append(0)
    return np.array(score_list)