"""General data treatment utilities."""

import json
# import torch
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from texttable import Texttable
import scipy.sparse as sp

def evaluate(result, label):
    result_sum = sum(result)
    result_index = np.argwhere(result_sum == 1)
    result = np.delete(result, result_index, axis=1)
    predict_label = np.zeros(label.shape)
    for i in range(result.shape[1]):
        tmp1 = np.dot(result[:, i].T, label)
        tmp2 = (tmp1 == max(tmp1)).astype(float)
        if np.sum(tmp2) > 1:
            index = tmp2.argmax()
            tmp2 = np.zeros(tmp1.shape)
            tmp2[index] = 1
        predict_label = predict_label + np.tile(tmp2, (result.shape[0], 1)) * np.tile(result[:, i],
                                                                                      (label.shape[1], 1)).T
    predict_label = (predict_label > 0).astype(float)
    tmp3 = np.sum(label, 1)
    p_label = np.tile((tmp3 > 0).astype(float), (label.shape[1], 1)).T * predict_label
    R = np.sum(np.sum(predict_label * label)) / np.sum(np.sum(label))
    P = np.sum(np.sum(predict_label * label)) / np.sum(np.sum(p_label))
    F_score = 2 * R * P / (R + P)
    if str(F_score) == 'nan':
        F_score = 0
    return F_score


def load_graph(file_name):
    graph = None
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['data'], loader['indices'],
                           loader['indptr']), shape=loader['shape'])
        # Remove self-loops
        A = A.tolil()
        A.setdiag(0)
        graph = A.tocsr()
    return graph


def load_feature(file_name):
    attr = None
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        attr = sp.csr_matrix((loader['data'], loader['indices'],
                           loader['indptr']), shape=loader['shape'])
        attr = attr.tocsr()
    return attr


def load_dataset(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'Z' : The community labels in sparse matrix format
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_matrix.data'], loader['adj_matrix.indices'],
                           loader['adj_matrix.indptr']), shape=loader['adj_matrix.shape'])

        if 'attr_matrix.data' in loader.keys():
            X = sp.csr_matrix((loader['attr_matrix.data'], loader['attr_matrix.indices'],
                               loader['attr_matrix.indptr']), shape=loader['attr_matrix.shape'])
        else:
            X = None

        Z = sp.csr_matrix((loader['labels.data'], loader['labels.indices'],
                           loader['labels.indptr']), shape=loader['labels.shape'])

        # Remove self-loops
        A = A.tolil()
        A.setdiag(0)
        A = A.tocsr()

        # Convert label matrix to numpy
        if sp.issparse(Z):
            Z = Z.toarray().astype(np.float32)

        graph = {
            'A': A,
            'X': X,
            'Z': Z
        }

        node_names = loader.get('node_names')
        if node_names is not None:
            node_names = node_names.tolist()
            graph['node_names'] = node_names

        attr_names = loader.get('attr_names')
        if attr_names is not None:
            attr_names = attr_names.tolist()
            graph['attr_names'] = attr_names

        class_names = loader.get('class_names')
        if class_names is not None:
            class_names = class_names.tolist()
            graph['class_names'] = class_names

        return graph


def generate_2hop_graph(graph_1hop):
    A = nx.adjacency_matrix(graph_1hop)
    A2 = A * A
    A2 += A
    A2[A2>0] = 1
    A2 = A2.tolil()
    A2.setdiag(0)
    graph_2hop = nx.from_scipy_sparse_matrix(A2)
    return graph_2hop


def generate_sim_graph(adj, feature, threshold=0.5):
    # feature = feature.tolil()
    inner_prod = np.array(feature * feature.transpose())
    modulus = np.array(np.sqrt(feature.power(2).sum(axis=1) + 1e-9))
    modulus = modulus * modulus.transpose()
    cos_sim = inner_prod / modulus
    adj[cos_sim>threshold] = 1
    adj.setdiag(0)
    graph = nx.from_scipy_sparse_matrix(adj)
    return graph


def output_res(partition, outfile_name):
    #排序
    key_list = list(partition.keys())
    key_list.sort()
    max_p = 0
    with open(outfile_name, 'w') as ofs:
        for key in key_list:
            if not partition[key]:
                continue
            curr_p = max(partition[key])
            max_p = max(max_p, curr_p)
        max_p += 1
        print('max_p:', max_p)
        for key in key_list:
            vec = [0] * max_p
            for p in partition[key]:
                vec[p] = 1
            line = ' '.join([str(x) for x in vec])
            ofs.write(line + '\n')
            

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def membership_saver(path, memberships):
    """
    Saving the membership dictionary as a JSON.
    :param path: Output path.
    :param memberships: Membership dictionary.
    """
    with open(path, "w") as f:
        json.dump(memberships, f)