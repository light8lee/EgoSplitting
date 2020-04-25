"""Running the clustering tool."""

from param_parser import parameter_parser
from ego_splitter import EgoNetSplitter
import networkx as nx
import numpy as np
import argparse
# from utils import tab_printer, graph_reader, membership_saver

# def main():
#     """
#     Parsing command line parameters, creating EgoNets.
#     Creating a partition of the persona graph. Saving the memberships.
#     """
#     args = parameter_parser()
#     tab_printer(args)
#     graph = graph_reader(args.edge_path)
#     splitter = EgoNetSplitter(args.resolution)
#     splitter.fit(graph)
#     membership_saver(args.output_path, splitter.overlapping_partitions)

from utils import load_dataset, load_feature, load_graph, output_res, evaluate

def validation(weight):
    for name in ("../train/train_0.npz", "../train/train_1.npz", "../train/train_2.npz"):
        graph = load_dataset(name)
        splitter = EgoNetSplitter(1.)
        G = nx.from_scipy_sparse_matrix(graph['A'])
        print("size of G:", len(G.nodes()))
        splitter.fit(G, graph['X'], weight=weight)
        print("size of partitions:", len(splitter.overlapping_partitions))

        output_res(splitter.overlapping_partitions, 'valid.txt')
        pred = np.loadtxt('valid.txt', dtype=np.int32)
        print("actual:", graph['Z'].shape[1])
        print("Score:", evaluate(pred, graph['Z']))

def predict(weight):
    graph_file = '../eval/graph.npz'
    attr_file = '../eval/attr.npz'

    train_graph = load_graph(graph_file)
    feature = load_feature(attr_file)

    G = nx.from_scipy_sparse_matrix(train_graph)
    print("size of G:", len(G.nodes()))
    splitter = EgoNetSplitter(1.)
    splitter.fit(G, feature, weight=weight)
    print("size of partitions:", len(splitter.overlapping_partitions))

    output_res(splitter.overlapping_partitions, f'ego_{weight}.txt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weight")
    args = parser.parse_args()

    # main()
    validation(args.weight)
    predict(args.weight)
