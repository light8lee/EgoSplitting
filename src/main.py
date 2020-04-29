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

from utils import load_dataset, load_feature, load_graph, output_res, evaluate, generate_2hop_graph, generate_sim_graph


np.random.seed(2020)

def validation(args):
    for name in ("../train/train_0.npz", "../train/train_1.npz", "../train/train_2.npz"):
        graph = load_dataset(name)
        splitter = EgoNetSplitter(args.resolution)
        if args.sim:
            G = generate_sim_graph(graph['A'], graph['X'], args.threshold)
        else:
            G = nx.from_scipy_sparse_matrix(graph['A'])
        if args.hop2:
            G = generate_2hop_graph(G)
        print("size of G:", len(G.nodes()))
        splitter.fit(G, graph['X'], weight=args.weight)
        print("size of partitions:", len(splitter.overlapping_partitions))

        output_res(splitter.overlapping_partitions, 'valid.txt')
        pred = np.loadtxt('valid.txt', dtype=np.int32)
        print("actual:", graph['Z'].shape[1])
        print("Score:", evaluate(pred, graph['Z']))

def predict(args):
    graph_file = '../eval/graph.npz'
    attr_file = '../eval/attr.npz'

    train_graph = load_graph(graph_file)
    feature = load_feature(attr_file)

    if args.sim:
        G = generate_sim_graph(train_graph, feature, args.threshold)
    else:
        G = nx.from_scipy_sparse_matrix(train_graph)
    if args.hop2:
        G = generate_2hop_graph(G)
    print("size of G:", len(G.nodes()))
    splitter = EgoNetSplitter(args.resolution)
    splitter.fit(G, feature, weight=args.weight)
    print("size of partitions:", len(splitter.overlapping_partitions))

    output_res(splitter.overlapping_partitions, f'ego_{args.weight}{"_H" if args.hop2 else ""}{"_S"+str(args.threshold) if args.sim else ""}_R{args.resolution}.txt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weight")
    parser.add_argument("--hop2", default=False, action="store_true", dest="hop2")
    parser.add_argument("--sim", default=False, action="store_true", dest="sim")
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--resolution", default=1.0, type=float)

    args = parser.parse_args()

    validation(args)
    predict(args)
