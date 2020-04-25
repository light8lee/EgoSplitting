"""Ego-Splitter class"""

import community
import numpy as np
import networkx as nx
from tqdm import tqdm


class EgoNetSplitter(object):
    """An implementation of `"Ego-Splitting" see:
    https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf
    From the KDD '17 paper "Ego-Splitting Framework: from Non-Overlapping to Overlapping Clusters".
    The tool first creates the egonets of nodes.
    A persona-graph is created which is clustered by the Louvain method.
    The resulting overlapping cluster memberships are stored as a dictionary.
    Args:
        resolution (float): Resolution parameter of Python Louvain. Default 1.0.
    """
    def __init__(self, resolution=1.0):
        self.resolution = resolution

    def _create_egonet(self, node):
        """
        Creating an ego net, extracting personas and partitioning it.

        Args:
            node: Node ID for egonet (ego node).
        """
        ego_net_minus_ego = self.graph.subgraph(self.graph.neighbors(node))
        components = {i: n for i, n in enumerate(nx.connected_components(ego_net_minus_ego))}
        new_mapping = {}
        personalities = []
        for k, v in components.items():
            personalities.append(self.index)
            for other_node in v:
                new_mapping[other_node] = self.index
            self.index = self.index+1
        self.components[node] = new_mapping
        self.personalities[node] = personalities

    def _create_egonets(self):
        """
        Creating an egonet for each node.
        """
        self.components = {}
        self.personalities = {}
        self.index = 0
        print("Creating egonets.")
        for node in tqdm(self.graph.nodes()):
            self._create_egonet(node)

    def _map_personalities(self):
        """
        Mapping the personas to new nodes.
        """
        self.personality_map = {p: n for n in self.graph.nodes() for p in self.personalities[n]}

    def _get_new_edge_ids(self, edge):
        """
        Getting the new edge identifiers.
        Args:
            edge: Edge being mapped to the new identifiers.
        """
        return (self.components[edge[0]][edge[1]], self.components[edge[1]][edge[0]])

    def _create_persona_graph(self):
        """
        Create a persona graph using the egonet components.
        """
        print("Creating the persona graph.")
        self.persona_graph_edges = [self._get_new_edge_ids(e) for e in tqdm(self.graph.edges())]
        self.persona_graph = nx.from_edgelist(self.persona_graph_edges)

    def _create_persona_feature(self, feature):
        persona_size = len(self.persona_graph.nodes())
        persona_feature_ids = [self.personality_map[n] for n in range(persona_size)]
        persona_feature = feature[persona_feature_ids]
        return persona_feature

    def _cosine_weight(self, G, feature):
        row, col = zip(*G.edges())
        row = list(row)
        col = list(col)
        inner_prod = np.array(feature[row].multiply(feature[col]).sum(axis=1)).reshape(-1)
        row_modulus = np.array(np.sqrt(feature[row].power(2).sum(axis=1))).reshape(-1)
        col_modulus = np.array(np.sqrt(feature[col].power(2).sum(axis=1))).reshape(-1)
        cosine = (inner_prod + 2) / (row_modulus * col_modulus + 1)
        G.add_weighted_edges_from(list(zip(row, col, cosine)))

    def _eucli_weight(self, G, feature):
        row, col = zip(*G.edges())
        row = list(row)
        col = list(col)
        dist = np.array((feature[row]-feature[col]).power(2).sum(axis=1)).reshape(-1)
        dist = 1 / (dist + 0.1)
        G.add_weighted_edges_from(list(zip(row, col, dist)))
    
    def _comb_weight(self, G, feature):
        row, col = zip(*G.edges())
        row = list(row)
        col = list(col)
        dist = np.array((feature[row]-feature[col]).power(2).sum(axis=1)).reshape(-1)
        dist = 1 / (dist + 0.1)
        inner_prod = np.array(feature[row].multiply(feature[col]).sum(axis=1)).reshape(-1)
        row_modulus = np.array(np.sqrt(feature[row].power(2).sum(axis=1))).reshape(-1)
        col_modulus = np.array(np.sqrt(feature[col].power(2).sum(axis=1))).reshape(-1)
        cosine = (inner_prod + 2) / (row_modulus * col_modulus + 1)
        weight = dist + cosine
        G.add_weighted_edges_from(list(zip(row, col, weight)))

    def _create_partitions(self):
        """
        Creating a non-overlapping clustering of nodes in the persona graph.
        """
        print("Clustering the persona graph.")
        self.partitions = community.best_partition(self.persona_graph, resolution=self.resolution)
        self.overlapping_partitions = {node: [] for node in self.graph.nodes()}
        for node, membership in self.partitions.items():
            self.overlapping_partitions[self.personality_map[node]].append(membership)

    def fit(self, graph, feature, weight='none'):
        """
        Fitting an Ego-Splitter clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self.graph = graph
        self._create_egonets()
        self._map_personalities()
        self._create_persona_graph()
        if weight == 'cosine':
            persona_feature = self._create_persona_feature(feature)
            self._cosine_weight(self.persona_graph, persona_feature)
        elif weight == 'eucli':
            persona_feature = self._create_persona_feature(feature)
            self._eucli_weight(self.persona_graph, persona_feature)
        elif weight == 'comb':
            persona_feature = self._create_persona_feature(feature)
            self._comb_weight(self.persona_graph, persona_feature)
        elif weight == 'none':
            pass
        else:
            raise ValueError("No such weight:", weight)
        self._create_partitions()

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.
        Return types:
            * **memberships** *(dictionary of lists)* - Cluster memberships.
        """
        return self.overlapping_partitions
