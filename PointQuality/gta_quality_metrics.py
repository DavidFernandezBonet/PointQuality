# PointQuality/gta_quality_metrics.py

from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import igraph as ig


class GTA_Quality_Metrics:
    """
    Compute global quality metrics based on a ground truth graph structure.

    :param edge_list: DataFrame with columns 'source' and 'target' representing edges.
    :param reconstructed_points: NumPy array of reconstructed points.
    :param k: Number of nearest neighbors for the GTA KNN metric.
    :param threshold: Maximum number of nodes for CPD calculations.
    """

    def __init__(self, edge_list, reconstructed_points, k=15, threshold=1000):
        self.k = k
        self.threshold = threshold
        self.edge_list = edge_list[['source', 'target']]
        self.reconstructed_points = reconstructed_points
        self.gta_knn_metric = None
        self.gta_knn_individual = None

    def _extract_neighbors_from_edge_list(self):
        """
        Extract neighbors from the edge list.
        """
        if isinstance(self.edge_list, pd.DataFrame):
            edges = [tuple(x) for x in self.edge_list.to_numpy()]
        else:
            edges = self.edge_list

        neighbors_dict = defaultdict(set)
        for u, v in edges:
            neighbors_dict[u].add(v)
            neighbors_dict[v].add(u)
        max_index = max(neighbors_dict.keys())
        neighbors_list = [list(neighbors_dict.get(i, set())) for i in range(max_index + 1)]
        return neighbors_list

    def get_gta_knn(self, visualize=False):
        original_neighbors = self._extract_neighbors_from_edge_list()
        reconstructed_tree = KDTree(self.reconstructed_points)
        reconstructed_neighbors = reconstructed_tree.query(self.reconstructed_points, self.k + 1)[1][:, 1:]
        individual_gta_knn = []
        for orig, recon in zip(original_neighbors, reconstructed_neighbors):
            n = len(orig)
            shared = len(set(orig).intersection(set(recon[:n])))
            individual_gta_knn.append(shared / n if n > 0 else 0)
        gta_knn = np.mean(individual_gta_knn)
        if visualize:
            plt.boxplot(individual_gta_knn)
            plt.title("Distribution of Individual GTA_KNN")
            plt.ylabel("Shared Neighbors Fraction")
            plt.savefig("individual_gta_knn_boxplot.png")
        self.gta_knn_individual = individual_gta_knn
        return individual_gta_knn, gta_knn

    def get_gta_cpd(self):
        ig_graph = ig.Graph.TupleList(self.edge_list.to_records(index=False), directed=False)
        num_points = ig_graph.vcount()
        if num_points > self.threshold:
            indices = np.random.choice(num_points, self.threshold, replace=False)
            sampled_points = self.reconstructed_points[indices]
        else:
            sampled_points = self.reconstructed_points
            indices = range(num_points)

        # Map indices to the order of igraph nodes (adjust as needed)
        indices_as_indices = [ig_graph.vs.find(name=str(n)).index for n in indices]
        graph_distances = ig_graph.shortest_paths_dijkstra(source=indices_as_indices, target=indices_as_indices)
        graph_distances_flat = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                graph_distances_flat.append(graph_distances[i][j])
        graph_distances_flat = np.array(graph_distances_flat)
        reconstructed_distances = pdist(sampled_points)
        correlation, _ = pearsonr(graph_distances_flat, reconstructed_distances)
        r_squared = correlation ** 2
        return r_squared

    def evaluate_metrics(self):
        self.gta_knn_individual, gta_knn = self.get_gta_knn()
        gta_cpd = self.get_gta_cpd()
        quality_metrics = {'GTA_KNN': gta_knn, 'GTA_CPD': gta_cpd}
        print(quality_metrics)
        return quality_metrics
