# PointQuality/quality_metrics.py

from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import scipy


class QualityMetrics:
    """
    Compute local quality metrics between original and reconstructed point clouds.

    Usage:
        qm = QualityMetrics(original_points, reconstructed_points)
        metrics = qm.evaluate_metrics()
    """

    def __init__(self, original_points, reconstructed_points, k=15, threshold=20000):
        """
        :param original_points: NumPy array of the original point cloud.
        :param reconstructed_points: NumPy array of the reconstructed point cloud.
        :param k: Number of neighbors for the KNN metric.
        :param threshold: Maximum number of points for CPD due to memory restrictions.
        """
        self.k = k
        self.threshold = threshold
        self.original_points = original_points
        self.reconstructed_points = reconstructed_points

        self.knn = None
        self.cpd = None
        self.knn_individual = None

    def knn_metric(self, visualize=False):
        num_points = len(self.original_points)
        original_tree = KDTree(self.original_points)
        reconstructed_tree = KDTree(self.reconstructed_points)
        original_neighbors = original_tree.query(self.original_points, self.k + 1)[1][:, 1:]
        reconstructed_neighbors = reconstructed_tree.query(self.reconstructed_points, self.k + 1)[1][:, 1:]

        individual_knn = []
        for orig, recon in zip(original_neighbors, reconstructed_neighbors):
            n = len(orig)
            shared = len(set(orig).intersection(set(recon[:n])))
            individual_knn.append(shared / n)
        self.knn_individual = individual_knn
        self.knn = sum(individual_knn) / len(individual_knn)

        if visualize and num_points > 50:
            point_index = 50  # example index for visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # Plot original points and neighbors
            axes[0].scatter(self.original_points[:, 0], self.original_points[:, 1], c='grey')
            axes[0].scatter(self.original_points[point_index, 0], self.original_points[point_index, 1], c='red')
            axes[0].scatter(self.original_points[original_neighbors[point_index], 0],
                            self.original_points[original_neighbors[point_index], 1], c='blue')
            axes[0].set_title('Original Point and Neighbors')
            # Plot reconstructed points and neighbors
            axes[1].scatter(self.reconstructed_points[:, 0], self.reconstructed_points[:, 1], c='grey')
            axes[1].scatter(self.reconstructed_points[point_index, 0], self.reconstructed_points[point_index, 1],
                            c='red')
            axes[1].scatter(self.reconstructed_points[reconstructed_neighbors[point_index], 0],
                            self.reconstructed_points[reconstructed_neighbors[point_index], 1], c='blue')
            axes[1].set_title('Reconstructed Point and Neighbors')
            plt.show()

        return self.knn

    def cpd_metric(self):
        num_points = len(self.original_points)
        if num_points > self.threshold:
            indices = np.random.choice(num_points, self.threshold, replace=False)
            original_points = self.original_points[indices]
            reconstructed_points = self.reconstructed_points[indices]
        else:
            original_points = self.original_points
            reconstructed_points = self.reconstructed_points

        original_distances = pdist(original_points)
        reconstructed_distances = pdist(reconstructed_points)
        correlation, _ = pearsonr(original_distances, reconstructed_distances)
        r_squared = correlation ** 2
        return r_squared

    def compute_distortion(self, original_positions, reconstructed_positions):
        """
        Compute the distortion between the original and reconstructed point clouds.

        :param original_positions: NumPy array of shape (N, 2)
        :param reconstructed_positions: NumPy array of shape (N, 2)
        :return: The median distortion
        """
        distances = [np.linalg.norm(o - r) for o, r in zip(original_positions, reconstructed_positions)]
        return np.median(distances)

    def evaluate_metrics(self, compute_distortion=False):
        knn_result = self.knn_metric()
        cpd_result = self.cpd_metric()
        quality_metrics = {'KNN': knn_result, 'CPD': cpd_result}
        if compute_distortion:
            quality_metrics["Distortion"] = self.compute_distortion(self.original_points, self.reconstructed_points)
        print(quality_metrics)
        return quality_metrics

    def compute_knn_multiple(self, k_values, visualize=False):
        """
        Compute the KNN metric for multiple values of k.

        :param k_values: List of integer k values.
        :param visualize: Whether to visualize the result for each k.
        :return: Dictionary mapping each k to its computed KNN metric.
        """
        knn_results = {}
        for k in k_values:
            self.k = k
            self.knn_metric(visualize=visualize)
            knn_results[k] = self.knn
        return knn_results

    def plot_knn_vs_k(self, knn_results):
        """
        Plot KNN metric vs. k using both linear and log scales.

        :param knn_results: Dictionary mapping k values to KNN metrics.
        """
        sorted_k = sorted(knn_results.keys())
        knn_values = [knn_results[k] for k in sorted_k]
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(sorted_k, knn_values, marker='o')
        axs[0].set_xlabel("k")
        axs[0].set_ylabel("KNN Metric")
        axs[0].set_title("KNN Metric vs. k (Linear Scale)")
        axs[1].plot(sorted_k, knn_values, marker='o')
        axs[1].set_xlabel("k")
        axs[1].set_ylabel("KNN Metric")
        axs[1].set_title("KNN Metric vs. k (Log Scale)")
        axs[1].set_xscale("log")
        plt.tight_layout()
        plt.show()
