# tests/test_quality_metrics.py

import unittest
import numpy as np
from PointQuality import QualityMetrics

class TestQualityMetrics(unittest.TestCase):
    def test_knn_metric(self):
        original_points = np.array([[0, 0], [1, 1], [2, 2]])
        reconstructed_points = np.array([[0, 0], [1, 1.1], [2, 2.1]])
        qm = QualityMetrics(original_points, reconstructed_points, k=1)
        knn = qm.knn_metric()
        self.assertGreaterEqual(knn, 0)
        self.assertLessEqual(knn, 1)

if __name__ == '__main__':
    unittest.main()
