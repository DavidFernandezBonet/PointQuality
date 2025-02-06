# examples/example_usage.py

import numpy as np
import pandas as pd
from PointQuality import QualityMetrics, GTA_Quality_Metrics

# Create dummy data for demonstration
original_points = np.random.rand(100, 2)
reconstructed_points = original_points + np.random.normal(scale=0.05, size=(100, 2))

# Evaluate local quality metrics
qm = QualityMetrics(original_points, reconstructed_points)
local_metrics = qm.evaluate_metrics()

# Create a dummy edge list for GTA quality metrics
edge_list = pd.DataFrame({
    'source': np.random.randint(0, 100, 200),
    'target': np.random.randint(0, 100, 200)
})

# Evaluate global (GTA) quality metrics
gta_qm = GTA_Quality_Metrics(edge_list, reconstructed_points)
global_metrics = gta_qm.evaluate_metrics()
